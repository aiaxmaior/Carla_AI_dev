"""
DataConsolidator.py
Centralized data collection and formatting for predictor indices.
Consolidates telemetry from all simulator modules into a unified stream.
"""

# ============================================================================
# PERF CHECK (file-level):
# ============================================================================
# [X] | Role: Data collection hub (consolidates telemetry from all modules)
# [X] | Hot-path functions: collect_frame_data() if called every tick
# [X] |- Heavy allocs in hot path? YES - large dict creation + deque append
# [X] |- pandas/pyarrow/json/disk/net in hot path? pandas used for analysis only
# [ ] | Graphics here? No
# [X] | Data produced (tick schema?): Frame dict per tick
# [X] | Storage (Parquet/Arrow/CSV/none): In-memory deque (1000 frames max)
# [X] | Queue/buffer used?: YES - deque (efficient) + optional queue.Queue for streaming
# [X] | Session-aware? Should add session_id
# [X] | Debug-only heavy features?: Statistics tracking (could be gated)
# Top 3 perf risks:
# 1. [PERF_HOT] collect_frame_data() creates comprehensive dict every tick if called
# 2. [PERF_OK] deque with maxlen is efficient (O(1) append/popleft)
# 3. [PERF_SPLIT] Thread safety lock on every access - may cause contention
# 4. [PERF_SPLIT] Multiple data collection modules - consolidate with DataIngestion?
# ============================================================================

import time
import json
import logging
import numpy as np
import pandas as pd
from collections import deque, defaultdict
from typing import Dict, Any, Optional, List, Tuple
import threading
import queue
import carla

class DataConsolidator:
    """
    Unified data collection hub for Q-DRIVE Cortex.
    Gathers, normalizes, and streams all relevant telemetry for ML models.
    """
    
    def __init__(self, buffer_size: int = 1000, streaming: bool = True):
        """
        Initialize the data consolidator.
        
        Args:
            buffer_size: Number of frames to keep in rolling buffer
            streaming: Whether to enable real-time streaming via queue
        """
        self.buffer_size = buffer_size
        self.streaming = streaming
        
        # Rolling buffers for time-series data
        self.telemetry_buffer = deque(maxlen=buffer_size)
        self.event_buffer = deque(maxlen=buffer_size)
        
        # Current frame data (gets updated each tick)
        self.current_frame_data = {}
        
        # Streaming queue for real-time consumers
        self.stream_queue = queue.Queue() if streaming else None
        
        # Statistics tracking
        self.stats = defaultdict(lambda: {
            'count': 0,
            'sum': 0.0,
            'sum_sq': 0.0,
            'min': float('inf'),
            'max': float('-inf')
        })
        
        # Feature engineering cache
        self.derived_features = {}
        
        # Thread safety
        self.lock = threading.Lock()
        
        logging.info(f"DataConsolidator initialized (buffer_size={buffer_size}, streaming={streaming})")
    
    def collect_frame_data(self, 
                           world_obj,
                           controller,
                           mvd_extractor,
                           world_snapshot,
                           display_fps: float) -> Dict[str, Any]:
        """
        Main collection method - gathers all data for current frame.
        Call this once per tick from Main.py.
        """
        timestamp = time.time()
        
        # Bail if player not ready
        if not world_obj or not world_obj.player or not world_obj.player.is_alive:
            return {}
        
        player = world_obj.player
        
        # === 1. VEHICLE DYNAMICS ===
        velocity = player.get_velocity()
        acceleration = player.get_acceleration()
        angular_velocity = player.get_angular_velocity()
        transform = player.get_transform()
        control = player.get_control()
        physics = player.get_physics_control()
        
        speed_ms = velocity.length()
        speed_kmh = 3.6 * speed_ms
        speed_mph = 2.237 * speed_ms
        
        # Compute lateral/longitudinal components
        forward_vec = transform.get_forward_vector()
        right_vec = transform.get_right_vector()
        
        vel_forward = velocity.x * forward_vec.x + velocity.y * forward_vec.y
        vel_lateral = velocity.x * right_vec.x + velocity.y * right_vec.y
        
        accel_forward = acceleration.x * forward_vec.x + acceleration.y * forward_vec.y
        accel_lateral = acceleration.x * right_vec.x + acceleration.y * right_vec.y
        
        # === 2. CONTROL INPUTS ===
        control_data = controller.get_datalog() if hasattr(controller, 'get_datalog') else {}
        
        raw_steer = control_data.get('raw_inputs', {}).get('steer', 0.0)
        raw_throttle = control_data.get('raw_inputs', {}).get('throttle', 0.0)
        raw_brake = control_data.get('raw_inputs', {}).get('brake', 0.0)
        
        # Normalized values actually sent to vehicle
        norm_steer = control_data.get('normalized_outputs', {}).get('steer', 0.0)
        norm_throttle = control_data.get('normalized_outputs', {}).get('throttle', 0.0)
        norm_brake = control_data.get('normalized_outputs', {}).get('brake', 0.0)
        
        # === 3. SENSOR DATA ===
        collision_data = world_obj.get_collision_data_and_reset() if hasattr(world_obj, 'get_collision_data_and_reset') else {}
        
        lane_invasion_state = None
        if world_obj.lane_invasion_sensor_instance:
            lane_invasion_state = world_obj.lane_invasion_sensor_instance.get_violation_state()
        
        lane_change_state = None
        if hasattr(world_obj, 'lane_manager'):
            lane_change_state = world_obj.lane_manager.get_lane_change_state()
        
        # === 4. MVD SCORING ===
        mvd_metrics = mvd_extractor.get_mvd_datalog_metrics() if mvd_extractor else {}
        overall_score = mvd_extractor.get_overall_mvd_score() if mvd_extractor else 100.0
        
        standardized = {}
        if mvd_extractor and hasattr(mvd_extractor, 'get_standardized_indices'):
            standardized = mvd_extractor.get_standardized_indices()
        
        # === 5. TRAFFIC & ENVIRONMENT ===
        # Get nearby vehicles
        nearby_vehicles = self._get_nearby_actors(world_obj, player, actor_type='vehicle', radius=50.0)
        nearby_pedestrians = self._get_nearby_actors(world_obj, player, actor_type='walker.pedestrian', radius=30.0)
        
        # Weather conditions
        weather = world_obj.world.get_weather() if hasattr(world_obj.world, 'get_weather') else None
        
        # === 6. VEHICLE STATE FLAGS ===
        vehicle_state = control_data.get('vehicle_state', {})
        
        # === 7. COMPILE FRAME DATA ===
        frame_data = {
            # Metadata
            'timestamp': timestamp,
            'frame': world_snapshot.frame if world_snapshot else 0,
            'sim_time': world_snapshot.timestamp.elapsed_seconds if world_snapshot else 0.0,
            'display_fps': display_fps,
            
            # Position & Orientation
            'position_x': transform.location.x,
            'position_y': transform.location.y,
            'position_z': transform.location.z,
            'rotation_pitch': transform.rotation.pitch,
            'rotation_yaw': transform.rotation.yaw,
            'rotation_roll': transform.rotation.roll,
            
            # Velocities
            'speed_ms': speed_ms,
            'speed_kmh': speed_kmh,
            'speed_mph': speed_mph,
            'velocity_x': velocity.x,
            'velocity_y': velocity.y,
            'velocity_z': velocity.z,
            'velocity_forward': vel_forward,
            'velocity_lateral': vel_lateral,
            
            # Accelerations
            'acceleration_x': acceleration.x,
            'acceleration_y': acceleration.y,
            'acceleration_z': acceleration.z,
            'acceleration_forward': accel_forward,
            'acceleration_lateral': accel_lateral,
            'acceleration_magnitude': acceleration.length(),
            
            # Angular Motion
            'angular_velocity_x': angular_velocity.x,
            'angular_velocity_y': angular_velocity.y,
            'angular_velocity_z': angular_velocity.z,  # Yaw rate
            
            # Control Inputs (Raw)
            'input_steer_raw': raw_steer,
            'input_throttle_raw': raw_throttle,
            'input_brake_raw': raw_brake,
            
            # Control Outputs (Normalized)
            'control_steer': norm_steer,
            'control_throttle': norm_throttle,
            'control_brake': norm_brake,
            'control_handbrake': vehicle_state.get('handbrake', False),
            'control_reverse': vehicle_state.get('reverse', False),
            'control_gear': vehicle_state.get('gear', 0),
            'control_manual_gear': vehicle_state.get('manual_gear_shift', False),
            
            # Indicators & Safety
            'blinker_state': vehicle_state.get('blinker_state', 0),
            'seatbelt_fastened': vehicle_state.get('seatbelt', False),
            'autopilot_enabled': vehicle_state.get('autopilot', False),
            
            # Collision Data
            'collision_occurred': collision_data.get('collided', False),
            'collision_intensity': collision_data.get('intensity', 0.0),
            'collision_actor_type': collision_data.get('other_actor_type', 'none'),
            'collision_relative_speed': collision_data.get('relative_speed_kmh', 0.0),
            
            # Lane Keeping
            'lane_invasion_active': lane_invasion_state is not None and lane_invasion_state != 'none',
            'lane_invasion_type': lane_invasion_state if lane_invasion_state else 'none',
            'lane_change_state': lane_change_state if lane_change_state else 'none',
            
            # MVD Scores
            'mvd_overall_score': overall_score,
            'mvd_mbi': standardized.get('mbi_0_1', 1.0),
            'mvd_lmi': standardized.get('lmi_0_1', 1.0),
            'mvd_collision_score': mvd_metrics.get('Collision Avoidance Score', 100.0),
            'mvd_lane_score': mvd_metrics.get('Lane Discipline Score', 100.0),
            'mvd_signal_score': mvd_metrics.get('Turn Signal Score', 100.0),
            'mvd_speed_score': mvd_metrics.get('Speed Compliance Score', 100.0),
            
            # Traffic Context
            'nearby_vehicles_count': len(nearby_vehicles),
            'nearest_vehicle_distance': nearby_vehicles[0]['distance'] if nearby_vehicles else 999.0,
            'nearest_vehicle_relative_speed': nearby_vehicles[0]['relative_speed'] if nearby_vehicles else 0.0,
            'nearby_pedestrians_count': len(nearby_pedestrians),
            'nearest_pedestrian_distance': nearby_pedestrians[0]['distance'] if nearby_pedestrians else 999.0,
            
            # Weather
            'weather_cloudiness': weather.cloudiness if weather else 0.0,
            'weather_precipitation': weather.precipitation if weather else 0.0,
            'weather_sun_altitude': weather.sun_altitude_angle if weather else 45.0,
            'weather_fog_density': weather.fog_density if weather else 0.0,
            'weather_wetness': weather.wetness if weather else 0.0,
        }
        
        # === 8. COMPUTE DERIVED FEATURES ===
        self._compute_derived_features(frame_data)
        
        # === 9. UPDATE BUFFERS & STATS ===
        with self.lock:
            self.current_frame_data = frame_data
            self.telemetry_buffer.append(frame_data)
            self._update_statistics(frame_data)
            
            # Stream to queue if enabled
            if self.streaming and self.stream_queue:
                try:
                    self.stream_queue.put_nowait(frame_data.copy())
                except queue.Full:
                    pass  # Drop frame if consumer is slow
        
        return frame_data
    
    def _get_nearby_actors(self, world_obj, player, actor_type: str, radius: float) -> List[Dict]:
        """Get nearby actors of specified type within radius."""
        nearby = []
        player_loc = player.get_location()
        
        try:
            actors = world_obj.world.get_actors()
            filtered = actors.filter(f'*{actor_type}*')
            
            for actor in filtered:
                if actor.id == player.id:
                    continue
                    
                actor_loc = actor.get_location()
                distance = player_loc.distance(actor_loc)
                
                if distance <= radius:
                    # Calculate relative velocity
                    actor_vel = actor.get_velocity()
                    player_vel = player.get_velocity()
                    rel_vel = (actor_vel - player_vel).length()
                    
                    nearby.append({
                        'id': actor.id,
                        'distance': distance,
                        'relative_speed': rel_vel * 3.6,  # Convert to km/h
                        'location': (actor_loc.x, actor_loc.y, actor_loc.z)
                    })
            
            # Sort by distance
            nearby.sort(key=lambda x: x['distance'])
            
        except Exception as e:
            logging.debug(f"Error getting nearby actors: {e}")
        
        return nearby
    
    def _compute_derived_features(self, frame_data: Dict):
        """Compute additional features from raw data."""
        # Time to collision (if approaching nearest vehicle)
        ttc = float('inf')
        if frame_data['nearest_vehicle_distance'] < 999.0:
            rel_speed_ms = frame_data['nearest_vehicle_relative_speed'] / 3.6
            if rel_speed_ms > 0.5:  # Approaching
                ttc = frame_data['nearest_vehicle_distance'] / rel_speed_ms
        
        frame_data['time_to_collision'] = min(ttc, 999.0)
        
        # Jerk (rate of acceleration change) - needs previous frame
        if len(self.telemetry_buffer) > 0:
            prev = self.telemetry_buffer[-1]
            dt = frame_data['timestamp'] - prev['timestamp']
            if dt > 0:
                jerk_x = (frame_data['acceleration_x'] - prev['acceleration_x']) / dt
                jerk_y = (frame_data['acceleration_y'] - prev['acceleration_y']) / dt
                jerk_z = (frame_data['acceleration_z'] - prev['acceleration_z']) / dt
                frame_data['jerk_magnitude'] = (jerk_x**2 + jerk_y**2 + jerk_z**2)**0.5
            else:
                frame_data['jerk_magnitude'] = 0.0
        else:
            frame_data['jerk_magnitude'] = 0.0
        
        # Steering rate (degrees per second)
        if len(self.telemetry_buffer) > 0:
            prev = self.telemetry_buffer[-1]
            dt = frame_data['timestamp'] - prev['timestamp']
            if dt > 0:
                steer_rate = (frame_data['control_steer'] - prev['control_steer']) / dt
                frame_data['steering_rate'] = abs(steer_rate)
            else:
                frame_data['steering_rate'] = 0.0
        else:
            frame_data['steering_rate'] = 0.0
        
        # Slip angle approximation
        if frame_data['speed_ms'] > 1.0:
            slip_angle_rad = np.arctan2(frame_data['velocity_lateral'], 
                                        abs(frame_data['velocity_forward']))
            frame_data['slip_angle_deg'] = np.degrees(slip_angle_rad)
        else:
            frame_data['slip_angle_deg'] = 0.0
        
        # G-force
        frame_data['g_force_lateral'] = frame_data['acceleration_lateral'] / 9.81
        frame_data['g_force_longitudinal'] = frame_data['acceleration_forward'] / 9.81
        frame_data['g_force_total'] = frame_data['acceleration_magnitude'] / 9.81
    
    def _update_statistics(self, frame_data: Dict):
        """Update running statistics for each metric."""
        for key, value in frame_data.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                stat = self.stats[key]
                stat['count'] += 1
                stat['sum'] += value
                stat['sum_sq'] += value**2
                stat['min'] = min(stat['min'], value)
                stat['max'] = max(stat['max'], value)
    
    def get_statistics(self) -> Dict[str, Dict]:
        """Get computed statistics for all metrics."""
        result = {}
        with self.lock:
            for key, stat in self.stats.items():
                if stat['count'] > 0:
                    mean = stat['sum'] / stat['count']
                    variance = (stat['sum_sq'] / stat['count']) - mean**2
                    std = variance**0.5 if variance > 0 else 0.0
                    
                    result[key] = {
                        'count': stat['count'],
                        'mean': mean,
                        'std': std,
                        'min': stat['min'],
                        'max': stat['max']
                    }
        return result
    
    def get_windowed_data(self, window_size: int = 100) -> pd.DataFrame:
        """Get recent data as a DataFrame for analysis."""
        with self.lock:
            if len(self.telemetry_buffer) == 0:
                return pd.DataFrame()
            
            # Get last N frames
            data = list(self.telemetry_buffer)[-window_size:]
            return pd.DataFrame(data)
    
    def export_session_data(self, filepath: str = None) -> str:
        """Export all collected data to CSV."""
        if filepath is None:
            filepath = f"session_data_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        
        df = self.get_windowed_data(window_size=self.buffer_size)
        if not df.empty:
            df.to_csv(filepath, index=False)
            logging.info(f"Exported {len(df)} frames to {filepath}")
        else:
            logging.warning("No data to export")
        
        return filepath
    
    def get_ml_features(self, lookback: int = 10) -> np.ndarray:
        """
        Get feature vector for ML models with temporal context.
        Returns the last `lookback` frames as a flattened feature array.
        """
        # Key features for prediction models
        feature_cols = [
            'speed_ms', 'acceleration_magnitude', 'jerk_magnitude',
            'control_steer', 'control_throttle', 'control_brake',
            'steering_rate', 'slip_angle_deg',
            'g_force_lateral', 'g_force_longitudinal',
            'nearest_vehicle_distance', 'time_to_collision',
            'lane_invasion_active', 'mvd_overall_score'
        ]
        
        with self.lock:
            if len(self.telemetry_buffer) < lookback:
                return np.zeros(len(feature_cols) * lookback)
            
            # Extract features from last N frames
            features = []
            for frame in list(self.telemetry_buffer)[-lookback:]:
                frame_features = []
                for col in feature_cols:
                    val = frame.get(col, 0.0)
                    # Convert booleans to float
                    if isinstance(val, bool):
                        val = float(val)
                    frame_features.append(val)
                features.extend(frame_features)
            
            return np.array(features, dtype=np.float32)
    
    def stream_generator(self):
        """Generator for streaming data to consumers."""
        while True:
            if self.stream_queue:
                try:
                    data = self.stream_queue.get(timeout=1.0)
                    yield data
                except queue.Empty:
                    continue
            else:
                time.sleep(0.1)