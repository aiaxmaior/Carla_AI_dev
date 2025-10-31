"""

-----------------------------
### DataIngestion.py
Author: Arjun Joshi
Recent Update: 10.17.2025
-----------------------------
Description:
Consolidate data ingestion and real-time violation flagging using pandas DataFrame.
This module defines a DataIngestion class that uses a pandas DataFrame to cache session data in
memory. It flags violations and events in real-time as frames are logged.

"""

import pandas as pd
import os
import time
import logging
import math
from typing import Dict, Any, Optional

try:
    import carla
except ImportError:
    carla = None  # Allow imports to work even without CARLA installed

class DataIngestion(object):
    """
    A stateful class using a pandas DataFrame to cache session data in-memory.
    Flags violations and events in real-time as frames are logged.
    """

    def __init__(self, buffer_size: int = 100):
        """
        Initialize with an empty DataFrame and schema.

        Args:
            buffer_size: Number of frames to buffer before batch inserting (default: 100)
        """
        self._session_df = pd.DataFrame()
        self._frame_count = 0
        self._buffer_size = buffer_size
        self._frame_buffer = []  # Buffer for batch inserts

        # Predictive indices (latest snapshot)
        self._last_predictive: Dict[str, Any] = {
            "p_lane_violation": 0.0,
            "tlc_s": 99.0,
            "dist_to_lane_boundary_m": 0.0,
            "p_collision": 0.0,
            "ttc_s": 99.0,
            "a_required_brake_mps2": 0.0,
            "p_harsh_operation": 0.0,
            "dominant_axis": "none",
        }

        logging.info(f"DataIngestion initialized with pandas DataFrame backend (buffer_size={buffer_size}).")

    def set_predictive_indices(self, preds: Dict[str, Any]) -> None:
        """Store the most recent predictive indices."""
        if not isinstance(preds, dict):
            return
        for k in list(self._last_predictive.keys()):
            if k in preds:
                self._last_predictive[k] = preds[k]

    def log_frame(
        self,
        world_obj,
        metrics: Dict[str, Any],
        mvd_data: Dict[str, Any] = None,
        control_data: Dict[str, Any] = None,
        vehicle_state: Dict[str, Any] = None,
        raw_inputs: Dict[str, Any] = None,
        norm_outputs: Dict[str, Any] = None,
        ackermann_targets: Dict[str, Any] = None,
        static_params: Dict[str, Any] = None,
        carla_map = None
    ):
        """
        Logs a frame and flags violations in real-time.
        Appends to the in-memory DataFrame.

        Args:
            world_obj: CARLA world object containing player, sensors, etc.
            metrics: Core metrics dictionary (frame, timestamp, speed, etc.)
            mvd_data: MVD scoring data (collision scores, lane scores, etc.)
            control_data: Control input data (input_mode, etc.)
            vehicle_state: Vehicle state data (gear, handbrake, blinker, etc.)
            raw_inputs: Raw controller inputs (steer, throttle, brake)
            norm_outputs: Normalized outputs sent to CARLA
            ackermann_targets: Ackermann controller settings and targets
            static_params: Static configuration parameters (deadzones, etc.)
            carla_map: CARLA map object for waypoint queries
        """
        try:
            # Initialize defaults for optional parameters
            mvd_data = mvd_data or {}
            control_data = control_data or {}
            vehicle_state = vehicle_state or {}
            raw_inputs = raw_inputs or {}
            norm_outputs = norm_outputs or {}
            ackermann_targets = ackermann_targets or {}
            static_params = static_params or {}

            player = world_obj.player
            if not player or not player.is_alive:
                return

            # Extract basic vehicle kinematics
            velocity = player.get_velocity()
            accel = player.get_acceleration()
            ang_vel = player.get_angular_velocity()
            loc = player.get_location()
            rot = player.get_transform().rotation

            # Extract complex data using helper methods
            environmental_data = self._extract_environmental_context(player, world_obj, carla_map)
            physics_data = self._extract_vehicle_physics(player)

            # --- Create a single, flat dictionary for this frame ---
            frame_record = {
                # Core Metrics
                'frame': metrics.get('frame'),
                'timestamp': metrics.get('timestamp'),
                
                #######################################################################
                # --- MVD SCORES ---------#
                #######################################################################
                'scores':{
                    'scoring_spacer':'||',
                    'overall_mvd_score': mvd_data.get('overall_score'),
                    'index_mbi_0_1': mvd_data.get('index_mbi_0_1'),
                    'index_lmi_0_1': mvd_data.get('index_lmi_0_1'),
                    'score_collision_raw': mvd_data.get('score_collision_raw'),
                    'score_harsh_driving_raw': mvd_data.get('score_harsh_driving_raw'),
                    'score_lane_raw': mvd_data.get('score_lane_raw'),
                    "PSS_ProactiveSafety":   float(mvd_data.get('_collision_avoidance_score', 0.0)),
                    "LDS_LaneDiscipline":    float(mvd_data.get('_lane_management_score', 0.0)),
                    "DSS_DrivingSmoothness": float(mvd_data.get('_harsh_driving_score', 0.0)),  
                },
                #######################################################################
                # --- PREDICTIVE INDICES (Real-time risk assessment) -----------------#
                #######################################################################
                'predictive_indices': {
                    'predictive_spacer': '||',
                    'p_lane_violation': float(self._last_predictive.get("p_lane_violation", 0.0)),
                    'tlc_s': float(self._last_predictive.get("tlc_s", 99.0)),
                    'dist_to_lane_boundary_m': float(self._last_predictive.get("dist_to_lane_boundary_m", 0.0)),
                    'p_collision': float(self._last_predictive.get("p_collision", 0.0)),
                    'ttc_s': float(self._last_predictive.get("ttc_s", 99.0)),
                    'a_required_brake_mps2': float(self._last_predictive.get("a_required_brake_mps2", 0.0)),
                    'p_harsh_operation': float(self._last_predictive.get("p_harsh_operation", 0.0)),
                    'dominant_axis': str(self._last_predictive.get("dominant_axis", "none")),
                },

                #######################################################################
                # --- EVENT ---------------- STATES ----------------------------------#
                #######################################################################
                'event_states':{
                    'event_spacer':'||',
                    'lane_violation_state': metrics.get('lane_violation_state').name if metrics.get('lane_violation_state') else 'NORMAL',
                    'lane_change_state': metrics.get('lane_change_state').name if metrics.get('lane_change_state') else 'NORMAL',
                    'collided': metrics.get('collision_data', {}).get('collided', False),
                    'collision_intensity': metrics.get('collision_data', {}).get('intensity'),
                    'collision_actor_type': metrics.get('collision_data', {}).get('actor_type'),
                    'cataphoric_failure': mvd_data.get('catastrophic_failure'),   
                },
                # Event States

                #######################################################################
                # --- INPUT, CONTROL  AND PERFORMANCE METRICS: ALL SOURCES -----------#
                #######################################################################
                
                'vehicle_spacer': '||',
                # Establish use of Ackermann Settings
                'ackermann_enabled': ackermann_targets.get('enabled'),

                # Controller Inputs
                'input_mode': control_data.get('input_mode'),

                # Vehicle State
                'handbrake': vehicle_state.get('handbrake'),
                'reverse': vehicle_state.get('reverse'),
                'manual_gear_shift': vehicle_state.get('manual_gear_shift'),
                'gear': vehicle_state.get('gear'),
                'blinker_state': vehicle_state.get('blinker_state'),
                'autopilot': vehicle_state.get('autopilot'),

                # Vehicle Performance (from CARLA)
                'vehicle_performance_spacer':'|',
                'location_x': loc.x,
                'location_y': loc.y,
                'location_z': loc.z,
                'latitude': environmental_data.get('latitude'),
                'longitude': environmental_data.get('longitude'),
                'rotation_pitch': rot.pitch,
                'rotation_yaw': rot.yaw,
                'rotation_roll': rot.roll,
                'carla_velocity_x': velocity.x if hasattr(velocity, 'x') else velocity,
                'carla_velocity_y': velocity.y if hasattr(velocity, 'y') else 0,
                'carla_velocity_z': velocity.z if hasattr(velocity, 'z') else 0,
                'acceleration_x': accel.x,
                'acceleration_y': accel.y,
                'acceleration_z': accel.z,
                'angular_velocity_x': ang_vel.x,
                'angular_velocity_y': ang_vel.y,
                'angular_velocity_z': ang_vel.z,
 
                # Steer **Metrics** (All Sources)
                'raw_steer': raw_inputs.get('steer'),
                'norm_steer': norm_outputs.get('steer'),
                'clamped_steer': raw_inputs.get('clamped_steer'),
                'ackermann_target_steer': ackermann_targets.get('target_steer'),

                # Throttle: Speed & Acceleration **Metrics** (All Sources)
                'raw_throttle': raw_inputs.get('throttle'),
                'norm_throttle': norm_outputs.get('throttle'),
                'ackermann_target_speed_ms': ackermann_targets.get('target_speed_ms'),
                'ackermann_target_accel_ms2': ackermann_targets.get('target_accel_ms2'),
                'speed_kmh': metrics.get('speed_kmh'),

                # Braking: Metrics
                'raw_brake': raw_inputs.get('brake'),
                'norm_brake': norm_outputs.get('brake'),

                #######################################################################
                # --- STATIC PARAMETERS ---------------------------------------#
                #######################################################################
                
                # Static Parameters
                'steer_deadzone': static_params.get('steer_deadzone'),
                'steer_linearity': static_params.get('steer_linearity'),
                'pedal_deadzone': static_params.get('pedal_deadzone'),

            }

            # Merge in environmental and physics data using dictionary unpacking
            frame_record.update(environmental_data)
            frame_record.update(physics_data)

            # ============================================================
            # FLAG VIOLATIONS HERE (Single Responsibility)
            # ============================================================
            frame_record.update(self._flag_violations(frame_record))

            # Add to buffer for batch insert
            self._frame_buffer.append(frame_record)
            self._frame_count += 1

            # Flush buffer when it reaches the buffer size
            if len(self._frame_buffer) >= self._buffer_size:
                self._flush_buffer()

        except Exception as e:
            logging.error(f"Error logging frame data: {e}", exc_info=True)

    def _flush_buffer(self) -> None:
        """Flush the frame buffer to the DataFrame using efficient batch insert."""
        if not self._frame_buffer:
            return

        try:
            # Flatten nested dictionaries before creating DataFrame
            flattened_buffer = [self._flatten_dict(frame) for frame in self._frame_buffer]

            # Create DataFrame from buffer and concatenate once
            new_df = pd.DataFrame(flattened_buffer)
            self._session_df = pd.concat([self._session_df, new_df], ignore_index=True)
            self._frame_buffer.clear()
            logging.debug(f"Flushed {len(new_df)} frames to DataFrame")
        except Exception as e:
            logging.error(f"Error flushing buffer: {e}", exc_info=True)

    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """
        Flatten nested dictionaries for CSV export.

        Args:
            d: Dictionary to flatten
            parent_key: Parent key for recursion
            sep: Separator between nested keys

        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _extract_vehicle_physics(self, player) -> Dict[str, Any]:
        """
        Extract vehicle physics parameters.

        Args:
            player: CARLA player/vehicle actor

        Returns:
            Dictionary with physics data
        """
        try:
            physics_control = player.get_physics_control()
            wheels = physics_control.wheels
            wheel_fl, wheel_fr, wheel_rl, wheel_rr = wheels[0], wheels[1], wheels[2], wheels[3]

            return {
                # Static Vehicle Physics Parameters
                'physics_spacer': '||',
                'phys_mass': physics_control.mass,
                'phys_drag_coefficient': physics_control.drag_coefficient,
                'phys_max_rpm': physics_control.max_rpm,
                'phys_moi': physics_control.moi,
                'phys_center_of_mass_x': physics_control.center_of_mass.x,
                'phys_center_of_mass_y': physics_control.center_of_mass.y,
                'phys_center_of_mass_z': physics_control.center_of_mass.z,

                # Per-Wheel Physics (Front-Left)
                'advanced_spacer': '||',
                'wheel_pos_fl_x': wheel_fl.position.x,
                'wheel_pos_fl_y': wheel_fl.position.y,
                'wheel_pos_fl_z': wheel_fl.position.z,
                'wheel_max_steer_angle_fl': wheel_fl.max_steer_angle,
                'wheel_damping_rate_fl': wheel_fl.damping_rate,
                'wheel_tire_friction_fl': wheel_fl.lat_stiff_value,

                # Per-Wheel Physics (Front-Right)
                'wheel_pos_fr_x': wheel_fr.position.x,
                'wheel_pos_fr_y': wheel_fr.position.y,
                'wheel_pos_fr_z': wheel_fr.position.z,
                'wheel_max_steer_angle_fr': wheel_fr.max_steer_angle,
                'wheel_damping_rate_fr': wheel_fr.damping_rate,
                'wheel_tire_friction_fr': wheel_fr.lat_stiff_value,

                # Per-Wheel Physics (Rear-Left)
                'wheel_pos_rl_x': wheel_rl.position.x,
                'wheel_pos_rl_y': wheel_rl.position.y,
                'wheel_pos_rl_z': wheel_rl.position.z,
                'wheel_max_steer_angle_rl': wheel_rl.max_steer_angle,
                'wheel_damping_rate_rl': wheel_rl.damping_rate,
                'wheel_tire_friction_rl': wheel_rl.lat_stiff_value,

                # Per-Wheel Physics (Rear-Right)
                'wheel_pos_rr_x': wheel_rr.position.x,
                'wheel_pos_rr_y': wheel_rr.position.y,
                'wheel_pos_rr_z': wheel_rr.position.z,
                'wheel_max_steer_angle_rr': wheel_rr.max_steer_angle,
                'wheel_damping_rate_rr': wheel_rr.damping_rate,
                'wheel_tire_friction_rr': wheel_rr.lat_stiff_value,
            }
        except Exception as e:
            logging.warning(f"Error extracting vehicle physics: {e}")
            return {}

    def _extract_environmental_context(self, player, world_obj, carla_map) -> Dict[str, Any]:
        """
        Extract environmental and contextual data.

        Args:
            player: CARLA player/vehicle actor
            world_obj: World object with sensors
            carla_map: CARLA map for waypoint queries

        Returns:
            Dictionary with environmental context
        """
        try:
            loc = player.get_location()

            # Get waypoint for the current location
            waypoint = None
            if carla_map and carla:
                try:
                    waypoint = carla_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
                except Exception as e:
                    logging.warning(f"Failed to get waypoint: {e}")

            # Calculate vehicle's distance from the center of its current lane
            distance_from_lane_center = None
            if waypoint:
                vehicle_vector = loc - waypoint.transform.location
                distance_from_lane_center = abs(vehicle_vector.dot(waypoint.transform.get_right_vector()))

            # Get data from the forward-facing radar sensor
            dist_nearest_vehicle = float('inf')
            relative_speed_nearest_vehicle = 0.0
            if world_obj.lane_manager and world_obj.lane_manager.radar_data:
                for detection in world_obj.lane_manager.radar_data:
                    # Consider only objects directly in front (low azimuth)
                    if abs(detection.azimuth) < math.radians(10):
                        if detection.depth < dist_nearest_vehicle:
                            dist_nearest_vehicle = detection.depth
                            relative_speed_nearest_vehicle = detection.velocity

            # Get world state information
            speed_limit = player.get_speed_limit()
            traffic_light_state = str(player.get_traffic_light_state())
            is_at_traffic_light = player.is_at_traffic_light()

            # Get GNSS sensor data
            gnss_sensor = world_obj.gnss_sensor_instance
            latitude = gnss_sensor.latitude if gnss_sensor else None
            longitude = gnss_sensor.longitude if gnss_sensor else None

            return {
                'environmental_spacer': '||',
                'speed_limit': speed_limit,
                'traffic_light_state': traffic_light_state,
                'is_at_traffic_light': is_at_traffic_light,
                'distance_from_lane_center': distance_from_lane_center,
                'lane_width': waypoint.lane_width if waypoint else None,
                'is_junction': waypoint.is_junction if waypoint else None,
                'dist_nearest_vehicle': dist_nearest_vehicle if dist_nearest_vehicle != float('inf') else None,
                'relative_speed_nearest_vehicle': relative_speed_nearest_vehicle,
                'latitude': latitude,
                'longitude': longitude,
            }
        except Exception as e:
            logging.warning(f"Error extracting environmental context: {e}")
            return {}

    def _flag_violations(self, frame: Dict[str, Any]) -> Dict[str, bool]:
        """
        Flags violations and events based on thresholds.
        This is where all violation logic lives - not in WindowProcessor.

        Returns dict of boolean flags to merge into frame_record.
        """
        flags = {}

        # Helper function to safely get nested or flat keys
        def safe_get(key, default=None):
            # Try flattened keys first (after dict merging)
            if key in frame:
                return frame[key]
            # Try with common prefixes
            for prefix in ['predictive_indices_', 'scores_', 'event_states_', 'environmental_']:
                full_key = f"{prefix}{key}"
                if full_key in frame:
                    return frame[full_key]
            return default

        # Speed violations
        speed_limit = safe_get('speed_limit', 50)
        current_speed = frame.get('speed_kmh', 0)
        flags['speed_exceeded'] = current_speed > (speed_limit * 1.1) if speed_limit else False  # 10% buffer
        flags['severe_speed_exceeded'] = current_speed > (speed_limit * 1.25) if speed_limit else False

        # Lane violations (using flattened keys)
        p_lane = safe_get('p_lane_violation', 0.0)
        tlc = safe_get('tlc_s', 99.0)
        flags['lane_violation_likely'] = p_lane > 0.7
        flags['lane_departure_imminent'] = tlc < 1.5

        # Collision risks (using flattened keys)
        p_collision = safe_get('p_collision', 0.0)
        ttc = safe_get('ttc_s', 99.0)
        flags['collision_warning'] = p_collision > 0.5
        flags['collision_critical'] = ttc < 2.0

        # Harsh maneuvers
        accel_x = frame.get('acceleration_x', 0.0)
        accel_y = frame.get('acceleration_y', 0.0)
        flags['harsh_braking'] = accel_x < -4.0
        flags['harsh_acceleration'] = accel_x > 3.5
        flags['harsh_cornering'] = abs(accel_y) > 0.3 * 9.81  # 0.3g lateral

        # Traffic violations
        traffic_light = safe_get('traffic_light_state', 'Unknown')
        at_light = safe_get('is_at_traffic_light', False)
        flags['red_light_violation'] = (
            traffic_light == 'Red' and
            at_light and
            current_speed > 5  # Moving through red
        )

        # Poor score thresholds (these will be flattened with scores_ prefix)
        flags['poor_lane_score'] = safe_get('score_lane_raw', 100) < 70
        flags['poor_collision_score'] = safe_get('score_collision_raw', 100) < 70
        flags['poor_overall_score'] = safe_get('overall_mvd_score', 100) < 75

        return flags

    def get_dataframe(self) -> pd.DataFrame:
        """Returns the current in-memory DataFrame (flushes buffer first)."""
        self._flush_buffer()
        return self._session_df.copy()

    def get_recent_frames(self, n: int = 100) -> pd.DataFrame:
        """Returns the last N frames (flushes buffer first)."""
        self._flush_buffer()
        return self._session_df.tail(n).copy()

    def save_to_csv(self, log_dir="./Session_logs/"):
        """Save the DataFrame to CSV (flushes buffer first)."""
        # Flush any remaining buffered frames
        self._flush_buffer()

        if self._session_df.empty:
            logging.warning("No session data to save.")
            return

        try:
            os.makedirs(log_dir, exist_ok=True)
            log_filename = f"mvd_session_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            log_path = os.path.join(log_dir, log_filename)

            self._session_df.to_csv(log_path, index=False)
            
            logging.info(f"Saved {len(self._session_df)} frames to {log_path}")

        except Exception as e:
            logging.error(f"Error saving session log: {e}", exc_info=True)

    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for the current session.

        Returns:
            Dictionary with session statistics
        """
        self._flush_buffer()

        if self._session_df.empty:
            return {"error": "No session data available"}

        summary = {
            'total_frames': len(self._session_df),
            'duration_s': 0.0,
            'statistics': {},
            'violations': {},
            'scores': {}
        }

        # Calculate duration
        if 'timestamp' in self._session_df.columns:
            summary['duration_s'] = float(
                self._session_df['timestamp'].max() - self._session_df['timestamp'].min()
            )

        # Get numeric columns for statistics
        numeric_cols = self._session_df.select_dtypes(include='number').columns

        # Speed statistics
        if 'speed_kmh' in numeric_cols:
            summary['statistics']['speed'] = {
                'mean_kmh': float(self._session_df['speed_kmh'].mean()),
                'max_kmh': float(self._session_df['speed_kmh'].max()),
                'std_kmh': float(self._session_df['speed_kmh'].std())
            }

        # Acceleration statistics
        for axis in ['acceleration_x', 'acceleration_y', 'acceleration_z']:
            if axis in numeric_cols:
                summary['statistics'][axis] = {
                    'mean': float(self._session_df[axis].mean()),
                    'max': float(self._session_df[axis].max()),
                    'min': float(self._session_df[axis].min())
                }

        # Count violations
        violation_cols = [c for c in self._session_df.columns if
                         any(keyword in c for keyword in ['violation', 'exceeded', 'harsh', 'warning', 'critical'])]

        for col in violation_cols:
            if self._session_df[col].dtype == bool:
                count = int(self._session_df[col].sum())
                if count > 0:
                    summary['violations'][col] = count

        # Score statistics
        score_cols = [c for c in numeric_cols if 'score' in c.lower() or 'mvd' in c.lower()]
        for col in score_cols:
            summary['scores'][col] = {
                'mean': float(self._session_df[col].mean()),
                'min': float(self._session_df[col].min()),
                'final': float(self._session_df[col].iloc[-1])
            }

        return summary

    def get_violation_timeline(self) -> pd.DataFrame:
        """
        Get a timeline of violations with timestamps.

        Returns:
            DataFrame with violations and their timestamps
        """
        self._flush_buffer()

        if self._session_df.empty:
            return pd.DataFrame()

        # Get violation columns
        violation_cols = [c for c in self._session_df.columns if
                         any(keyword in c for keyword in ['violation', 'exceeded', 'harsh', 'warning', 'critical'])]

        if not violation_cols:
            return pd.DataFrame()

        # Filter to only rows where at least one violation occurred
        violation_mask = self._session_df[violation_cols].any(axis=1)
        violations_df = self._session_df[violation_mask][['timestamp', 'frame'] + violation_cols].copy()

        return violations_df

    def clear_session(self):
        """Clear all session data and reset counters."""
        self._session_df = pd.DataFrame()
        self._frame_buffer.clear()
        self._frame_count = 0
        logging.info("Session data cleared")