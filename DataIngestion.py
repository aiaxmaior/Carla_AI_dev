# ============================================================================
# PERF CHECK (file-level):
# ============================================================================
# [X] | Role: ROOT-LEVEL DataIngestion (likely legacy - Core/Simulation/DataIngestion is newer)
# [X] | Hot-path functions: log_frame() called every tick
# [X] |- Heavy allocs in hot path? YES - dict creation + list append per frame
# [X] |- pandas/pyarrow/json/disk/net in hot path? pandas conversion in get_dataframe()
# [ ] | Graphics here? No
# [X] | Data produced (tick schema?): Frame dict per tick (appended to list)
# [X] | Storage (Parquet/Arrow/CSV/none): CSV export (save_to_csv)
# [X] | Queue/buffer used?: Simple list (_session_data) - no batching
# [X] | Session-aware? Should add session_id
# [ ] | Debug-only heavy features?: None
# Top 3 perf risks:
# 1. [PERF_HOT] log_frame() creates large dict + CARLA queries every tick
# 2. [PERF_HOT] No buffering - list.append() every frame, DataFrame conversion on demand
# 3. [PERF_SPLIT] Duplicate with Core/Simulation/DataIngestion.py - consolidate?
# ============================================================================

import pandas as pd
import os
import time
import logging
import carla
import math
from typing import Dict, Any

class DataIngestion(object):
    """
    A stateful class to handle the collection, consolidation, and saving of
    per-frame simulation data into a structured Pandas DataFrame.
    """

    def __init__(self):
        """
        Initializes the data logger with an empty list to store frame data.
        """
        self._session_data = []
        logging.info("DataIngestion object created. Ready to log session data.")
        # Predictive indices (latest snapshot; merged into each frame record)
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
        self._session_data_df = pd.DataFrame()
        self._log_frame = None

    def set_predictive_indices(self, preds: Dict[str, Any]) -> None:
        """
        Store the most recent predictive indices to be merged into the next frame record.
        Only keys we know are taken; extra keys are ignored.
        """
        if not isinstance(preds, dict):
            return
        for k in list(self._last_predictive.keys()):
            if k in preds:
                self._last_predictive[k] = preds[k]

    def get_dataframe(self):
        """
        Return the current session data as a Pandas DataFrame.
        """
        if not self._session_data:
            return pd.DataFrame()
        self._session_data_df = pd.DataFrame(self._session_data)
        return self._session_data_df

    def log_frame(self, world_obj, metrics):
        """
        Processes a dictionary of metrics for a single frame, flattens the nested
        data, and appends a single, structured record to the session data list.

        Args:
            metrics (dict): A dictionary containing all data for the current frame.
        """
        try:
            # Get helper objects for easy access
            player = world_obj.player
            carla_map = world_obj.world.get_map()

            # --- Flatten the nested dictionaries ---
            mvd_data = metrics.get('mvd_datalog', {})
            control_data = metrics.get('controller_datalog', {})
            raw_inputs = control_data.get('raw_inputs', {})
            norm_outputs = control_data.get('normalized_outputs', {})
            vehicle_state = control_data.get('vehicle_state', {})
            ackermann_targets = control_data.get('ackermann_targets', {})
            static_params = control_data.get('static_params', {})
            
            # --- Get CARLA-derived vehicle performance data ---
            accel = player.get_acceleration()
            velocity = player.get_velocity()
            ang_vel = player.get_angular_velocity()
            loc = player.get_location()
            rot = player.get_transform().rotation
            
            # --- Get Advanced Vehicle Physics Parameters ---
            physics_control = player.get_physics_control()
            ackermann_applied = player.get_ackermann_controller_settings()
            wheels = physics_control.wheels
            wheel_fl, wheel_fr, wheel_rl, wheel_rr = wheels[0], wheels[1], wheels[2], wheels[3]

            # --- NEW: Get Environmental and Contextual Data ---
            # Get waypoint for the current location to understand the road context
            waypoint = carla_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            
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
                # --- EVENT ---------------- STATES ----------------------------------#
                #######################################################################
                'predictive_indices': {
                    
                },
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
                'latitude': latitude,
                'longitude': longitude,
                'rotation_pitch': rot.pitch,
                'rotation_yaw': rot.yaw,
                'rotation_roll': rot.roll,
                'carla_velocity_x': velocity,
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

                #######################################################################
                # --- ENVIRONMENTAL PARAMETERS ---------------------------------------#
                #######################################################################

                # NEW: Environmental Context
                # Add spacer column for easy reading
                'environmental_spacer':'||',
                'speed_limit': speed_limit,
                'traffic_light_state': traffic_light_state,
                'is_at_traffic_light': is_at_traffic_light,
                'distance_from_lane_center': distance_from_lane_center,
                'lane_width': waypoint.lane_width if waypoint else None,
                'is_junction': waypoint.is_junction if waypoint else None,
                'dist_nearest_vehicle': dist_nearest_vehicle if dist_nearest_vehicle != float('inf') else None,
                'relative_speed_nearest_vehicle': relative_speed_nearest_vehicle,

                #######################################################################
                # --- PHYSICS PARAMETERS ---------------------------------------------#
                #######################################################################
                
                # Static Vehicle Physics Parameters
                'physics_spacer':'||',
                'phys_mass': physics_control.mass,
                'phys_drag_coefficient': physics_control.drag_coefficient,
                'phys_max_rpm': physics_control.max_rpm,
                'phys_moi': physics_control.moi,
                'phys_center_of_mass_x': physics_control.center_of_mass.x,
                'phys_center_of_mass_y': physics_control.center_of_mass.y,
                'phys_center_of_mass_z': physics_control.center_of_mass.z,

                #######################################################################
                # --- ADVANCED PARAMETERS, METRICS/DATA ------------------------------#
                #######################################################################

                # Per-Wheel Physics (Front-Left)
                # Static Vehicle Physics Parameters
                'advanced_spacer':'||',
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
            
            frame_record.update({
                "p_lane_violation": float(self._last_predictive.get("p_lane_violation", 0.0)),
                "tlc_s": float(self._last_predictive.get("tlc_s", 99.0)),
                "dist_to_lane_boundary_m": float(self._last_predictive.get("dist_to_lane_boundary_m", 0.0)),
                "p_collision": float(self._last_predictive.get("p_collision", 0.0)),
                "ttc_s": float(self._last_predictive.get("ttc_s", 99.0)),
                "a_required_brake_mps2": float(self._last_predictive.get("a_required_brake_mps2", 0.0)),
                "p_harsh_operation": float(self._last_predictive.get("p_harsh_operation", 0.0)),
                "dominant_axis": str(self._last_predictive.get("dominant_axis", "none")),
            })
            self._log_frame = frame_record
            self._session_data.append(frame_record)

        except Exception as e:
            logging.error(f"Error logging frame data: {e}", exc_info=True)

    def get_last_logged_frame(self):
        """
        Returns the most recently logged frame record.
        """
        try:
            return self._log_frame
        except Exception as e:
            logging.error(f"Error retrieving last logged frame: {e}", exc_info=True)
            return None

    def save_to_csv(self, log_dir="./Session_logs/"):
        """
        Consolidates the collected session data into a Pandas DataFrame and
        saves it to a timestamped CSV file.
        """
        if not self._session_data:
            logging.warning("No session data to save.")
            return

        try:
            os.makedirs(log_dir, exist_ok=True)
            log_filename = f"mvd_session_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            log_path = os.path.join(log_dir, log_filename)

            # Create the DataFrame from the list of flat records in one efficient operation
            log_df = pd.DataFrame(self._session_data)

            # Save to the desired format
            log_df.to_csv(log_path, index=False)
            
            logging.info(f"MVD session data consolidated into DataFrame ({len(log_df)} rows) and saved to {log_path}")

        except Exception as e:
            logging.error(f"Error consolidating or saving MVD session log: {e}", exc_info=True)
