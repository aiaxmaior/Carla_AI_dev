# MVD.py
import logging
import carla
import math
import json
import os

# Configure logging for this module
logger = logging.getLogger(__name__)


class MVDFeatureExtractor:
    """
    Extracts features from sensor data and calculates Motor Vehicle Dynamics (MVD) scores.
    Focuses on three components: Collision Avoidance, Lane Management, and Harsh Driving.
    Penalties and thresholds can be loaded from an external JSON file.
    """

    def __init__(self, config_path="mvd_config.json"):
        """
        Initializes the MVD scores and state variables.
        """
        # Load the penalty configuration from a file or use defaults.
        self._load_penalty_config(config_path)

        # Internal raw scores (start at 100 and decrease with penalties)
        self._collision_avoidance_score = float(self.penalties["initial_score"])
        self._lane_management_score = float(self.penalties["initial_score"])
        self._harsh_driving_score = float(self.penalties["initial_score"])

        # State variables
        self.previous_velocity_vector = carla.Vector3D(0, 0, 0)
        self.last_update_time = None
        self._catastrophic_failure_occurred = False
        self._previous_waypoint = None

        # Per-tick event flags
        self._collision_detected_this_tick = False
        self._lane_violation_detected_this_tick = False

        logger.info(
            "MVDFeatureExtractor initialized with 3 components: Collision Avoidance, Lane Management, Harsh Driving."
        )

    def _load_penalty_config(self, path):
        """Loads penalty values from a JSON file, with hardcoded defaults."""
        self.penalties = {
            "initial_score": 100,
            "collision_penalties": {
                "major_intensity_threshold": 2000,
                "moderate_intensity_threshold": 500,
                "moderate_penalty": 60,
                "minor_penalty": 25,
            },
            "lane_management_penalties": {
                "violation_minor": 5,
                "violation_moderate": 15,
                "violation_severe": 30,
                "violation_critical": 50,
                "maneuver_unsignalled": 10,
                "maneuver_unsafe": 25,
                "turn_signal_fail": 10,
            },
            "harsh_driving_penalties": {
                "acceleration_g": 0.45,
                "braking_g": 0.50,
                "cornering_g": 0.55,
                "accel_penalty": 15,
                "brake_penalty": 15,
                "corner_penalty": 20,
            },
            "score_recovery": {"rate_per_tick": 0.05},
        }

        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    loaded_config = json.load(f)
                    for category, values in loaded_config.items():
                        if category in self.penalties and isinstance(values, dict):
                            self.penalties[category].update(values)
                    logging.info(f"Successfully loaded MVD config from {path}")
            except Exception as e:
                logging.error(
                    f"Error loading MVD config file '{path}': {e}. Using default values."
                )
        else:
            logging.info(
                f"MVD config file not found at '{path}'. Using default values."
            )

    def reset_scores(self):
        """
        Resets all MVD scores and state variables to their initial perfect state.
        """
        self._collision_avoidance_score = float(self.penalties["initial_score"])
        self._lane_management_score = float(self.penalties["initial_score"])
        self._harsh_driving_score = float(self.penalties["initial_score"])

        self.previous_velocity_vector = carla.Vector3D(0, 0, 0)
        self.last_update_time = None
        self._collision_detected_this_tick = False
        self._lane_violation_detected_this_tick = False
        self._catastrophic_failure_occurred = False
        self._previous_waypoint = None

        logger.info("MVD scores reset to initial values.")

    def update_scores(
        self,
        collision_data: dict,
        lane_violation_state,
        lane_change_state,
        speed_kmh: float,
        player,
        world,
        current_time: float,
        blinker_state: int,
    ) -> dict:
        """
        Updates the internal MVD scores based on the latest sensor data and states.
        This is the main entry point for scoring each tick.
        """
        # Reset per-tick flags
        self._collision_detected_this_tick = False
        self._lane_violation_detected_this_tick = False

        # Calculate scores for each component
        self._calculate_collision_score(collision_data)
        self._calculate_lane_score(
            lane_violation_state,
            lane_change_state,
            speed_kmh,
            player,
            world,
            blinker_state,
        )
        self._calculate_harsh_driving_score(player, current_time)

        return {
            "collision": self._collision_avoidance_score,
            "lane_discipline": self._lane_management_score,
            "harsh_driving": self._harsh_driving_score,
        }

    def _calculate_collision_score(self, collision_data: dict):
        """Calculates and updates the Collision Avoidance score."""
        p = self.penalties["collision_penalties"]
        initial_score = self.penalties["initial_score"]
        recovery_rate = self.penalties["score_recovery"]["rate_per_tick"]

        if collision_data["collided"]:
            self._collision_detected_this_tick = True
            actor_type = collision_data.get("actor_type", "")
            intensity = collision_data.get("intensity", 0.0)
            penalty = 0

            if "walker.pedestrian" in actor_type:
                self._collision_avoidance_score = 0
                self._catastrophic_failure_occurred = True
            elif intensity > p["major_intensity_threshold"]:
                self._collision_avoidance_score = 0
            elif intensity > p["moderate_intensity_threshold"]:
                penalty = p["moderate_penalty"]
            else:
                penalty = p["minor_penalty"]

            if penalty > 0:
                self._collision_avoidance_score -= penalty
            self._collision_avoidance_score = max(0, self._collision_avoidance_score)

        elif self._collision_avoidance_score < initial_score:
            self._collision_avoidance_score = min(
                initial_score, self._collision_avoidance_score + recovery_rate
            )

    def _calculate_lane_score(
        self,
        violation_state,
        lane_change_state,
        speed_kmh: float,
        player,
        world,
        blinker_state: int,
    ):
        """Calculates Lane Management score based on violations, maneuvers, and turn signal usage."""
        p = self.penalties["lane_management_penalties"]
        initial_score = self.penalties["initial_score"]
        recovery_rate = self.penalties["score_recovery"]["rate_per_tick"]
        total_penalty = 0

        # Penalty from lane line violations
        if violation_state and violation_state.name != "NORMAL":
            self._lane_violation_detected_this_tick = True
            violation_penalties = {
                "MINOR": p["violation_minor"],
                "MODERATE": p["violation_moderate"],
                "SEVERE": p["violation_severe"],
                "CRITICAL": p["violation_critical"],
            }
            total_penalty += violation_penalties.get(violation_state.name, 0)

        # Penalty from lane change maneuvers
        if lane_change_state and lane_change_state.name != "NORMAL":
            self._lane_violation_detected_this_tick = True
            maneuver_penalties = {
                "SIGNALLED": 0,
                "UNSIGNALLED": p["maneuver_unsignalled"],
                "UNSAFE": p["maneuver_unsafe"],
            }
            total_penalty += maneuver_penalties.get(lane_change_state.name, 0)

        # Penalty from failing to use turn signals at intersections
        turn_signal_penalty = self._track_turn_signal_usage(
            player, world, blinker_state
        )
        if turn_signal_penalty > 0:
            self._lane_violation_detected_this_tick = True
            total_penalty += turn_signal_penalty

        # Apply penalties or recover score
        if total_penalty > 0:
            self._lane_management_score = max(
                0, self._lane_management_score - total_penalty
            )
        elif self._lane_management_score < initial_score:
            self._lane_management_score = min(
                initial_score, self._lane_management_score + recovery_rate
            )

    def _track_turn_signal_usage(self, player, world, blinker_state: int) -> int:
        """Checks if a turn signal is used appropriately before a turn. Returns a penalty value."""
        current_waypoint = world.get_map().get_waypoint(
            player.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if not current_waypoint or not self._previous_waypoint:
            self._previous_waypoint = current_waypoint
            return 0

        # Detect if a turn was made by comparing road IDs
        if (
            current_waypoint.road_id != self._previous_waypoint.road_id
            and not current_waypoint.is_junction
        ):
            # Check if the player was at an intersection just before the turn
            if self._previous_waypoint.is_junction:
                if (
                    blinker_state == 0
                ):  # Blinker was off during a turn from an intersection
                    logger.warning("Failed to use turn signal at intersection.")
                    self._previous_waypoint = current_waypoint
                    return self.penalties["lane_management_penalties"][
                        "turn_signal_fail"
                    ]

        self._previous_waypoint = current_waypoint
        return 0

    def _calculate_harsh_driving_score(self, player, current_time):
        """Calculates acceleration in Gs and penalizes harsh driving, with score recovery."""
        p = self.penalties["harsh_driving_penalties"]
        initial_score = self.penalties["initial_score"]
        recovery_rate = self.penalties["score_recovery"]["rate_per_tick"]

        if self.last_update_time is None:
            self.last_update_time = current_time
            self.previous_velocity_vector = player.get_velocity()
            return

        delta_time = current_time - self.last_update_time
        if delta_time == 0:
            return

        current_velocity = player.get_velocity()
        delta_velocity = current_velocity - self.previous_velocity_vector
        accel_vec = delta_velocity / delta_time

        forward_accel = accel_vec.dot(player.get_transform().get_forward_vector())
        lateral_accel = accel_vec.dot(player.get_transform().get_right_vector())

        forward_g = forward_accel / 9.806
        lateral_g = abs(lateral_accel) / 9.806

        penalty = 0
        if forward_g > p["acceleration_g"]:
            penalty = p["accel_penalty"]
        elif forward_g < -p["braking_g"]:
            penalty = p["brake_penalty"]
        elif lateral_g > p["cornering_g"]:
            penalty = p["corner_penalty"]

        if penalty > 0:
            self._harsh_driving_score -= penalty
        elif self._harsh_driving_score < initial_score:
            self._harsh_driving_score += recovery_rate

        self._harsh_driving_score = max(
            0, min(initial_score, self._harsh_driving_score)
        )
        self.previous_velocity_vector = current_velocity
        self.last_update_time = current_time

    def _standardize_score(self, score: float) -> float:
        """Standardizes a given score to a 0-1 range."""
        max_score = self.penalties["initial_score"]
        clamped_score = max(0.0, min(max_score, score))
        return clamped_score / max_score if max_score > 0 else 0.0

    def get_standardized_indices(self) -> dict:
        """Returns the standardized component scores (0-1 range)."""
        weighted_mbi_score = (self._collision_avoidance_score * 0.7) + (
            self._harsh_driving_score * 0.3
        )
        return {
            "mbi_0_1": self._standardize_score(weighted_mbi_score),
            "lmi_0_1": self._standardize_score(self._lane_management_score),
        }

    def get_overall_mvd_score(self) -> float:
        """Calculates and returns the overall MVD score as a percentage (0-100%)."""
        indices = self.get_standardized_indices()
        total_indices = indices["mbi_0_1"] + indices["lmi_0_1"]
        num_indices = len(indices)
        overall_score_0_1 = total_indices / num_indices if num_indices > 0 else 0.0
        return max(0, min(100, overall_score_0_1 * 100.0))

    def get_mvd_datalog_metrics(self) -> dict:
        """Returns a dictionary of all MVD scores and indices for logging."""
        indices = self.get_standardized_indices()   
        return {
#            "current_velocity": self.previous_velocity_vector,
#            "collision_detected": self._collision_detected_this_tick,
#            "lane_violation_detected": self._lane_violation_detected_this_tick,
            "catastrophic_failure": self._catastrophic_failure_occurred,
            "overall_score": self.get_overall_mvd_score(),
            # new, friendlier names (aliases; no behavior change)
            "PSS_ProactiveSafety":   float(self._collision_avoidance_score),
            "LDS_LaneDiscipline":    float(self._lane_management_score),
            "DSS_DrivingSmoothness": float(self._harsh_driving_score),

            # keep existing fields for compatibility
            "index_mbi_0_1": indices.get("mbi_0_1"),
            "score_collision_raw": self._collision_avoidance_score,
            "index_lmi_0_1": indices.get("lmi_0_1"),
            "score_lane_raw": self._lane_management_score,
            "score_harsh_driving_raw": self._harsh_driving_score,
        }