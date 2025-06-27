import carla
import weakref
import logging
import math
from enum import Enum, auto
import numpy as np

# ==============================================================================
# -- Enums for Lane Management and Violation States ----------------------------
# ==============================================================================


class LaneViolationState(Enum):
    """Enumeration for the severity of a lane marking violation."""

    NORMAL = auto()
    MINOR = auto()
    MODERATE = auto()
    SEVERE = auto()
    CRITICAL = auto()


class LaneChangeState(Enum):
    """Enumeration for the state of a lane change maneuver."""

    NORMAL = auto()
    SIGNALLED = auto()
    UNSIGNALLED = auto()
    UNSAFE = auto()


# ==============================================================================
# -- Lane Violation State Machine ----------------------------------------------
# ==============================================================================


class LaneViolationStateMachine:
    """
    Manages the state of lane violations based on the type of line crossed.
    """

    def __init__(self, hud):
        self.state = LaneViolationState.NORMAL
        self.hud = hud

    def classify_violation(self, marking: carla.LaneMarking) -> LaneViolationState:
        """Classifies a single lane marking crossing into a violation state."""
        color = marking.color
        marking_type = marking.type

        if marking_type in [
            carla.LaneMarkingType.Grass,
            carla.LaneMarkingType.Curb,
            carla.LaneMarkingType.NONE,
        ]:
            return LaneViolationState.CRITICAL
        if hasattr(marking, "lane_type") and marking.lane_type in [
            carla.LaneType.Median,
            carla.LaneType.Restricted,
        ]:
            return LaneViolationState.CRITICAL
        if marking_type in [
            carla.LaneMarkingType.Solid,
            carla.LaneMarkingType.SolidSolid,
        ]:
            return (
                LaneViolationState.SEVERE
                if color == carla.LaneMarkingColor.Yellow
                else LaneViolationState.MODERATE
            )
        if marking_type == carla.LaneMarkingType.Broken:
            return LaneViolationState.MINOR
        if hasattr(marking, "lane_type") and marking.lane_type in [
            carla.LaneType.Shoulder,
            carla.LaneType.Sidewalk,
            carla.LaneType.Biking,
        ]:
            return LaneViolationState.SEVERE

        return LaneViolationState.NORMAL

    def handle_violation(self, markings: list):
        """Processes a list of crossed markings and transitions to the most severe state."""
        highest_severity = LaneViolationState.NORMAL
        for marking in markings:
            violation = self.classify_violation(marking)
            if violation.value > highest_severity.value:
                highest_severity = violation
        self.transition(highest_severity)

    def transition(self, new_state: LaneViolationState):
        """Transitions the state machine to a new state and triggers notifications."""
        if new_state == self.state and new_state != LaneViolationState.NORMAL:
            return

        self.state = new_state
        self.on_state_enter(new_state)

    def on_state_enter(self, state: LaneViolationState):
        """Handles logic for entering a new state, primarily for HUD notifications."""
        if state != LaneViolationState.NORMAL:
            notifications = {
                LaneViolationState.MINOR: ("Minor Lane Drift", (255, 255, 0)),
                LaneViolationState.MODERATE: (
                    "Moderate Violation: Solid Line",
                    (255, 165, 0),
                ),
                LaneViolationState.SEVERE: (
                    "SEVERE Violation: No Passing",
                    (255, 69, 0),
                ),
                LaneViolationState.CRITICAL: (
                    "CRITICAL VIOLATION: Off-Road/Median!",
                    (255, 0, 0),
                ),
            }
            message, color = notifications[state]
            self.hud.notification(message, seconds=3.0, text_color=color)


# ==============================================================================
# -- Lane Change Management System ---------------------------------------------
# ==============================================================================
class LaneManagement:
    """
    Evaluates the safety and signaling of lane change maneuvers using radar.
    """

    def __init__(self, parent_actor, hud, controller):
        self.actor = parent_actor
        self.world = parent_actor.get_world()
        self.hud = hud
        self.controller = controller
        self.state = LaneChangeState.NORMAL
        self.radar_data = []
        self.proximity_threshold = 15.0
        self.radar_sensor = None

        radar_bp = self.world.get_blueprint_library().find("sensor.other.radar")
        radar_bp.set_attribute("horizontal_fov", "30")
        radar_bp.set_attribute("range", "50")
        radar_transform = carla.Transform(carla.Location(x=2.5, z=1.0))

        try:
            self.radar_sensor = self.world.spawn_actor(
                radar_bp, radar_transform, attach_to=self.actor
            )
            weak_self = weakref.ref(self)
            self.radar_sensor.listen(
                lambda data: LaneManagement._on_radar_detect(weak_self, data)
            )
            logging.info("Lane Management radar sensor spawned.")
        except Exception as e:
            logging.error(f"Failed to spawn Lane Management radar sensor: {e}")

    @staticmethod
    def _on_radar_detect(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        self.radar_data = [det for det in radar_data if det.velocity > -1]

    def evaluate_lane_change(self, lane_markings: list):
        """Evaluates the current lane change based on blinker status and radar data."""
        if not any(m.type == carla.LaneMarkingType.Broken for m in lane_markings):
            return

        blinker_state = self.controller.get_blinker_state()
        signalling = blinker_state in [1, 2]
        closest_vehicle_distance = min(
            [det.depth for det in self.radar_data] or [float("inf")]
        )

        if signalling:
            self._transition_state(
                LaneChangeState.SIGNALLED
                if closest_vehicle_distance > self.proximity_threshold
                else LaneChangeState.UNSAFE
            )
        else:
            self._transition_state(
                LaneChangeState.UNSIGNALLED
                if closest_vehicle_distance > self.proximity_threshold
                else LaneChangeState.UNSAFE
            )

    def _transition_state(self, new_state: LaneChangeState):
        """Transitions the maneuver state and triggers a notification."""
        if self.state != new_state:
            self.state = new_state
            messages = {
                LaneChangeState.SIGNALLED: (
                    "Good lane change: Signalled & Safe.",
                    (0, 255, 0),
                ),
                LaneChangeState.UNSIGNALLED: (
                    "Unsignalled Lane Change!",
                    (255, 165, 0),
                ),
                LaneChangeState.UNSAFE: ("UNSAFE Lane Change: Too Close!", (255, 0, 0)),
            }
            if new_state in messages:
                message, color = messages[new_state]
                self.hud.notification(f"{message}", seconds=3.0, text_color=color)

    def get_lane_change_state(self):
        """Getter method to poll the state from Main.py."""
        current_state = self.state
        if self.state != LaneChangeState.NORMAL:
            self.state = LaneChangeState.NORMAL
        return current_state

    def destroy(self):
        if self.radar_sensor and self.radar_sensor.is_alive:
            self.radar_sensor.stop()
            self.radar_sensor.destroy()


# ==============================================================================
# -- Core Sensor Classes -------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    """Simulates a GNSS (Global Navigation Satellite System) sensor."""

    def __init__(self, parent_actor, hud_instance):
        self.sensor = None
        self._parent = parent_actor
        self.latitude, self.longitude = 0.0, 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find("sensor.other.gnss")
        if bp:
            self.sensor = world.spawn_actor(
                bp,
                carla.Transform(carla.Location(x=1.0, z=2.8)),
                attach_to=self._parent,
            )
            weak_self = weakref.ref(self)
            self.sensor.listen(
                lambda event: GnssSensor._on_gnss_event(weak_self, event)
            )

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.latitude, self.longitude = event.latitude, event.longitude

    def destroy(self):
        if self.sensor:
            self.sensor.destroy()


class CollisionSensor(object):
    """Simulates a collision sensor."""

    def __init__(self, parent_actor, hud_instance):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud_instance
        self.collided, self.actor_type, self.intensity = False, None, 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find("sensor.other.collision")
        if bp:
            self.sensor = world.spawn_actor(
                bp, carla.Transform(), attach_to=self._parent
            )
            weak_self = weakref.ref(self)
            self.sensor.listen(
                lambda event: CollisionSensor._on_collision(weak_self, event)
            )

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.collided = True
        self.actor_type = event.other_actor.type_id if event.other_actor else "unknown"
        impulse = event.normal_impulse
        self.intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.hud.notification(
            "COLLISION!", text_color=(255, 0, 0), is_critical_center=True
        )
        self.hud.play_sound_for_event("collision", force_play=True)

    def get_collision_data_and_reset(self):
        data = {
            "collided": self.collided,
            "actor_type": self.actor_type,
            "intensity": self.intensity,
        }
        self.collided, self.actor_type, self.intensity = False, None, 0.0
        return data

    def destroy(self):
        if self.sensor:
            self.sensor.destroy()


class LaneInvasionSensor(object):
    """Integrates the LaneViolationStateMachine and the LaneManagement system."""

    def __init__(self, parent_actor, hud_instance, lane_manager: LaneManagement = None):
        self.sensor = None
        self._parent = parent_actor
        self.lane_manager = lane_manager
        self.violation_sm = LaneViolationStateMachine(hud_instance)
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find("sensor.other.lane_invasion")
        if bp:
            self.sensor = world.spawn_actor(
                bp, carla.Transform(), attach_to=self._parent
            )
            weak_self = weakref.ref(self)
            self.sensor.listen(
                lambda event: LaneInvasionSensor._on_invasion(weak_self, event)
            )

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        if self.lane_manager:
            self.lane_manager.evaluate_lane_change(event.crossed_lane_markings)
        self.violation_sm.handle_violation(event.crossed_lane_markings)

    def get_violation_state(self):
        """Getter method to poll the state from Main.py."""
        current_state = self.violation_sm.state
        if self.violation_sm.state != LaneViolationState.NORMAL:
            self.violation_sm.state = LaneViolationState.NORMAL
        return current_state

    def destroy(self):
        if self.sensor:
            self.sensor.destroy()
