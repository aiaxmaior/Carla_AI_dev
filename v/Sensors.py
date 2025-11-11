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
        # NEW: Track the last frame a violation was detected
        self._last_violation_frame = 0

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

    def handle_violation(self, markings: list, current_frame: int):
        """Processes a list of crossed markings and transitions to the most severe state."""
        self._last_violation_frame = current_frame
        highest_severity = LaneViolationState.NORMAL
        for marking in markings:
            violation = self.classify_violation(marking)
            if violation.value > highest_severity.value:
                highest_severity = violation
        self.transition(highest_severity)

    def tick(self, current_frame: int):
        """
        Called every frame to check if we should reset the state to NORMAL.
        If no violation has been detected for a few frames, we can assume we are back to normal.
        """
        if self.state != LaneViolationState.NORMAL and current_frame > self._last_violation_frame + 5:
             self.transition(LaneViolationState.NORMAL)


    def transition(self, new_state: LaneViolationState):
        """Transitions the state machine to a new state and triggers notifications."""
        if new_state == self.state: # Removed "and new_state != LaneViolationState.NORMAL" to allow resetting
            return

        self.state = new_state
        self.on_state_enter(new_state)

    def on_state_enter(self, state: LaneViolationState):
        """Handles logic for entering a new state, primarily for HUD notifications."""
        if state != LaneViolationState.NORMAL:
            notifications = {
                LaneViolationState.MINOR: ("Minor Lane Drift", (255, 255, 0), "lane_drift"),
                LaneViolationState.MODERATE: ("Moderate Violation: Solid Line", (255, 165, 0), "solid_line_crossing"),
                LaneViolationState.SEVERE: ("SEVERE Violation: No Passing", (255, 69, 0), "solid_line_crossing"),
                LaneViolationState.CRITICAL: ("CRITICAL VIOLATION: Off-Road/Median!", (255, 0, 0), "solid_line_crossing"),
            }
            message, color, sound = notifications[state]
            self.hud.event_manager.report('lane_violation', details={'message': message, 'color': color, 'sound': sound})

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
        # NEW: Add a timer to control how long the state remains active.
        self.state_active_until = 0.0

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

    def evaluate_lane_change(self, lane_markings: list) -> bool:
        """
        Evaluates the current lane change based on blinker status and radar data.
        Returns True if the event was handled as a lane change maneuver, False otherwise.
        """
        if not any(m.type == carla.LaneMarkingType.Broken for m in lane_markings):
            return False

        direction = None
        for marking in lane_markings:
            if marking.lane_change == carla.LaneChange.Left: direction = "Left"; break
            elif marking.lane_change == carla.LaneChange.Right: direction = "Right"; break
        
        if not direction:
            self._transition_state(LaneChangeState.UNSIGNALLED)
            return True

        blinker_state = self.controller.get_blinker_state()
        correct_signal = (direction == "Left" and blinker_state == 1) or \
                         (direction == "Right" and blinker_state == 2)

        is_safe = min([det.depth for det in self.radar_data] or [float("inf")]) > self.proximity_threshold

        if not is_safe: self._transition_state(LaneChangeState.UNSAFE)
        elif correct_signal: self._transition_state(LaneChangeState.SIGNALLED)
        else: self._transition_state(LaneChangeState.UNSIGNALLED)
            
        return True

    def _transition_state(self, new_state: LaneChangeState):
        """MODIFIED: Transitions the maneuver state, reports the event, and sets a timer."""
        # Only transition if the state is new, to avoid resetting the timer.
        if self.state != new_state:
            self.state = new_state
            # NEW: Set the state to be active for 3 seconds from the current simulation time.
            self.state_active_until = self.world.get_snapshot().timestamp.elapsed_seconds + 3.0

            messages = {
                LaneChangeState.SIGNALLED: ("Good lane change: Signalled & Safe.", (0, 255, 0), "lane_drift"),
                LaneChangeState.UNSIGNALLED: ("Unsignalled Lane Change!", (255, 165, 0), "lane_drift"),
                LaneChangeState.UNSAFE: ("UNSAFE Lane Change: Too Close!", (255, 0, 0), "solid_line_crossing"),
            }
            if new_state in messages:
                message, color, sound = messages[new_state]
                self.hud.event_manager.report(new_state.name, details={'message': message, 'color': color})

    def get_lane_change_state(self):
        """MODIFIED: Getter method that checks the timer before resetting the state."""
        # If the state is not NORMAL, check if its active time has expired.
        if self.state != LaneChangeState.NORMAL and self.world.get_snapshot().timestamp.elapsed_seconds > self.state_active_until:
            self.state = LaneChangeState.NORMAL
        return self.state

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
            self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self();
        if not self: return
        self.latitude, self.longitude = event.latitude, event.longitude

    def destroy(self):
        if self.sensor: self.sensor.destroy()


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
            self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self: return
        self.collided = True
        self.actor_type = event.other_actor.type_id if event.other_actor else "unknown"
        impulse = event.normal_impulse
        self.intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        
        location = self._parent.get_location()
        self.hud.event_manager.report('collision', details={'location': location})

    def get_collision_data_and_reset(self):
        data = {"collided": self.collided, "actor_type": self.actor_type, "intensity": self.intensity}
        self.collided, self.actor_type, self.intensity = False, None, 0.0
        return data

    def destroy(self):
        if self.sensor: self.sensor.destroy()


class LaneInvasionSensor(object):
    """Integrates the LaneViolationStateMachine and the LaneManagement system."""

    def __init__(self, parent_actor, hud_instance, lane_manager: LaneManagement = None):
        self.sensor = None
        self._parent = parent_actor
        self.world = self._parent.get_world() # Get world reference
        self.lane_manager = lane_manager
        self.violation_sm = LaneViolationStateMachine(hud_instance)
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find("sensor.other.lane_invasion")
        if bp:
            self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self: return
        
        current_frame = event.frame
        was_handled_as_maneuver = False
        if self.lane_manager:
            was_handled_as_maneuver = self.lane_manager.evaluate_lane_change(event.crossed_lane_markings)

        if not was_handled_as_maneuver:
            self.violation_sm.handle_violation(event.crossed_lane_markings, current_frame)

    def tick(self):
        """
        NEW: This method should be called from your main game loop every frame.
        It allows the violation state machine to check if it should reset itself.
        """
        current_frame = self.world.get_snapshot().frame
        self.violation_sm.tick(current_frame)


    def get_violation_state(self):
        """MODIFIED: Getter method that no longer resets the state."""
        return self.violation_sm.state

    def destroy(self):
        if self.sensor: self.sensor.destroy()
