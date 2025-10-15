import time
import numpy as np
from Core.Vision.VisionPerception import Perception
# ==============================================================================
# -- Event Manager Class (For Visual Notifications) ----------------------------
# ==============================================================================
class EventManager(object):
    """Manages the state and cooldown of VISUAL notifications to prevent spam."""

    def __init__(self, hud_instance):
        self.hud = hud_instance
        self.active_events = {}
        self.event_cooldowns = {
            "collision": 3.0,
            "lane_violation": 2.0,
            "SIGNALLED": 2.0,
            "UNSIGNALLED": 2.0,
            "UNSAFE": 3.0,
            "speeding": 5.0,
            "PROXIMITY_ALERT": 1.5,
        }
        self.collision_distance_threshold = 5.0  # Meters

    def report(self, event_type, details=None):
        """Sensors report events here. Manager decides if a new VISUAL notification is needed."""
        details = details or {}

        # severity-aware key so warning/critical don't suppress each other
        sev = (details.get("severity") or "").lower()  # "warning" | "critical" | ""
        event_key = f"{event_type}:{sev}" if sev else event_type
        if event_key in self.active_events:
            return
        self.active_events[event_key] = {"time": time.time()}

        # message & color
        message = details.get("message", "Event")
        color   = details.get("color")  # can be overridden below
        seconds = float(details.get("seconds", 3.0))

        # map severity -> color/duration defaults
        if event_type == "collision":
            message, seconds = "COLLISION!", 3.0
            color = (255, 0, 0)
            is_center = True
            sound_to_play = "collision"
        elif event_type == "PROXIMITY_ALERT":
            # center flag: accept several aliases coming from perception helper
            is_center = bool(details.get("center") or
                            details.get("_is_critical_center") or
                            details.get("is_critical"))

            if sev == "critical":
                color = color or (255, 0, 0)
                seconds = max(seconds, 3.0)
            elif sev == "warning":
                color = color or (255, 255, 0)
                seconds = max(seconds, 2.5)
            else:
                color = color or (200, 200, 200)
            sound_to_play('PROXIMITY_ALERT')
        else:
            is_center = bool(details.get("center") or details.get("is_critical"))

        # fallbacks
        if color is None:
            color = (255, 255, 0)
        sound_to_play = details.get("sound")

        # show the legacy CENTER notification when requested
        self.hud.notification(
            message, seconds=seconds, text_color=color, is_critical_center=is_center
        )

        if sound_to_play:
            force = (event_type == "collision")
            self.hud.play_sound_for_event(sound_to_play, force_play=force)

    def tick(self):
        current_time = time.time()
        events_to_remove = [
            k
            for k, v in self.active_events.items()
            if current_time > v["time"] + self.event_cooldowns.get(k, 2.0)
        ]
        for k in events_to_remove:
            if k in self.active_events:
                del self.active_events[k]


# ==============================================================================
# -- Other UI Classes (EndScreen, BlinkingAlert, etc.) -------------------------
# ==============================================================================