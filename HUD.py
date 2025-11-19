# ============================================================================
# PERF CHECK (file-level):
# ============================================================================
# [X] | Role: HUD rendering, camera management, vision overlay, bounding boxes
# [X] | Hot-path functions: tick(), render(), bbox_blit(), _image_processor()
# [X] |- Heavy allocs in hot path? YES - Many temp dicts, font renders, surfaces
# [X] |- pandas/pyarrow/json/disk/net in hot path? pandas import but minimal use
# [X] | Graphics here? YES - PRIMARY RENDERER (cameras, overlays, panels)
# [X] | Data produced (tick schema?): Display state only (no logging)
# [X] | Storage (Parquet/Arrow/CSV/none): None (pure rendering)
# [X] | Queue/buffer used?: YES - image_queues for camera frames (threaded)
# [X] | Session-aware? No - stateless renderer per frame
# [X] | Debug-only heavy features?: vision_compare mode, recording, logging spam
# Top 3 perf risks:
# 1. [PERF_HOT] Vision overlay compute() called for EVERY camera tile EVERY frame (L858, L794)
# 2. [PERF_HOT] pygame.transform.smoothscale() on large surfaces every frame (L852)
# 3. [PERF_HOT] Font rendering every frame for dynamic text (no caching of static labels)
# 4. [PERF_HOT] Logging spam: L1005, L1008 (CRITICAL logs every blinker frame)
# 5. [PERF_SPLIT] CameraManager spawns 4+ cameras + threads - no resolution/FPS tuning flags
# ============================================================================

import carla
import pygame
import os
import math
import time
import logging
import weakref
import numpy as np
import Sensors
import queue
import threading
from Utility.Font.FontIconLibrary import IconLibrary, FontLibrary
from Core.Vision.VisionPerception import Perception
from Core.Vision.VisionPerception_MinimalViable import MinimalVisionPerception
from Core.Vision.VisionPerception_MetadataRoute import MetadataPerception
from Core.Vision.VisionPerception_LidarHybrid import LidarHybridPerception, ThreatZone
from EventManager import EventManager
from Helpers import EndScreen, PersistentWarningManager, BlinkingAlert, HelpText
import pygame
import pandas as pd

iLib = IconLibrary()
fLib = FontLibrary()
# ==============================================================================
# -- Notes ---------------------------------------------------------------------
# ==============================================================================
"""
*
This version of the HUD script is modified for a dual-monitor panoramic display.
It assumes the OS is in "Extended Desktop" mode and the Pygame window spans
both monitors.
*
"""

class HUD(object):
    def __init__(self, width, height, args):
        self.dim = (width, height)
        self.event_manager = EventManager(self)
        pygame.init()
        iLib = IconLibrary()
        fLib = FontLibrary()
        self._fov = 90

        self._blinker_left_img, self._blinker_right_img = None, None
        try:
            self._blinker_left_img = pygame.image.load(
                "./images/left_blinker.png"
            ).convert_alpha()
            self._blinker_right_img = pygame.image.load(
                "./images/right_blinker.png"
            ).convert_alpha()
        except pygame.error as e:
            iLib.ilog("warning","Could not load blinker images: {e}.","alerts","wn")


        self._server_clock = pygame.time.Clock()
        self.scale_factor = height / 1080.0
        logging.info(f"Screen-Vehicle Scale Factor{self.scale_factor}")
        iLib.ilog("info", f"Screen-Vehicle Scale Factor: {self.scale_factor}", "sysUI","i")
        """
        # Define base font sizes/
        base_sizes = {
            "title": 12,
            "main_score": 40,
            "sub_label": 40,
            "sub_value": 56,
            "large_val": 28,
            "small_label": 9,
        }

        #### PREVIOUS LOCATION OF self.info_renders

        # Apply the scaling factor to each font size
        scaled_sizes = {k: int(v * scale_factor) for k, v in base_sizes.items()}

        font_path = self.custom_font_path if self._use_custom_font else None
        self.panel_fonts = {
            key: pygame.font.Font(font_path, size) for key, size in scaled_sizes.items()
        }
        """
        self.panel_fonts = fLib.get_loaded_fonts(font="tt-supermolot-neue-trl.bd-it", type="panel_fonts", scale=self.scale_factor)
        ##### Define renderers for each info type   
        self.info_renderers = {
            "title": lambda surf, x, y, val, panel_x, panel_w, padding: self._render_text(
                surf, x, y, val, self.panel_fonts["title"], (200, 200, 200), -5
            ),
            "main_score": lambda surf, x, y, val, panel_x, panel_w, padding: self._render_text(
                surf, x, y, val, self.panel_fonts["main_score"], (255, 255, 255), 15
            ),
            # Custom renderers that need panel geometry for right-alignment
            "sub_label": lambda surf, x, y, val, panel_x, panel_w, padding: self._render_sub_labels(
                surf, x, y, val, panel_x, panel_w, padding
            ),
            "speed_mph": lambda surf, x, y, val, panel_x, panel_w, padding: self._render_speed_and_gear(
                surf, x, y, val, panel_x, panel_w, padding
            ),  # uses speed_mph & gear from self._info_text
        }

        #####
        pygame.mixer.init()
        self.sounds, self.sound_cooldowns = (
            {},
            {
                "collision": 3.0,
                "lane_drift": 2.0,
                "solid_line_crossing": 2.0,
                "speeding": 5.0,
                "error": 1.0,
                'PROXIMITY_ALERT': 3.0
            },
        )
        self._last_sound_time = {k: 0.0 for k in self.sound_cooldowns}
        sound_files = {
            "collision": "./audio/alerts/collision_alert_sound.wav",
            "lane_drift": "./audio/alerts/lane_deviation_sound.wav",
            "solid_line_crossing": "./audio/alerts/solid_line_sound.wav",
            "speeding": "./audio/alerts/speed_violation_sound.wav",
            "error": "./audio/alerts/error_encountered_sound.wav",
            "PROXIMITY_ALERT": ":./audio/alerts/imminent_warning.wav",
            "pedestrian_warning":":./audio/alerts/pedestrian_proximity.wav",
        }
        for k, v in sound_files.items():
            if os.path.exists(v):
                self.sounds[k] = pygame.mixer.Sound(v)

        self.help = HelpText(self.panel_fonts["small_label"], width, height)
        self.server_fps, self.frame, self.simulation_time = 0, 0, 0
        self._show_info, self._info_text, self._active_notifications,self._debug_values, self._sub_label_values = True, {}, [], {}, {}
        self._notification_base_pos_y, self._notification_spacing = (
            int(self.dim[1] * 0.65),
            8,
        )
        self._last_speed_warning_frame_warned = 0
        self.overall_dp_score_display, self.mbi_display, self.lmi_display = (
            100,
            1.0,
            1.0,
        )
        self._blinker_state = 0
        self.warning_manager = PersistentWarningManager(self.dim, self.scale_factor)

        # NEW: notification debounce state
        self._notify_cooldown = 1.0  # seconds  # NEW
        self._last_notify_ts = 0.0  # NEW
        self._notification_duration = 0.0

        self.reset()

        # NEW (08.21.25): Add Flags and MP4 writer
        self.vision_compare = getattr(args, "vision_compare", False)
        self.perception = None
        self._vision_writer = None

        # Store args for perception configuration
        self.args = args
        self.perception_mode = getattr(args, "perception_mode", "lidar-hybrid")
        rec_path = getattr(args, "record_vision_demo", None)
        if rec_path:
            try:
                import imageio.v2 as imageio
                self._vision_writer = imageio.get_writer(rec_path, fps=30, codec="libx264", quality=8)
                logging.info(f"[VisionDemo] Recording to {rec_path}")
            except Exception as e:
                logging.error(f"[VisionDemo] Failed to open writer: {e}")
                self._vision_writer = None
        
        # Object distance thresholds for notifications

    def _initialize_perception(self, world_obj, camera_actor=None):
        """
        Initialize the appropriate perception system based on args.perception_mode

        Args:
            world_obj: World instance with .world and .player
            camera_actor: Camera actor for programmatic mode (optional for others)

        Returns:
            Initialized perception instance
        """
        mode = self.perception_mode
        args = self.args

        logging.info(f"[Perception] Initializing mode: {mode}")

        if mode == "programmatic":
            # Original VisionPerception (legacy)
            perception = Perception(world_obj, camera_actor=camera_actor)
            logging.info(f"[Perception] Programmatic mode initialized (legacy)")

        elif mode == "minimal-viable":
            # MinimalViable: Update/Query separation with persistent tracking
            perception = MinimalVisionPerception(
                world_obj,
                max_distance=getattr(args, "safe_distance", 100.0)
            )
            logging.info(f"[Perception] MinimalViable mode initialized (max_distance={args.safe_distance}m)")

        elif mode == "metadata":
            # Metadata Route: Lightest approach, metadata only
            perception = MetadataPerception(
                world_obj,
                max_distance=getattr(args, "safe_distance", 100.0)
            )
            logging.info(f"[Perception] Metadata mode initialized (max_distance={args.safe_distance}m)")

        elif mode == "lidar-hybrid":
            # LIDAR Hybrid: GPU-accelerated sensor-based detection (recommended)
            perception = LidarHybridPerception(
                world_obj,
                danger_distance=getattr(args, "danger_distance", 15.0),
                caution_distance=getattr(args, "caution_distance", 30.0),
                safe_distance=getattr(args, "safe_distance", 100.0),
                show_danger_bbox=getattr(args, "show_danger_bbox", True),
                show_caution_bbox=getattr(args, "show_caution_bbox", False),
                enable_ground_truth_matching=not getattr(args, "no_ground_truth_matching", False)
            )

            # Attach LIDAR sensor
            lidar_range = getattr(args, "lidar_range", args.safe_distance)
            lidar_pps = getattr(args, "lidar_points_per_second", 56000)
            lidar_freq = getattr(args, "lidar_rotation_frequency", 10.0)

            perception.attach_lidar_sensor(
                lidar_range=lidar_range,
                points_per_second=lidar_pps,
                rotation_frequency=lidar_freq
            )

            logging.info(f"[Perception] LIDAR Hybrid initialized - "
                        f"danger={args.danger_distance}m, caution={args.caution_distance}m, "
                        f"safe={args.safe_distance}m, lidar_range={lidar_range}m, "
                        f"show_danger_bbox={args.show_danger_bbox}, show_caution_bbox={args.show_caution_bbox}")

        else:
            # Fallback to programmatic
            logging.warning(f"[Perception] Unknown mode '{mode}', falling back to programmatic")
            perception = Perception(world_obj, camera_actor=camera_actor)

        return perception

    def _get_perception_objects(self, max_objects=24, include_2d=True, camera_transform=None, camera_intrinsics=None):
        """
        Unified interface to get objects from any perception mode

        Args:
            max_objects: Maximum number of objects to return
            include_2d: Whether to include 2D bounding boxes (expensive for some modes)
            camera_transform: Camera transform for bbox projection
            camera_intrinsics: Camera intrinsics (width, height, fov_deg)

        Returns:
            List of object dicts with {track_id, cls, distance_m, rel_speed_mps, bbox_xyxy}
        """
        if not self.perception:
            return []

        mode = self.perception_mode

        try:
            if mode == "programmatic":
                # Original VisionPerception API
                return self.perception.compute(max_objects=max_objects, include_2d=include_2d)

            elif mode == "minimal-viable":
                # MinimalViable API
                return self.perception.get_objects_as_dict(
                    max_objects=max_objects,
                    include_2d=include_2d,
                    camera_transform=camera_transform,
                    camera_intrinsics=camera_intrinsics
                )

            elif mode == "metadata":
                # Metadata Route API
                objects = self.perception.get_objects(max_objects=max_objects)

                # Optionally add bboxes on-demand (expensive!)
                if include_2d and camera_transform and camera_intrinsics:
                    for obj in objects[:10]:  # Only bbox for closest 10
                        bbox = self.perception.get_bbox_for_object(
                            obj['track_id'],
                            camera_transform,
                            width=camera_intrinsics.get('width', 1920),
                            height=camera_intrinsics.get('height', 1080),
                            fov_deg=camera_intrinsics.get('fov_deg', 90.0)
                        )
                        obj['bbox_xyxy'] = bbox
                else:
                    # No bbox
                    for obj in objects:
                        obj['bbox_xyxy'] = None

                return objects

            elif mode == "lidar-hybrid":
                # LIDAR Hybrid API
                return self.perception.get_clusters_as_dict(
                    zones=[ThreatZone.DANGER, ThreatZone.CAUTION],  # Only danger/caution zones
                    include_bbox=include_2d,
                    camera_transform=camera_transform,
                    camera_intrinsics=camera_intrinsics
                )

            else:
                logging.warning(f"[Perception] Unknown mode '{mode}'")
                return []

        except Exception as e:
            logging.error(f"[Perception] Error getting objects: {e}")
            return []

        # Object distance thresholds for notifications
        self.distance_alerts = {"warning": 20.0,"critical":7.5 }
        # Increase sensitivity for pedestrians
        self.pedestrian_multiplier = 1.40
        # time-to-collision
        self.ttc_alert_s = 2.0
        # min approach speed
        self.approach_min_mps = 0.5

        #per-object alert gating
        self._prox_state = {}
        self._prox_realert_s = 2.0
        self._prox_hysteresis_m = 1.0
        self._scores_frame_dict = {
            'scores': {
                'overall_mvd_score': 0.0,
                'lane_violation_score': 0.0,
                'unsafe_lane_change_score': 0.0,
                'collision_score': 0.0,
                'speed_score': 0.0
            },
            'predictive': {}
        }
        self._scores_df = None
        
    
    def _thresholds_for(self, cls: str):
        warn = float(self.distance_alerts["warning"])
        crit = float(self.distance_alerts["critical"])
        if cls == "pedestrian":
            warn *= self.pedestrian_multiplier
            crit *= self.pedestrian_multiplier
        return warn, crit

    def _zone_for(self, cls: str, dist: float | None, ttc: float | None):
        warn, crit = self._thresholds_for(cls)
        # TTC wins if approaching fast
        if ttc is not None and ttc <= self.ttc_alert_s:
            return "critical"
        if dist is None:
            return "none"
        return "critical" if dist <= crit else ("warning" if dist <= warn else "none")

    def _object_key(self, o: dict):
        # Prefer stable id (CARLA actor id / tracker id); fallback to class+rounded center
        k = o.get("track_id")
        if k is not None:
            return f"obj:{k}"
        bb = o.get("bbox_xyxy")
        if bb:
            cx = int(0.5 * (bb[0] + bb[2]) / 16)  # coarse bins to keep keys stable-ish
            cy = int(0.5 * (bb[1] + bb[3]) / 16)
            return f"{o.get('cls','obj')}:{cx}:{cy}"
        return f"{o.get('cls','obj')}:na"

    def _gate_and_notify(self, o: dict, severity: str, dist: float | None, ttc: float | None):
        """Fire center notification only on zone transitions with cooldown + hysteresis."""
        import time as _time
        cls = o.get("cls", "object")
        key = self._object_key(o)

        # per-frame dedupe (set at top of render loop)
        if hasattr(self, "_seen_frame"):
            if key in self._seen_frame:
                return
            self._seen_frame.add(key)

        zone_now = self._zone_for(cls, dist, ttc)
        key = self._object_key(o)
        st = self._prox_state.get(key, {"zone": "none", "last_alert_t": 0.0})
        zone_prev, last_t = st["zone"], float(st["last_alert_t"])

        # Hysteresis: require stepping past thresholds + hysteresis to de-escalate
        warn, crit = self._thresholds_for(cls)
        if dist is not None:
            if zone_prev == "critical" and dist <= (crit + self._prox_hysteresis_m):
                zone_now = "critical"
            elif zone_prev == "warning" and dist <= (warn + self._prox_hysteresis_m):
                zone_now = max(zone_now, "warning", key=("none","warning","critical").index)

        now = _time.time()
        escalate = (("none","warning","critical").index(zone_now) >
                    ("none","warning","critical").index(zone_prev))
        changed  = (zone_now != zone_prev)

        # Only alert when we enter warning/critical OR escalate (warn->critical), and observe cooldown
        if zone_now in ("warning","critical") and (changed or escalate) and (now - last_t >= self._prox_realert_s):
            # Build message and call EventManager
            msg = f"{zone_now.upper()}: {cls}"
            if dist is not None:
                msg += f" {dist:.1f}m"
            if ttc is not None:
                msg += f" (TTC {ttc:.1f}s)"
            try:
                self.event_manager.report("PROXIMITY_ALERT", {
                    "message": msg,
                    "severity": zone_now,
                    "center": True  # legacy center banner
                })
            except Exception:
                pass
            st["last_alert_t"] = now

        # Save state
        st["zone"] = zone_now
        self._prox_state[key] = st


    def get_dim(self):
        return self.dim
    
    def _render_text(self, surf, x, y, text, font, color, spacing):
        """Draw a single line of text and return the new y-offset."""
        t_surf = font.render(str(text), True, color)
        surf.blit(t_surf, (x, y))
        return y + t_surf.get_height() + spacing

    def _thresholds_for(self, cls: str):
        """Return (warn_m, crit_m) possibly adjusted per class."""
        warn = float(self.distance_alerts["warning"])
        crit = float(self.distance_alerts["critical"])
        if cls == "pedestrian":
            warn *= self.pedestrian_multiplier
            crit *= self.pedestrian_multiplier
        return warn, crit

    def compute_bbox_style(self, o: dict):
        """
        Decide color/severity for a detection dict `o` from Perception.compute().
        Returns (color_rgb, severity_str, label_text, ttc_s_or_None)
        severity_str in {"none","warning","critical"}.
        """
        # base label (you may already set o['label'] in VisionPerception)
        base = o.get("label") or f"{o.get('cls','obj')} {o.get('distance_m',0.0):0.0}m {o.get('rel_speed_mps',0.0):+0.0f}m/s"

        dist = o.get("distance_m")
        relv = o.get("rel_speed_mps")  # signed along LOS; negative ≈ approaching
        cls  = o.get("cls") or "vehicle"

        warn, crit = self._thresholds_for(cls)

        # TTC (if approaching fast enough)
        ttc = None
        if dist is not None and relv is not None and relv < -self.approach_min_mps:
            ttc = dist / (-relv) if dist > 0 else None

        # severity by TTC then distance
        severity = "none"
        color = (0, 255, 0)  # green = safe
        if ttc is not None and ttc <= self.ttc_alert_s:
            severity = "critical"; color = (255, 0, 0)
            base += f"  TTC {ttc:.1f}s"
        elif dist is not None:
            if dist <= crit:
                severity = "critical"; color = (255, 0, 0)
            elif dist <= warn:
                severity = "warning"; color = (255, 255, 0)

        # nicer label color: match box color for quick scanning
        return color, severity, base, ttc
    
    def _maybe_notify_proximity(self, o: dict, severity: str, dist: float, ttc: float | None):
        """
        Fire your existing EventManager notification with cooldowns.
        Keep the payload simple so it fits your current handler.
        """
        if severity == "none" or not hasattr(self, "event_manager"):
            return
        cls = o.get("cls", "object")
        msg = f"{severity.upper()}: {cls} at {dist:.1f}m"
        if ttc is not None:
            msg += f" (TTC {ttc:.1f}s)"
        try:
            # your EventManager already de-bounces; we include severity for styling
            self.event_manager.report("PROXIMITY_ALERT", {
                "message": msg,
                "severity": severity,
                "_is_critical_center": True,
                "center":True
            })
        except Exception:
            pass
    
    def bbox_blit(self, surf, o: dict, font):
        bb = o.get("bbox_xyxy")
        if not bb:
            return
        x1, y1, x2, y2 = map(int, bb)

        # compute label + severity + color (you can keep your own logic here)
        dist = o.get("distance_m")
        relv = o.get("rel_speed_mps")
        cls  = o.get("cls", "vehicle")
        # TTC
        ttc = None
        if dist is not None and relv is not None and relv < -self.approach_min_mps:
            ttc = dist / (-relv) if dist > 0 else None

        # severity from zone
        zone = self._zone_for(cls, dist, ttc)
        if zone == "critical":
            color = (255, 0, 0)
        elif zone == "warning":
            color = (255, 255, 0)
        else:
            color = (0, 255, 0)

        # draw
        import pygame
        pygame.draw.rect(surf, color, pygame.Rect(x1, y1, x2 - x1, y2 - y1), 2)
        label = o.get("label") or f"{cls} {dist:.1f}m {relv:+.1f}m/s" if dist is not None and relv is not None else (o.get("label") or cls)
        surf.blit(font.render(label, True, color), (x1, max(0, y1 - 18)))

        # alert (center legacy) — gated
        self._gate_and_notify(o, zone, dist, ttc)


    def _render_sub_labels(self, surf, x, y, sub_dict, panel_x, panel_w, padding):
        """Render a left/right list of label:value pairs inside the info panel.
        - Labels are left-aligned at (x, y)
        - Values are right-aligned within the panel (panel_x..panel_x+panel_w)
        Returns the updated y-offset.
        """
        for sub_key, sub_value in (sub_dict or {}).items():
            label_surf = self.panel_fonts["sub_label"].render(str(sub_key), True, (200, 200, 200))
            value_surf = self.panel_fonts["sub_value"].render(str(sub_value), True, (255, 255, 255))
            # Draw label on the left
            surf.blit(label_surf, (x, y))
            # Right-align value within the panel
            value_x = panel_x + panel_w - value_surf.get_width() - padding
            surf.blit(value_surf, (value_x, y))
            y += max(label_surf.get_height(), value_surf.get_height()) + 10
        return y


    def _render_speed_and_gear(self, surf, x, y, _ignored, panel_x, panel_w, _padding):
        """Render speed (MPH) on the left and gear on the right of the info panel row.
        Pulls values from self._info_text["speed_mph"] and ["gear"].
        Returns the updated y-offset.
        """
        speed_val = str(self._info_text.get("speed_mph", "0"))
        gear_val = str(self._info_text.get("gear", "N"))

        speed_surf = self.panel_fonts["large_val"].render(speed_val, True, (255, 255, 255))
        mph_surf = self.panel_fonts["small_label"].render("MPH", True, (200, 200, 200))
        surf.blit(speed_surf, (x, y))
        surf.blit(mph_surf, (x + speed_surf.get_width() + 5, y + 30))

        gear_surf = self.panel_fonts["large_val"].render(gear_val, True, (255, 255, 255))
        gear_label_surf = self.panel_fonts["small_label"].render("GEAR", True, (200, 200, 200))
        # Right side of panel
        gear_x = panel_x + panel_w - gear_surf.get_width() - 20
        surf.blit(gear_surf, (gear_x, y))
        surf.blit(gear_label_surf, (gear_x, y + gear_surf.get_height()))

        return y + max(speed_surf.get_height(), gear_surf.get_height()) + 10

    def reset(self):
        self.overall_dp_score_display, self.mbi_display, self.lmi_display = (
            100,
            1.0,
            1.0,
        )
        self._active_notifications, self._blinker_state = [], 0
        self._last_speed_warning_frame_warned = 0
        self.event_manager.active_events.clear()
        self._last_sound_time = {k: 0.0 for k in self.sound_cooldowns}

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps, self.frame, self.simulation_time = (
            self._server_clock.get_fps(),
            timestamp.frame,
            timestamp.elapsed_seconds,
        )

    def play_sound_for_event(self, event_type, force_play=False):
        sound = self.sounds.get(event_type)
        if not sound:
            return
        current_time = time.time()
        cooldown = self.sound_cooldowns.get(event_type, 0.0)
        if force_play or (
            current_time > self._last_sound_time.get(event_type, 0.0) + cooldown
        ):
            sound.play()
            self._last_sound_time[event_type] = current_time

    def update_mvd_scores_for_display(self, data_ingestor):
        self._scores_frame_dict = data_ingestor.get_last_logged_frame()
        if self._scores_df is None:
            return
        
#        self.overall_dp_score_display, self.mbi_display, self.lmi_display = (
#            dp_score,
#            mbi,
#            lmi,
#        )
 
    def update_predictive_indices(self, indices):
        """Update predictive indices for display"""
        self._predictive_indices = indices

    def toggle_info(self):
        self._show_info = not self._show_info

    # -------------------------------------------
    # Notifications, Debouncer (eliminate spamming)
    # -------------------------------------------
    # Update: added text_color and is_critical_center keyword-only args
    # Update: continues to debounce using _notify_cooldown / _last_notify_ts
    # Update: refactored code for queue

    # Deals with notification spamming - subsequent simulation crashes.
    def notification(self, text: str, seconds: float = 2.0, text_color=(255,255,255), is_critical_center=False):
        now = time.time()
        if (now - self._last_notify_ts) < self._notify_cooldown:
            return
        self._last_notify_ts = now
        self._enqueue_notification(text, seconds, text_color=text_color, is_critical_center=is_critical_center)


    # Notification Queue, accepts 4 tuples per entry
    def _enqueue_notification(self, text: str, seconds: float, *, text_color=(255,255,255), is_critical_center=False):
        if not hasattr(self, '_messages'):
            self._messages = []
        expires_at = time.time() + float(seconds)
        self._messages.append((expires_at, str(text), tuple(text_color), bool(is_critical_center)))

    # Draw queued notifications (panel-local; center left to legacy)
    # EDIT: draw only panel-local messages here (is_center == False), supports per-message text color
    # legacy _active_notifications loop can still render center/critical messages for overlays.

    def _draw_notifications(self, surf, x, y):
        if not getattr(self, '_messages', None):
            return y
        now = time.time()
        keep = []
        for expires_at, text, color, is_center in self._messages:
            if expires_at > now:
                if not is_center:
                    t_surf = self.panel_fonts['small_label'].render(text, True, color)
                    surf.blit(t_surf, (x, y))
                    y += t_surf.get_height() + 6
                keep.append((expires_at, text, color, is_center))
        self._messages = keep
        return y
    
    #Draw queued center/critical messages over Screen 2 using cached fonts
    def _draw_center_notifications(self, surf):    
        if not getattr(self, '_messages', None):
            return
        now = time.time()
        keep = []
        single_screen_width = self.dim[0] // 4
        main_screen_start_x = single_screen_width
        center_x = main_screen_start_x + (single_screen_width // 2)
        y_off = getattr(self, '_notification_base_pos_y', int(self.dim[1] * 0.85))

        for expires_at, text, color, is_center in reversed(self._messages):
            if expires_at > now:
                if is_center:
                    # big, centered banner using a cached font
                    t_surf = self.panel_fonts['large_val'].render(text, True, color)
                    x_pos = center_x - (t_surf.get_width() // 2)
                    y_pos = y_off - t_surf.get_height()
                    if y_pos < self.dim[1] * 0.15:
                        break
                    surf.blit(t_surf, (x_pos, y_pos))
                    y_off -= t_surf.get_height() + getattr(self, '_notification_spacing', 8)
                keep.append((expires_at, text, color, is_center))
        self._messages = keep

    def draw_3d_bounding_box(self, display, camera, bounding_box, world_transform, color=(0, 255, 0)):
        """
        Projects the 8 vertices of a 3D bounding box into the 2D camera view
        and draws lines to visualize it.
        """
        # Manually construct the camera's calibration matrix
        image_w = int(camera.attributes.get('image_size_x'))
        image_h = int(camera.attributes.get('image_size_y'))
        fov = float(camera.attributes.get('fov'))
        
        calibration = np.identity(3)
        calibration[0, 2] = image_w / 2.0
        calibration[1, 2] = image_h / 2.0
        calibration[0, 0] = calibration[1, 1] = image_w / (2.0 * math.tan(fov * math.pi / 360.0))

        camera_transform = camera.get_transform()
        box_vertices = bounding_box.get_world_vertices(world_transform)

        # --- FIX: Manually perform the matrix transformation using NumPy ---
        # Get the inverse matrix as a NumPy array
        world_to_camera_matrix = np.array(camera_transform.get_inverse_matrix())
        
        points_2d = []
        for vertex in box_vertices:
            # Create a homogeneous coordinate for the 3D world point
            point_in_world = np.array([vertex.x, vertex.y, vertex.z, 1.0])
            
            # Transform world point to camera's local space via matrix multiplication
            point_in_camera_homogeneous = world_to_camera_matrix.dot(point_in_world)
            
            # The result is a 4D vector, we only need the first 3 components for the location
            p_camera = carla.Location(x=point_in_camera_homogeneous[0], y=point_in_camera_homogeneous[1], z=point_in_camera_homogeneous[2])
            # --- End of FIX ---
            
            # Project 3D point to 2D image plane using the calculated K matrix
            p_image = np.array([p_camera.x, p_camera.y, p_camera.z])
            p_image = np.dot(calibration, p_image)
            
            if p_image[2] > 0: # Check if the point is in front of the camera
                p_image = np.array([p_image[0] / p_image[2], p_image[1] / p_image[2]])
                points_2d.append(p_image)
            else:
                points_2d.append(None)

        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0], # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4], # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]  # Connecting sides
        ]

        for edge in edges:
            p1 = points_2d[edge[0]]
            p2 = points_2d[edge[1]]
            if p1 is not None and p2 is not None:
                pygame.draw.line(display, color, (p1[0], p1[1]), (p2[0], p2[1]), 2)


    def error(self, text):
        self.notification(f"ERROR: {text.upper()}", 5.0, (255, 50, 50), True)
        self.play_sound_for_event("error", force_play=True)

    def get_vehicle_rpm(self,vehicle):
        """
            Calculates an estimated RPM for a CARLA vehicle.
        """
        # Get the forward speed of the vehicle in m/s
        velocity = vehicle.get_velocity()
        forward_speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        # Get the vehicle's physics control settings
        physics_control = vehicle.get_physics_control()

        # Find the current gear of the vehicle from its control state
        vehicle_control = vehicle.get_control()
        current_gear = vehicle_control.gear

        # Use gear 1 if the vehicle is in neutral (gear 0) and moving
        if current_gear == 0:
            current_gear = 1

        if current_gear > 0:
            # Get the gear ratio for the current gear
            # Note: gear numbers are 1-based, but list is 0-based
            gear_ratio = physics_control.forward_gears[current_gear - 1].ratio
            
            # Get final drive ratio and wheel radius
            final_ratio = physics_control.final_ratio
            wheel_radius = physics_control.wheels[0].radius / 100.0 # Convert cm to meters
            
            if wheel_radius > 0 and gear_ratio > 0:
                # Calculate wheel RPM
                wheel_rpm = (forward_speed * 60) / (2 * math.pi * wheel_radius)
                
                # Calculate engine RPM
                engine_rpm = wheel_rpm * gear_ratio * final_ratio
                return engine_rpm
            
    def _render_predictive_panel(self, display, x, y):
        """Render predictive safety indices"""
        if not self._predictive_indices:
            return y
            
        # Title
        title_surf = self.panel_fonts["sub_label"].render("Predictive Safety", True, (200, 200, 200))
        display.blit(title_surf, (x, y))
        y += title_surf.get_height() + 5
        
        # Display key indices with color coding
        for key, value in self._predictive_indices.items():
            if key in ['risk_score', 'collision_probability']:
                # Color code based on risk level
                if value > 0.7:
                    color = (255, 50, 50)  # Red for high risk
                elif value > 0.4:
                    color = (255, 200, 0)  # Yellow for medium
                else:
                    color = (0, 255, 0)    # Green for low
                    
                label = key.replace('_', ' ').title()
                text = f"{label}: {value:.2f}"
                surf = self.panel_fonts["small_label"].render(text, True, color)
                display.blit(surf, (x + 10, y))
                y += surf.get_height() + 3
                
        return y
    

    # [PERF_HOT] Called every frame from Main game loop
    def tick(self, world_instance, clock, idling , controller, display_fps):
        """
        Processes actions, states for each frame or "tick" of the simulation.  For the HUD
        this is processing the current state information
        """
        self.event_manager.tick()
        self.world = world_instance

        try:
            cm = self.world.camera_manager
            cam_actor = cm.sensors.get('left_dash_cam')  # second tile ("main")
            if cam_actor:
                if not hasattr(self, 'perception') or self.perception is None:
                    # Use factory method to initialize correct perception mode
                    self.perception = self._initialize_perception(self.world, camera_actor=cam_actor)

                    # Log perception info (mode-dependent)
                    if hasattr(self.perception, 'fov_deg'):
                        # Programmatic mode has camera info
                        logging.info(f"[Vision] bound to left_dash FOV={self.perception.fov_deg:.1f} "
                                    f"size={self.perception.image_width}x{self.perception.image_height}")
                else:
                    # Update camera if perception already exists (programmatic mode only)
                    if hasattr(self.perception, 'set_camera'):
                        self.perception.set_camera(cam_actor)

            # Update perception state once per tick (for new perception modes)
            if self.perception and hasattr(self.perception, 'update'):
                self.perception.update()

        except Exception as e:
            logging.debug(f"[Vision] bind skipped: {e}")
        self.control = controller
        self._blinker_state = self.control.get_blinker_state()
        if self._blinker_state !=0:
            print(f'blinker state: {self._blinker_state}')
        if self.control:
            new_state = controller.get_blinker_state()
            if new_state != self._blinker_state:
                iLib.log('info',f"HUD blinker state changed: {self._blinker_state} -> {new_state}",'items','js',6)
            self._blinker_state = new_state

        self._active_notifications = [
            n for n in self._active_notifications if n.tick(clock)
        ]

        """
        if self.vision_compare and self.world and self.world.camera_manager:
            cm = self.world.camera_manager
            cam_actor = cm.sensors.get('left_dash_cam') or cm.get_camera_actor_for_queue('main')
            if cam_actor:
                w, h = cm.single_monitor_dim
                if self.perception is None:
                    self.perception = Perception(self.world, image_width=w, image_height=h, fov_deg = self._fov, camera_actor=cam_actor)
                else:
                    self.perception.set_camera(cam_actor, image_width=w, image_height=h)
        """


        self._info_text = {}
        if self._show_info and world_instance.player and world_instance.player.is_alive:
            v = world_instance.player.get_velocity()
            c = world_instance.player.get_control()
            v_physics = world_instance.player.get_physics_control()
            rpm = self.get_vehicle_rpm(world_instance.player)
            speed_mph = int(2.237 * v.length())

            if controller.is_parked():
                gear = "P"
            else:
                gear = (
                    {-1: "R", 0: "N"}.get(c.gear, str(c.gear))
                    if isinstance(c, carla.VehicleControl)
                    else "N/A"
                )

            self._sub_label_values = {
                "Vehicle Control":f"{self.mbi_display:.2f}",
                "Lane Management":f"{self.lmi_display:.2f}",                
            }

            accel = world_instance.player.get_acceleration()
            ang_vel = world_instance.player.get_angular_velocity()
            rpm = self.get_vehicle_rpm(world_instance.player)
            ackermann_settings = world_instance.player.get_ackermann_controller_settings()

            steer = self.control._ackermann_control.steer
            clamped_steer = self.control._clamped_steer
            self._debug_values = {
                "Server FPS": f"{self.server_fps:.0f}",
                "Render FPS": f"{display_fps:.0f}"
            }
            """
            self._debug_values = {
                "Speed (MPH)": f"{speed_mph}",
                "Acceleration": f"{accel.x:.2f}, {accel.y:.2f}, {accel.z:.2f}",
                "Angular Vel": f"{ang_vel.x:.2f}, {ang_vel.y:.2f}, {ang_vel.z:.2f}",
                "RPM": f"{rpm:.0f}" if rpm is not None else "N/A",
                "Steer": f"{steer: .2f}",
            }
            """

            self._info_text = {
                "title":"Overall Score",
                "main_score":f"{self._scores_frame_dict['scores']['overall_mvd_score']:.0f}",
                "sub_label": self._sub_label_values,
                "speed_mph":speed_mph,
                "RPM": f"{rpm:.0f}" if rpm is not None else "N/A",
                "coll_score": self._scores_frame_dict['scores'].get('PSS_ProactiveSafety', 0),
                "lane_score": f"{self._scores_frame_dict['scores'].get('LDS_LaneDiscipline', 0)}",
                "harsh_score": f"{self._scores_frame_dict['scores'].get('DSS_DrivingSmoothness', 0)}",
#                "gear":gear,
#                "debug_info":self._debug_values,
            }
        else:
            self._info_text = {"title": "Player not ready"}
        for name, parameter in vars(ackermann_settings).items():
            self._info_text[name] = parameter

    # [PERF_HOT] Main rendering function - called every frame
    def render(self, display):
        """
        Renders cameras and the HUD.
        - FIX: Restructured to ensure the main HUD panel always renders.
        - FIX: Updates the perception module with the correct camera for each panel.
        - FIX: Corrected function call to get camera image arrays.
        - FIX: Removed redundant/incorrect drawing code and enabled overlays on all panels.
        """
        # --- 1) Render either the Vision Comparison view OR the Panoramic view ---
        
        # Vision compare mode: LEFT = raw front-left cam, RIGHT = same with overlay
        if self.vision_compare and self.world and self.world.camera_manager:
            cm = self.world.camera_manager
            W, H = cm.single_monitor_dim
            left_arr = cm.get_latest_array('main')

            if left_arr is not None:
                # Left (raw)
                left_surf = pygame.surfarray.make_surface(left_arr.swapaxes(0, 1)).convert()
                display.blit(left_surf, (0, 0))

                # Right (overlay)
                right_surf = left_surf.copy()
                if getattr(self, 'perception', None):
                    cam_actor = cm.get_camera_actor_for_queue('main') or cm.sensors.get('left_dash_cam')
                    if cam_actor:
                        # Update camera for programmatic mode
                        if hasattr(self.perception, 'set_camera'):
                            self.perception.set_camera(cam_actor)

                    # Get camera info for projection
                    camera_transform = cam_actor.get_transform() if cam_actor else None
                    camera_intrinsics = {
                        'width': W,
                        'height': H,
                        'fov_deg': getattr(self, '_fov', 90.0)
                    }

                    # Use unified interface (supports all perception modes)
                    objs = self._get_perception_objects(
                        max_objects=24,
                        include_2d=True,
                        camera_transform=camera_transform,
                        camera_intrinsics=camera_intrinsics
                    )

                    if objs:
                        font = self.panel_fonts.get('small_label', pygame.font.Font(None, 16))
                        px_w = getattr(self.perception, 'image_width', right_surf.get_width()) or right_surf.get_width()
                        px_h = getattr(self.perception, 'image_height', right_surf.get_height()) or right_surf.get_height()
                        sx = right_surf.get_width()  / float(px_w)
                        sy = right_surf.get_height() / float(px_h)

                        for o in objs:
                            bb = o.get("bbox_xyxy")
                            if not bb:
                                continue
                            x1 = int(bb[0] * sx); y1 = int(bb[1] * sy)
                            x2 = int(bb[2] * sx); y2 = int(bb[3] * sy)
                            self.bbox_blit(right_surf, {**o, "bbox_xyxy": [x1, y1, x2, y2]}, font)

                if self._vision_writer is not None:
                    try:
                        right_np = pygame.surfarray.array3d(right_surf).swapaxes(0, 1)
                        frame = np.concatenate([left_arr, right_np], axis=1)
                        self._vision_writer.append_data(frame)
                    except Exception as e:
                        logging.error(f"[VisionDemo] frame write failed: {e}")
        
        elif hasattr(self, 'world') and self.world and self.world.player and self.world.camera_manager:
            cm = self.world.camera_manager
            W, H = cm.single_monitor_dim

            panels = [
                ('left_side_cam',  0 * W),
                ('left_dash_cam',  1 * W),
                ('right_dash_cam', 2 * W),
                ('right_side_cam', 3 * W),
            ]

            # seen-per-frame guard so the same actor spotted by multiple cams doesn’t spam
            self._seen_frame = set()

            for name, xoff in panels:
                cam_actor = cm.sensors.get(name)
                if not cam_actor:
                    continue

                # Get latest RGB frame for this camera's queue
                try:
                    queue_key = cm.config['panoramic_setup'][name]['queue']
                except Exception:
                    queue_key = 'main'

                arr = cm.get_latest_array(queue_key)

                # Build surface (camera-native) → scale once to tile size
                if arr is None:
                    surf = pygame.Surface((W, H)).convert()
                    surf.fill((10, 10, 10))
                else:
                    native = pygame.surfarray.make_surface(arr.swapaxes(0, 1)).convert()
                    # [PERF_HOT] smoothscale is expensive on large surfaces - consider pre-scaling cameras
                    surf = pygame.transform.smoothscale(native, (W, H)) if native.get_size() != (W, H) else native

                # ---- Vision overlay ON THIS TILE ----
                if getattr(self, 'perception', None):
                    # use this tile's camera for pose/FOV so boxes align
                    if hasattr(self.perception, 'set_camera'):
                        self.perception.set_camera(cam_actor)

                    # Get camera info for projection
                    camera_transform = cam_actor.get_transform() if cam_actor else None
                    camera_intrinsics = {
                        'width': surf.get_width(),
                        'height': surf.get_height(),
                        'fov_deg': getattr(self, '_fov', 90.0)
                    }

                    # [PERF_HOT] CRITICAL: This runs for ALL 4 camera tiles EVERY FRAME!
                    # With LIDAR Hybrid, this is now much faster (1-2ms vs 40-60ms)
                    objs = self._get_perception_objects(
                        max_objects=24,
                        include_2d=True,
                        camera_transform=camera_transform,
                        camera_intrinsics=camera_intrinsics
                    )

                    if objs:
                        font = self.panel_fonts.get('small_label', pygame.font.Font(None, 16))
                        # scale bboxes if the perception intrinsics != surf size
                        px_w = getattr(self.perception, 'image_width', surf.get_width()) or surf.get_width()
                        px_h = getattr(self.perception, 'image_height', surf.get_height()) or surf.get_height()
                        sx = surf.get_width()  / float(px_w)
                        sy = surf.get_height() / float(px_h)

                        for o in objs:
                            bb = o.get("bbox_xyxy")
                            if not bb:
                                continue
                            x1 = int(bb[0] * sx); y1 = int(bb[1] * sy)
                            x2 = int(bb[2] * sx); y2 = int(bb[3] * sy)
                            # draw + (gated) notify on this tile
                            self.bbox_blit(surf, {**o, "bbox_xyxy": [x1, y1, x2, y2]}, font)

                # Blit the (possibly annotated) tile
                display.blit(surf, (xoff, 0))
        # --- 2) Render the HUD panel, notifications, etc. (This code now runs always) ---
        if self._show_info and getattr(self, '_info_text', None):
            # Layout across 4 screens
            main_screen_offset_x = self.dim[0] // 4
            single_screen_width = self.dim[0] // 4

            panel_h = int(self.dim[1] * 0.4)
            panel_w = int(single_screen_width * 0.25)
            panel_x = main_screen_offset_x + (single_screen_width - panel_w) / 2
            panel_y = int(self.dim[1] * 0.58)
            panel_bg_color = (20, 20, 20, 150)
            panel_border_color = (0, 150, 255, 200)
            padding = math.floor(20*self.scale_factor)

            info_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
            pygame.draw.rect(info_surf, panel_bg_color, info_surf.get_rect(), border_radius=15)
            pygame.draw.rect(info_surf, panel_border_color, info_surf.get_rect(), width=2, border_radius=15)
            display.blit(info_surf, (panel_x, panel_y))

            v_offset = panel_y + padding
            h_offset = panel_x + padding

            for key, value in self._info_text.items():
                renderer = self.info_renderers.get(key)
                if renderer:
                    v_offset = renderer(display, h_offset, v_offset, value, panel_x, panel_w, padding)
                else:
                    v_offset = self._render_text(
                        display, h_offset, v_offset, f"{key}: {value}", self.panel_fonts["small_label"], (200, 200, 200), 10
                    )
            
            v_offset = self._draw_notifications(display, h_offset, v_offset)
            self._draw_center_notifications(display)

            panel_rect = pygame.Rect(int(panel_x), int(panel_y), int(panel_w), int(panel_h))
            self.warning_manager.render(display, panel_rect, v_offset)
        
        if self._predictive_indices:
            self._render_predictive_panel(display, h_offset, v_offset+math.floor(20*self.scale_factor))

        # Legacy fullscreen notifications
        if hasattr(self, '_active_notifications') and self._active_notifications:
            y_off = getattr(self, '_notification_base_pos_y', int(self.dim[1] * 0.85))
            for notif in reversed(self._active_notifications):
                if not notif.seconds_left > 0 and notif.surface.get_alpha() == 0:
                    continue
                if getattr(notif, 'is_critical_center', False):
                    notif.render(display)
                else:
                    single_screen_width = self.dim[0] // 4
                    main_screen_start_x = single_screen_width
                    x_pos = main_screen_start_x + (single_screen_width - notif.surface.get_width()) / 2
                    y_pos = y_off - notif.surface.get_height()
                    if y_pos < self.dim[1] * 0.15:
                        break
                    display.blit(notif.surface, (x_pos, y_pos))
                    y_off -= notif.surface.get_height() + getattr(self, '_notification_spacing', 8)
        self._render_blinker_indicator(display)
        if self._render_rearview_pip:
            self._render_rearview_pip(display)


    def _render_rearview_pip(self, display):
        try:
            if hasattr(self, 'world') and self.world and self.world.camera_manager:
                cm = self.world.camera_manager
                with cm.array_lock:
                    rv = cm.processed_arrays.get('rearview')
                if rv is not None:
                    img = pygame.surfarray.make_surface(rv.swapaxes(0, 1))
                    img = pygame.transform.flip(img, True, False)

                    bezel = 8
                    shadow = 4
                    final_w = cm.rearview_res_w + bezel*2 + shadow
                    final_h = cm.rearview_res_h + bezel*2 + shadow
                    final = pygame.Surface((final_w, final_h), pygame.SRCALPHA)
                    pygame.draw.rect(final, (0,0,0,70), pygame.Rect(shadow, shadow, final_w-shadow, final_h-shadow), border_radius=20)
                    pygame.draw.rect(final, (20,20,20,220), pygame.Rect(0,0,final_w-shadow,final_h-shadow), border_radius=10)
                    pygame.draw.rect(final, (100,100,100,100), pygame.Rect(0,0,final_w-shadow,final_h-shadow), 1, border_radius=10)
                    final.blit(img, (bezel, bezel))

                    # panel 3 (right-dash) geometry
                    panel_w = self.dim[0] // 4
                    panel_left = 2 * panel_w
                    x_pos = int(panel_left + (panel_w - final.get_width()) // 2)  # centered on panel 3
                    y_pos = 20  # margin from top

                    # sanity: draw a faint backdrop so you can see it even over bright video
                    pygame.draw.rect(display, (0,0,0,60),
                                    pygame.Rect(x_pos-6, y_pos-6, final.get_width()+12, final.get_height()+12), border_radius=14)

                    display.blit(final, (x_pos, y_pos))  # NOTE: (x, y) order
        except Exception as _e:
            pass

    def _surf_from_queue(self, cm, queue_key):
        with cm.array_lock:
            arr = cm.processed_arrays.get(queue_key)
        if arr is None:
            s = pygame.Surface(cm.single_monitor_dim).convert()
            s.fill((10,10,10))
            return s
        return pygame.surfarray.make_surface(arr.swapaxes(0,1)).convert()


    def _render_blinker_indicator(self, display):
        import pygame
        if self._blinker_state == 0:
            return
        # simple blink gate
        if (pygame.time.get_ticks() // 750) % 2 == 0:
            return

        W, H = self.dim
        panel_w    = W // 4
        panel_left = 1 * panel_w              # second panel
        cx         = panel_left + panel_w//2
        y          = int(H * 0.80)

        left_x  = int(cx - panel_w * 0.10)
        right_x = int(cx + panel_w * 0.10)

        if self._blinker_state == 1 and self._blinker_left_img:
            display.blit(self._blinker_left_img,
                        self._blinker_left_img.get_rect(center=(left_x, y)))
            # [PERF_HOT][DEBUG_ONLY] CRITICAL: Logs EVERY frame blinker is on! Remove or gate with --debug
            logging.critical(f"blinker triggered left ⬅️ location: X={left_x},y= {y}")

        elif self._blinker_state == 2 and self._blinker_right_img:
            # [PERF_HOT][DEBUG_ONLY] CRITICAL: Logs EVERY frame blinker is on! Remove or gate with --debug
            logging.critical(f"blinker triggered right ▶️ location: X={right_x},y= {y}")
            display.blit(self._blinker_right_img,
                        self._blinker_right_img.get_rect(center=(right_x, y)))

        elif self._blinker_state == 3:
            if self._blinker_left_img:
                display.blit(self._blinker_left_img,
                            self._blinker_left_img.get_rect(center=(left_x, y)))
            if self._blinker_right_img:
                display.blit(self._blinker_right_img,
                            self._blinker_right_img.get_rect(center=(right_x, y)))



class CameraManager(object):
    def __init__(self, parent_actor, hud, fov):
        # --- Standard Initialization ---
        self._parent = parent_actor
        self._actor_model = self._parent.attributes.get('ros_name')
        self.hud = hud
        self.dim = hud.get_dim()
        self.single_monitor_dim = (self.dim[0] // 4, self.dim[1])
        self.world = parent_actor.get_world()                          # [ADD]

        # --- Threading and Queue Setup ---
        self.array_lock = threading.Lock()
        self.stop_threads = threading.Event()
        self.threads = []
        self.image_queues = {
            'main': queue.Queue(maxsize=1),
            'left_side': queue.Queue(maxsize=1),
            'right_dash': queue.Queue(maxsize=1),
            'right_side': queue.Queue(maxsize=1),
            'rearview': queue.Queue(maxsize=1)
        }
        self.processed_arrays = {k: None for k in self.image_queues}

        # --- Get vehicle-specific configuration ---
        self.config = self._get_vehicle_camera_config()                # [KEEP]
        self.view_sets = self.config['view_sets']
        self.view_index = 0                                            # [FIX] unify on view_index

        self.rearview_res_w = self.config['rearview_setup'].get('rearview_res_w')
        self.rearview_res_h = self.config['rearview_setup'].get('rearview_res_h')

        # --- Storage for sensors & seg maps ---
        self.sensors = {}                                              # [MOVE UP] used immediately below
        self.seg_sensors = {}                                          # [ADD]
        self.seg_tags = {}                                             # [ADD] name -> uint8 [H,W]
        self.seg_inst = {}                                             # [ADD] name -> uint16 [H,W]

        # --- Build camera name ↔ queue maps (from config) ---
        self.name_to_queue = {n: s['queue'] for n, s in self.config['panoramic_setup'].items()}
        self.queue_to_name = {}
        for n, q in self.name_to_queue.items():
            self.queue_to_name.setdefault(q, n)

        # --- Panoramic RGB spawn (single loop) + optional SEG attach per camera ---
        bp_library = self.world.get_blueprint_library()
        initial_view_set = self.view_sets[self.view_index]

        for name, settings in self.config['panoramic_setup'].items():
            # RGB
            cam_bp = bp_library.find("sensor.camera.rgb")
            cam_bp.set_attribute("image_size_x", str(self.single_monitor_dim[0]))
            cam_bp.set_attribute("image_size_y", str(self.single_monitor_dim[1]))
            cam_bp.set_attribute("fov", str(settings['fov']))

            # minor perf tweak for side cams
            #if 'side' in name:
            #    cam_bp.set_attribute("enable_postprocess_effects", "false")

            xform = initial_view_set[name]
            rgb = self.world.spawn_actor(cam_bp, xform, attach_to=self._parent)
            self.sensors[name] = rgb

            queue_key = settings['queue']
            rgb.listen(lambda image, key=queue_key: self._add_to_queue(key, image))
            self.threads.append(threading.Thread(target=self._image_processor, args=(queue_key,)))

            # SEG (optional, matched intrinsics so pixels align 1:1 with RGB)
            if settings.get("seg", False):
                seg_bp = bp_library.find("sensor.camera.semantic_segmentation")
                seg_bp.set_attribute("image_size_x", str(self.single_monitor_dim[0]))
                seg_bp.set_attribute("image_size_y", str(self.single_monitor_dim[1]))
                seg_bp.set_attribute("fov", str(settings['fov']))
                tick = float(settings.get("seg_tick", 0.0))
                if tick > 0.0:
                    seg_bp.set_attribute("sensor_tick", str(tick))

                seg = self.world.spawn_actor(seg_bp, xform, attach_to=self._parent)
                self.seg_sensors[name] = seg

                # capture current name to avoid late-binding
                seg.listen(lambda img, _n=name: self._on_seg_image(_n, img))

        # --- Rearview Camera (single block; remove duplicate) ---
        self.rearview_cam = None
        rear_config = self.config.get('rearview_setup')
        if rear_config:
            rear_bp = bp_library.find("sensor.camera.rgb")
            rear_bp.set_attribute("image_size_x", str(rear_config['rearview_res_w']))
            rear_bp.set_attribute("image_size_y", str(rear_config['rearview_res_h']))
            rear_bp.set_attribute("fov", str(rear_config['fov']))
            rear_bp.set_attribute("enable_postprocess_effects", "false")
            self.rearview_cam = self.world.spawn_actor(rear_bp, rear_config['transform'], attach_to=self._parent)
            self.rearview_cam.listen(lambda image: self._add_to_queue('rearview', image))
            self.threads.append(threading.Thread(target=self._image_processor, args=('rearview',)))

        # --- Start all threads ---
        for t in self.threads:
            t.daemon = True
            t.start()

    # SEG callback + accessor
    def _on_seg_image(self, cam_name: str, image):
        import numpy as np
        h, w = int(image.height), int(image.width)
        buf = np.frombuffer(image.raw_data, np.uint8).reshape(h, w, 4)
        tags = buf[:, :, 2].copy()                                              # R = semantic tag id
        inst = (buf[:, :, 1].astype(np.uint16) << 8) | buf[:, :, 0].astype(np.uint16)  # (G<<8)|B
        with self.array_lock:
            self.seg_tags[cam_name] = tags
            self.seg_inst[cam_name] = inst

    def get_seg_maps(self, cam_name: str):
        with self.array_lock:
            return self.seg_tags.get(cam_name), self.seg_inst.get(cam_name)

    def _get_vehicle_camera_config(self):
        screen_width_inches = 96.6
        vehicle_width_inches = 79.3
        print(f"DEBUG: Looking for vehicle model ID: '{self._actor_model}'")

        fov_scaling_factor = screen_width_inches / vehicle_width_inches
        bounding_box = self._parent.bounding_box
        extent = bounding_box.extent


        #### DRIVER POSITION DEFINITIONS AND ADJUSTMENT
        if self._actor_model == 'vehicle.mitsubishi.fusorosa':
            driver_loc_x = bounding_box.location.x + (extent.x*0.58)
            driver_loc_y = bounding_box.location.y + (extent.y*-0.33)
            driver_loc_z = bounding_box.location.z + (extent.z*0.38)
        else:    
            driver_loc_x = bounding_box.location.x + (extent.x*0.65)
            driver_loc_y = bounding_box.location.y + (extent.y*-0.52)
            driver_loc_z = bounding_box.location.z + (extent.z*0.33)
        logging.info(f'DRIVER LOCATION: X: {driver_loc_x}, Y:{driver_loc_y}, Z:{driver_loc_z}')

        central_location = carla.Location(x=driver_loc_x, y=driver_loc_y,z=driver_loc_z)

        # --- Define the TRANSFORM (Location and Rotation) Parameters ---
        
        # Central Driver Location (Views will be customized in dictionary VEHICLE_CONFIGS below)
        fusorosa_driver_loc = central_location

        # Rotation Parameters
        fusorosa_chase_rot = carla.Rotation(pitch=-20)
        fusorosa_top_rot = carla.Rotation(pitch=-90)
        fusorosa_right_rot= carla.Rotation (pitch=0, yaw= -90)
        fusorosa_left_rot= carla.Rotation (pitch=0, yaw= 90)

        # Location Parameters
        fusorosa_left_loc = carla.Location(x=driver_loc_x,y= -15, z = driver_loc_z)
        fusorosa_right_loc = carla.Location(x=driver_loc_x,y= 15, z= driver_loc_z)
        fusorosa_top_loc = carla.Location(x=driver_loc_x, y=0, z=30)
        fusorosa_chase_loc = carla.Location(x=-20, y=0, z=10)
        

        VEHICLE_CONFIGS = {
            'vehicle.mitsubishi.fusorosa': {
                'panoramic_setup': { # Defines FOV and queue for each camera
                    'left_side_cam':  {'fov': 45*fov_scaling_factor, 'queue': 'left_side'},
                    'left_dash_cam':  {'fov': 45*fov_scaling_factor, 'queue': 'main'},
                    'right_dash_cam': {'fov': 45*fov_scaling_factor, 'queue': 'right_dash'},
                    'right_side_cam': {'fov': 45*fov_scaling_factor, 'queue': 'right_side'}
                },
                'view_sets': [ # A list of views. Each view defines the FINAL transform for each camera.
                    { # --- View 0: Driver ---
                        'left_side_cam':  carla.Transform(fusorosa_driver_loc, carla.Rotation(yaw=-53*fov_scaling_factor, pitch=-5)),
                        'left_dash_cam':  carla.Transform(fusorosa_driver_loc, carla.Rotation(yaw=-7*fov_scaling_factor, pitch=-5)),
                        'right_dash_cam': carla.Transform(fusorosa_driver_loc, carla.Rotation(yaw=39*fov_scaling_factor, pitch=-5)),
                        'right_side_cam': carla.Transform(fusorosa_driver_loc #+ carla.Location(x=driver_loc_x,y=driver_loc_y, z=driver_loc_z)
                                                          ,carla.Rotation(yaw=84*fov_scaling_factor, pitch=-5))
                    },
                    { # --- View 1: Chase ---
                        'left_side_cam':  carla.Transform(fusorosa_chase_loc, fusorosa_chase_rot),
                        'left_dash_cam':  carla.Transform(fusorosa_chase_loc, fusorosa_chase_rot),
                        'right_dash_cam': carla.Transform(fusorosa_chase_loc, fusorosa_chase_rot),
                        'right_side_cam': carla.Transform(fusorosa_chase_loc, fusorosa_chase_rot)
                    },
                    {# --- View 2: Top-Down
                        'left_side_cam': carla.Transform(fusorosa_top_loc, fusorosa_top_rot), 
                        'left_dash_cam': carla.Transform(fusorosa_top_loc, fusorosa_top_rot),
                        'right_dash_cam': carla.Transform(fusorosa_top_loc, fusorosa_top_rot),
                        'right_side_cam': carla.Transform(fusorosa_top_loc, fusorosa_top_rot),
                    },
                    {# --- View 3: Right
                        'left_side_cam': carla.Transform(fusorosa_right_loc, fusorosa_right_rot), 
                        'left_dash_cam': carla.Transform(fusorosa_right_loc, fusorosa_right_rot), 
                        'right_dash_cam': carla.Transform(fusorosa_right_loc, fusorosa_right_rot), 
                        'right_side_cam': carla.Transform(fusorosa_right_loc, fusorosa_right_rot), 
                    },
                    {# --- View 4: Left
                        'left_side_cam': carla.Transform(fusorosa_left_loc, fusorosa_left_rot), 
                        'left_dash_cam': carla.Transform(fusorosa_left_loc, fusorosa_left_rot), 
                        'right_dash_cam': carla.Transform(fusorosa_left_loc, fusorosa_left_rot), 
                        'right_side_cam': carla.Transform(fusorosa_left_loc, fusorosa_left_rot), 
                    }                                                            
                ],
                'rearview_setup': {
                    'rearview_res_w': self.single_monitor_dim[0] // 2, 
                    'rearview_res_h': self.single_monitor_dim[1] // 6, 'fov': 110,
                    'transform': carla.Transform(carla.Location(x=3.2, z=2.5), carla.Rotation(yaw=190, pitch=-17))
                }
            },
            'default': { # Simpler default config
                 'panoramic_setup': {
                    'left_side_cam':  {'fov': 45, 'queue': 'left_side'}, 'left_dash_cam':  {'fov': 45, 'queue': 'main'},
                    'right_dash_cam': {'fov': 45, 'queue': 'right_dash'}, 'right_side_cam': {'fov': 45, 'queue': 'right_side'}
                },
                'view_sets': [
                    { # Driver View
                        'left_side_cam':  carla.Transform(
                            central_location
                           # carla.Location(x=driver_loc_x, z=driver_loc_z, y=-extent.y)
                            , carla.Rotation(yaw=-55,pitch=0)),
                        'left_dash_cam':  carla.Transform(central_location#carla.Location(x=driver_loc_x, y=driver_loc_y, z=driver_loc_z)
                            , carla.Rotation(yaw=-10, pitch=0)),
                        'right_dash_cam': carla.Transform(central_location#carla.Location(x=driver_loc_x, y=driver_loc_y, z=driver_loc_z)
                                                          , carla.Rotation(yaw=35, pitch=0)),
                        'right_side_cam': carla.Transform(
                            central_location
                            #carla.Location(x=driver_loc_x, z=driver_loc_z, y=extent.y+2)
                            , carla.Rotation(yaw=80,pitch=0))
                    },
                    { # Chase View
                        'left_side_cam': carla.Transform(carla.Location(x=-10, z=7), carla.Rotation(pitch=-20)),
                        'left_dash_cam': carla.Transform(carla.Location(x=-10, z=7), carla.Rotation(pitch=-20)),
                        'right_dash_cam': carla.Transform(carla.Location(x=-10, z=7), carla.Rotation(pitch=-20)),
                        'right_side_cam': carla.Transform(carla.Location(x=-10, z=7), carla.Rotation(pitch=-20))
                    },
                    {# --- View 2: Top-Down
                        'left_side_cam': carla.Transform(fusorosa_top_loc, fusorosa_top_rot), 
                        'left_dash_cam': carla.Transform(fusorosa_top_loc, fusorosa_top_rot),
                        'right_dash_cam': carla.Transform(fusorosa_top_loc, fusorosa_top_rot),
                        'right_side_cam': carla.Transform(fusorosa_top_loc, fusorosa_top_rot),
                    },
                    {# --- View 3: Right
                        'left_side_cam': carla.Transform(fusorosa_right_loc, fusorosa_right_rot), 
                        'left_dash_cam': carla.Transform(fusorosa_right_loc, fusorosa_right_rot), 
                        'right_dash_cam': carla.Transform(fusorosa_right_loc, fusorosa_right_rot), 
                        'right_side_cam': carla.Transform(fusorosa_right_loc, fusorosa_right_rot), 
                    },
                    {# --- View 4: Left
                        'left_side_cam': carla.Transform(fusorosa_left_loc, fusorosa_left_rot), 
                        'left_dash_cam': carla.Transform(fusorosa_left_loc, fusorosa_left_rot), 
                        'right_dash_cam': carla.Transform(fusorosa_left_loc, fusorosa_left_rot), 
                        'right_side_cam': carla.Transform(fusorosa_left_loc, fusorosa_left_rot),
                    }
                ],
                'rearview_setup': {
                    'rearview_res_w': self.single_monitor_dim[0] // 3, 
                    'rearview_res_h': self.single_monitor_dim[1] // 6, 'fov': 110,
                    'transform': carla.Transform(carla.Location(x=3.2, z=3.75), carla.Rotation(yaw=190, pitch=-12))
                }
            }
        }
        
        cfg = VEHICLE_CONFIGS.get(self._actor_model, VEHICLE_CONFIGS['default'])   # [FIX] no 'vehicle_id' here

        # [ADD] augment panoramic entries with seg defaults + per-camera tag policy
        for cam_name, cam_cfg in cfg['panoramic_setup'].items():
            cam_cfg.setdefault('seg', True)                 # enable seg for this camera
            cam_cfg.setdefault('seg_tick', 0.066)           # ~15 FPS; 0.0 means full rate
            cam_cfg.setdefault('seg_tags_include', [        # names or ids; empty list = allow all
                "person", "rider", "bicycle", "motorcycle", "car", "truck", "bus"
            ])
            cam_cfg.setdefault('seg_tags_exclude', [])

        return cfg

    
    def _on_seg_image(self,cam_name:str,image):
        try:
            h,w = int(image.height), int(image.width)
            buf = np.frombuffer(image.raw_data, dtype = np.uint8).reshape(h,w,4)
            tags = buf[:,:,2].copy()  # Semantic tag_id
            inst = (buf[:,:,1].astype(np.uint16) << 8) | buf[:,:,0].astype(np.uint16)
            with self.array_lock:
                self.seg_tags[cam_name] = tags
                self.seg_inst[cam_name] = inst
        except Exception as e:
            logging.error(f"[Seg] parse fail for {cam_name}: {e}")
    
    def get_set_maps(self,cam_name:str):
        with self.array_lock:
            return self.seg_tags.get(cam_name),self.seg_inst.get(cam_name)
        
    def _add_to_queue(self, queue_key, image):
        if self.image_queues[queue_key].full():
            try:
                self.image_queues[queue_key].get_nowait()
            except queue.Empty:
                pass
        self.image_queues[queue_key].put(image)

    def _image_processor(self, queue_key):
        """
        Pull frames from self.image_queues[queue_key], convert BGRA -> RGB (HxWx3 uint8),
        and stash into self.processed_arrays[queue_key]. Exits when self.stop_threads is set.
        """
        first_logged = False
        while not self.stop_threads.is_set():
            try:
                # 1) Block for the next frame from this camera queue
                image = self.image_queues[queue_key].get(timeout=1.0)
            except queue.Empty:
                continue  # loop and check stop flag again

            try:
                h, w = int(image.height), int(image.width)
                buf = np.frombuffer(image.raw_data, dtype=np.uint8)

                # Sanity: CARLA packs BGRA => 4 bytes/pixel
                if buf.size != w * h * 4:
                    logging.debug(f"[CM] {queue_key}: unexpected raw_data size={buf.size} (w*h*4={w*h*4})")
                    continue

                # 2) Reshape and drop alpha -> BGR
                arr = buf.reshape((h, w, 4))[:, :, :3]

                # 3) Convert BGR -> RGB (copy to ensure the array is contig & not a view)
                arr = arr[:, :, ::-1].copy()

                # 4) Publish for HUD/renderer
                with self.array_lock:
                    self.processed_arrays[queue_key] = arr

                if not first_logged:
                    logging.info(f"[CM] first frame -> {queue_key} {arr.shape}")
                    first_logged = True

                # Optional: tell the queue we’re done (only if you use .join() elsewhere)
                # self.image_queues[queue_key].task_done()

            except Exception as e:
                logging.error(f"[CM] image_processor[{queue_key}] error: {e}")
                # keep looping so the thread survives transient errors

    def get_camera_actor_for_queue(self, queue_key):
        name = getattr(self, 'queue_to_name', {}).get(queue_key)
        return self.sensors.get(name) if name else None

    def get_latest_array(self, queue_key):
        with self.array_lock:
            arr = self.processed_arrays.get(queue_key)
            return None if arr is None else arr.copy()

    def _spawn_rearview_camera(self):
        rear_bp = (
            self._parent.get_world().get_blueprint_library().find("sensor.camera.rgb")
        )
        rear_bp.set_attribute("image_size_x", str(self.rearview_res_w))
        rear_bp.set_attribute("image_size_y", str(self.rearview_res_h))
        rear_bp.set_attribute("fov","110")
        
        if self._actor_model=='vehicle.mitsubishi.fusorosa':
            transform = carla.Transform(carla.Location(x=3.2, z=3.75), carla.Rotation(yaw=190,pitch=-17))
        else:
            transform = carla.Transform(
                carla.Location(x=-4.0, z=2.5), carla.Rotation(yaw=180)
            )
        try:
            self.rearview_cam = self._parent.get_world().spawn_actor(
                rear_bp, transform, attach_to=self._parent
            )
        except Exception as e:
            logging.error(f"Failed to spawn rearview camera: {e}")


    def render(self, display):
        with self.array_lock:
            main_array = self.processed_arrays.get('main')
            left_array = self.processed_arrays.get('left_side')
            right_dash_array = self.processed_arrays.get('right_dash')
            right_side_array = self.processed_arrays.get('right_side')
            rearview_array = self.processed_arrays.get('rearview')

        if left_array is not None:
            surface = pygame.surfarray.make_surface(left_array.swapaxes(0, 1))
            display.blit(surface, (0, 0))
        if main_array is not None:
            surface = pygame.surfarray.make_surface(main_array.swapaxes(0, 1))
            display.blit(surface, (self.single_monitor_dim[0], 0))
        if right_dash_array is not None:
            surface = pygame.surfarray.make_surface(right_dash_array.swapaxes(0, 1))
            display.blit(surface, (self.single_monitor_dim[0] * 2, 0))
        if right_side_array is not None:
            surface = pygame.surfarray.make_surface(right_side_array.swapaxes(0, 1))
            display.blit(surface, (self.single_monitor_dim[0] * 3, 0))
        if rearview_array is not None:
                    image_surface = pygame.surfarray.make_surface(rearview_array.swapaxes(0, 1))
                    flipped_surface = pygame.transform.flip(image_surface, True, False)
                    
                    bezel_thickness = 8
                    shadow_offset = 4
                    final_width = self.rearview_res_w + (bezel_thickness * 2) + shadow_offset
                    final_height = self.rearview_res_h + (bezel_thickness * 2) + shadow_offset
                    final_surface = pygame.Surface((final_width, final_height), pygame.SRCALPHA)
                    shadow_rect = pygame.Rect(shadow_offset, shadow_offset, self.rearview_res_w + (bezel_thickness * 2), self.rearview_res_h + (bezel_thickness * 2))
                    pygame.draw.rect(final_surface, (0, 0, 0, 70), shadow_rect, border_radius=20)
                    bezel_rect = pygame.Rect(0, 0, self.rearview_res_w + (bezel_thickness * 2), self.rearview_res_h + (bezel_thickness * 2))
                    pygame.draw.rect(final_surface, (20, 20, 20, 220), bezel_rect, border_radius=10)
                    pygame.draw.rect(final_surface, (100, 100, 100, 100), bezel_rect, 1, border_radius=10)
                    final_surface.blit(flipped_surface, (bezel_thickness, bezel_thickness))
                    inner_rect = pygame.Rect(bezel_thickness, bezel_thickness, self.rearview_res_w, self.rearview_res_h)
                    pygame.draw.rect(final_surface, (0, 0, 0, 150), inner_rect, 1)
                    
                    padding = 20
                    # FIXED: Position relative to the right dash monitor (monitor 2) instead of total display width
                    right_dash_monitor_start = self.single_monitor_dim[0] * 2  # Start of third monitor
                    right_dash_monitor_end = right_dash_monitor_start + self.single_monitor_dim[0]  # End of third monitor
                    x_pos = right_dash_monitor_end - final_width - padding  # Position from right edge of right dash monitor
                    y_pos = padding
                    display.blit(final_surface, (x_pos, y_pos))
#                    self.hud._render_blinker_indicator(display)
                    logging.info(f'rearview x,y ({x_pos},{y_pos} rearview_res_xy ({self.rearview_res_w},{self.rearview_res_h})')
#        self.hud._render_blinker_indicator(display)

    def destroy(self):
        """Properly destroys all sensors."""
        self.stop_threads.set()
        for t in self.threads:
            t.join()

        all_sensors = list(self.sensors.values())
        if self.rearview_cam:
            all_sensors.append(self.rearview_cam)

        for sensor in all_sensors:
            if not sensor:
                continue
            try: sensor.stop()
            except Exception: pass
            try: sensor.listen(None)
            except Exception: pass
            try: sensor.destroy()
            except Exception: pass

        self.sensors.clear()
        self.rearview_cam = None
    

    def set_sensor(self, index, notify=True):
        index %= len(self.sensors)
        if self.blueprints[index] is None:
            if notify:
                self.hud.error(f"Sensor '{self.sensors[index][2]}' unavailable.")
            return
        if self.sensor:
            self.sensor.destroy()

        bp = self.blueprints[index]
        if self.sensors[index][0].startswith("sensor.camera"):
            bp.set_attribute("image_size_x", str(self.single_monitor_dim[0]))
            bp.set_attribute("image_size_y", str(self.single_monitor_dim[1]))
            bp.set_attribute(
                "fov",
                str(
                    self.fov if self.sensors[index][0] == "sensor.camera.rgb" else "90"
                ),
            )

        self.sensor = self._parent.get_world().spawn_actor(
            bp, self._camera_transforms[self.transform_index], attach_to=self._parent
        )
        # Note: The listener for this sensor is now set in __init__
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index


    def toggle_camera(self):
        """Cycles to the next view set and updates each camera's transform."""
        self.view_index = (self.view_index + 1) % len(self.view_sets)
        new_view_set = self.view_sets[self.view_index]

        for name, sensor_actor in self.sensors.items():
            if sensor_actor and sensor_actor.is_alive:
                new_transform = new_view_set.get(name)
                if new_transform:
                    sensor_actor.set_transform(new_transform)
                


    def next_sensor(self):
        self.set_sensor(self.index + 1 if self.index is not None else 0)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification("Recording %s" % ("On" if self.recording else "Off"))
