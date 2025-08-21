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
from VisionPerception import Perception

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
        }
        self.collision_distance_threshold = 5.0  # Meters

    def report(self, event_type, details=None):
        """Sensors report events here. The manager decides if a new VISUAL notification is needed."""
        details = details or {}
        if event_type in self.active_events:
            return

        self.active_events[event_type] = {"time": time.time()}

        message, color = (
            details.get("message", "Event"),
            details.get("color", (255, 255, 0)),
        )
        is_critical = details.get("is_critical", False)
        sound_to_play = details.get("sound")

        if event_type == "collision":
            message, is_critical, sound_to_play = "COLLISION!", True, "collision"

        self.hud.notification(
            message, seconds=3.0, text_color=color, is_critical_center=is_critical
        )

        if sound_to_play:
            force = event_type == "collision"
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

class EndScreen(object):
    def __init__(self, display_surface, final_scores: dict, hud_fonts: dict):
        self.surface = display_surface
        self.dim = display_surface.get_size()
        self.final_scores = final_scores
        self.fonts = hud_fonts  # Use the fonts passed from the main HUD
        self._background_surface = None
        self._height = self.dim[1]

        # Define colors
        self.title_color = (220, 38, 38)
        self.score_color = (252, 211, 77)
        self.text_color = (229, 231, 235)
        self.button_color = (55, 65, 81)
        self.button_hover_color = (75, 85, 99)
        self.button_text_color = (255, 255, 255)

        # --- Corrected Positioning for Multi-Monitor ---
        main_screen_offset_x = self.dim[0] // 4
        single_screen_width = self.dim[0] // 4
        center_x = main_screen_offset_x + (single_screen_width / 2)
        
        button_w, button_h = 350, 60
        button_y_start = self._height * 0.75
        button_spacing = button_h + 20

        self.buttons = {
            "restart": pygame.Rect(center_x - button_w / 2, button_y_start, button_w, button_h),
            "exit": pygame.Rect(center_x - button_w / 2, button_y_start + button_spacing, button_w, button_h),
        }
        self.button_labels = {"restart": "Restart Simulation", "exit": "Exit to Desktop"}

    def draw(self):
        # Draw the background gradient
        if not self._background_surface or self._background_surface.get_size() != self.dim:
            self._background_surface = pygame.Surface(self.dim)
            top_color, bottom_color = (44, 62, 80), (27, 38, 49)
            for y in range(self._height):
                r = top_color[0] + (bottom_color[0] - top_color[0]) * y // self._height
                g = top_color[1] + (bottom_color[1] - top_color[1]) * y // self._height
                b = top_color[2] + (bottom_color[2] - top_color[2]) * y // self._height
                pygame.draw.line(self._background_surface, (r, g, b), (0, y), (self.dim[0], y))
        self.surface.blit(self._background_surface, (0, 0))

        # --- Positioning for Multi-Monitor ---
        main_screen_offset_x = self.dim[0] // 4
        single_screen_width = self.dim[0] // 4
        center_x = main_screen_offset_x + (single_screen_width / 2)

        # Draw the main title, using the correct font key
        height_ratio = self._height/2160
        title_font = self.fonts.get("main_score", pygame.font.Font(None, math.floor(82*height_ratio)))
        title_surf = title_font.render("SESSION ENDED", True, self.title_color)
        title_rect = title_surf.get_rect(center=(center_x, self._height * 0.25))
        self.surface.blit(title_surf, title_rect)
        
        # --- Draw Final Scores using corrected font keys ---
        # CORRECTED: Mapped to the actual keys in the self.fonts dictionary
        score_index_font = self.fonts.get('title', pygame.font.Font(None, math.floor(40*height_ratio)))
        score_label_font = self.fonts.get('sub_label', pygame.font.Font(None, math.floor(24*height_ratio)))
        score_value_font = self.fonts.get('sub_value', pygame.font.Font(None, math.floor(32*height_ratio)))
        
        score_area_width = single_screen_width * 0.6
        score_area_x_start = center_x - (score_area_width / 2)
        y_pos = self._height * 0.30

        for key, value in self.final_scores.items():
            if key.startswith('spacer'):
                y_pos += 20
                continue

            # Format label text (your custom logic is preserved)
            if key == 'index_mbi_0_1':
                label_text = 'Overall Vehicle Control & Collisions'
            elif key == 'index_lmi_0_1':
                label_text = 'Overall Lane Management Score'
            else:
                label_text = key.replace('_', ' ').replace(' raw', '').title()
            
            # Format value text (your custom logic is preserved)
            value_text = ""
            if value is not None and value != 'None':
                try: value_text = f"{float(value):.1f}"
                except (ValueError, TypeError): value_text = str(value)

            is_index = 'index' in key
            label_surf = (score_index_font if is_index else score_label_font).render(label_text, True, self.text_color)
            value_surf = (score_value_font if is_index else score_value_font).render(value_text, True, self.score_color)

            self.surface.blit(label_surf, (score_area_x_start, y_pos))
            self.surface.blit(value_surf, (score_area_x_start + score_area_width - value_surf.get_width(), y_pos))
            y_pos += value_surf.get_height() + (15 if is_index else 10)

        # Draw buttons
        mouse_pos = pygame.mouse.get_pos()
        button_font = self.fonts.get("sub_value", pygame.font.Font(None, 32))
        for key, rect in self.buttons.items():
            color = self.button_hover_color if rect.collidepoint(mouse_pos) else self.button_color
            pygame.draw.rect(self.surface, color, rect, border_radius=12)
            label_surf = button_font.render(self.button_labels[key], True, self.button_text_color)
            label_rect = label_surf.get_rect(center=rect.center)
            self.surface.blit(label_surf, label_rect)

        pygame.display.flip()

    def run(self, mapped_keys=None):
        map_keys = mapped_keys
        logging.info("EndScreen is now active, waiting for user input.")
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "exit"
                if event.type == pygame.KEYDOWN:
                    # --- Re-integrated Keyboard Shortcuts ---
                    if event.key == pygame.K_r:
                        return "restart"
                    if event.key == pygame.K_ESCAPE or \
                       ((event.key == pygame.K_q or event.key == pygame.K_c) and (pygame.key.get_mods() & pygame.KMOD_CTRL)):
                        return "exit"
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for key, rect in self.buttons.items():
                        if rect.collidepoint(event.pos):
                            return key
                if event.type == pygame.JOYBUTTONDOWN:
                    if (map_keys and event.instance_id == map_keys.get('Escape', {}).get('joy_id') and
                        event.button == map_keys.get('Escape', {}).get('button_id')):
                        return "exit"
                    
                    # Check if the pressed button is the mapped ENTER button (acts as Restart)
                    elif (map_keys and event.instance_id == map_keys.get('Enter', {}).get('joy_id') and
                          event.button == map_keys.get('Enter', {}).get('button_id')):
                        return "restart"                    
            self.draw()


class PersistentWarningManager(object):
    def __init__(self, font_object, dim):
        self.font = font_object
        self.screen_dim = dim
        self.active_warnings = {}

    def add_warning(self, key, text):
        self.active_warnings[key] = text.upper()

    def remove_warning(self, key):
        if key in self.active_warnings:
            del self.active_warnings[key]

    def get_warnings(self):
        return list(self.active_warnings.values())

    def render(self, display, panel_rect, start_y):
        if not self.active_warnings:
            return
        y_offset = start_y
        for text in self.active_warnings.values():
            symbol_font = pygame.font.Font(pygame.font.get_default_font(), 18)
            symbol_texture = symbol_font.render("⚠", True, (255, 255, 0))
            text_texture = self.font.render(text, True, (255, 255, 200))
            padding = 8
            box_height = text_texture.get_height() + padding
            content_width = text_texture.get_width() + symbol_texture.get_width() + 5
            box_width = content_width + padding * 2
            surface = pygame.Surface((box_width, box_height), pygame.SRCALPHA)
            surface.fill((80, 80, 0, 180), surface.get_rect())
            surface.blit(
                symbol_texture,
                (padding, (box_height - symbol_texture.get_height()) // 2),
            )
            surface.blit(
                text_texture,
                (
                    padding + symbol_texture.get_width() + 5,
                    (box_height - text_texture.get_height()) // 2,
                ),
            )
            box_x = panel_rect.x + (panel_rect.width - box_width) / 2
            display.blit(surface, (box_x, y_offset))
            y_offset += box_height + 5


class BlinkingAlert(object):
    def __init__(self, font, screen_dim):
        self.font, self.screen_dim = font, screen_dim
        self.surface = pygame.Surface((0, 0), pygame.SRCALPHA)
        self.current_pos, self.seconds_left = [0, 0], 0
        (
            self.start_time,
            self.duration,
            self.text,
            self.is_blinking,
            self.is_critical_center,
        ) = 0.0, 0.0, "", False, False

    def set_text(
        self,
        text,
        text_color=(255, 255, 255),
        seconds=2.0,
        is_blinking=False,
        is_critical_center=False,
    ):
        (
            self.text,
            self.seconds_left,
            self.duration,
            self.is_blinking,
            self.is_critical_center,
        ) = text, seconds, seconds or 0.001, is_blinking, is_critical_center
        self.start_time = pygame.time.get_ticks() / 1000.0
        text_surf = self.font.render(
            text.upper() if is_critical_center or is_blinking else text,
            True,
            text_color,
        )
        pad_h, pad_v = (20, 15) if is_critical_center else (15, 10)
        box_h = text_surf.get_height() + 2 * pad_v
        box_w = text_surf.get_width() + 2 * pad_h
        self.surface = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        self.surface.fill((20, 20, 20, 180))
        self.surface.blit(
            text_surf,
            ((box_w - text_surf.get_width()) / 2, (box_h - text_surf.get_height()) / 2),
        )

        # --- FIX: Calculate position relative to the main (right) screen ---
        main_screen_offset_x = self.screen_dim[0] // 4
        single_screen_width = self.screen_dim[0] // 4
        center_x_on_main_screen = (
            main_screen_offset_x + (single_screen_width - box_w) / 2
        )

        self.initial_pos = (
            [center_x_on_main_screen, int(self.screen_dim[1] * 0.4) - box_h / 2]
            if is_critical_center
            else [center_x_on_main_screen, self.screen_dim[1]]
        )
        self.current_pos = list(self.initial_pos)

    def tick(self, clock):
        self.seconds_left = max(0.0, self.seconds_left - clock.get_time() * 1e-3)
        if self.seconds_left > 0:
            elapsed = (pygame.time.get_ticks() / 1000.0) - self.start_time
            alpha = (
                int(abs(math.sin(elapsed * math.pi * 3.0)) * 255)
                if self.is_blinking
                else int(255 * (self.seconds_left / self.duration))
            )
            self.surface.set_alpha(alpha)
        else:
            self.surface.set_alpha(0)
        return self.seconds_left > 0

    def render(self, display):
        if self.surface.get_alpha() > 0:
            display.blit(self.surface, self.current_pos)


class HelpText(object):
    def __init__(self, font, width, height):
        lines = [
            "SIMULATION CONTROLS:",
            "TAB: Change Camera",
            "H: Toggle Help",
            "A: Toggle Autopilot",
            "N: Next Weather Setting",
            "SHIFT+N: Previous Weather Setting",
            "TAB: Cycle (Camera) View",
            "BACKSPACE: Respawn in New Location",
            "CTRL+BACKSPACE: Respawn and Reset Scores in New Location",
            "",
            "CAR ACCESSORY CONTROLS",
            "Z: Left Blinker",
            "X: Right Blinker",
            "SHIFT+Z: Hazard Blinkers",
            "",
            "TRANSMISSION CONTROL:",
            "Q: Toggle Gear: Reverse",
            "P: Toggle Gear: Park",
            "SPACE: Handbrake",
            "M: Change Transmission Type",
            ". (PERIOD) : GEAR UP",
            ", (COMMA) : GEAR DOWN",
            "",                        
            "ESC, CTRL+C, CTRL+Q: Quit",

        ]
        self.font, self.dim, self._render = font, (width, height), False
        max_w = max(self.font.size(l)[0] for l in lines) if lines else 0
        surf_w, surf_h = (
            min(max_w + 44, int(width * 0.8)),
            min(len(lines) * self.font.get_linesize() + 24, int(height * 0.8)),
        )
        self.pos = (width / 2.5 - surf_w / 2.5, height / 2 - surf_h / 2)
        self.surface = pygame.Surface((surf_w, surf_h), pygame.SRCALPHA)
        pygame.draw.rect(
            self.surface, (0, 0, 0, 200), self.surface.get_rect(), border_radius=15
        )
        y = 18
        for line in lines:
            if y + self.font.get_linesize() > surf_h - 12:
                break
            self.surface.blit(self.font.render(line, True, (255, 255, 255)), (22, y))
            y += self.font.get_linesize()

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


class HUD(object):
    def __init__(self, width, height, args):
        self.dim = (width, height)
        self.event_manager = EventManager(self)

        self.custom_font_path = (
            "./CarlaUE4/Content/Carla/Fonts/tt-supermolot-neue-trl.bd-it.ttf"
        )
        self._use_custom_font = os.path.exists(self.custom_font_path)
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
            logging.warning(f"Could not load blinker images: {e}.")

        self._server_clock = pygame.time.Clock()

        font_path = self.custom_font_path if self._use_custom_font else None

        scale_factor = height / 1080.0
        logging.info(f"LOGGING:HUD HEIGHT{height}")
        # Define base font sizes
        base_sizes = {
            "title": 12,
            "main_score": 40,
            "sub_label": 40,
            "sub_value": 56,
            "large_val": 28,
            "small_label": 9,
        }

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

        # Apply the scaling factor to each font size
        scaled_sizes = {k: int(v * scale_factor) for k, v in base_sizes.items()}

        font_path = self.custom_font_path if self._use_custom_font else None
        self.panel_fonts = {
            key: pygame.font.Font(font_path, size) for key, size in scaled_sizes.items()
        }

        pygame.mixer.init()
        self.sounds, self.sound_cooldowns = (
            {},
            {
                "collision": 3.0,
                "lane_drift": 2.0,
                "solid_line_crossing": 2.0,
                "speeding": 5.0,
                "error": 1.0,
            },
        )
        self._last_sound_time = {k: 0.0 for k in self.sound_cooldowns}
        sound_files = {
            "collision": "./audio/alerts/collision_alert_sound.wav",
            "lane_drift": "./audio/alerts/lane_deviation_sound.wav",
            "solid_line_crossing": "./audio/alerts/solid_line_sound.wav",
            "speeding": "./audio/alerts/speed_violation_sound.wav",
            "error": "./audio/alerts/error_encountered_sound.wav",
        }
        for k, v in sound_files.items():
            if os.path.exists(v):
                self.sounds[k] = pygame.mixer.Sound(v)

        self.help = HelpText(self.panel_fonts["small_label"], width, height)
        self.server_fps, self.frame, self.simulation_time = 0, 0, 0
        self._show_info, self._info_text, self._active_notifications,self._debug_values, self._sub_label_values = True, {}, [], {}, {}
        self._notification_base_pos_y, self._notification_spacing = (
            int(self.dim[1] * 0.85),
            8,
        )
        self._last_speed_warning_frame_warned = 0
        self.overall_dp_score_display, self.mbi_display, self.lmi_display = (
            100,
            1.0,
            1.0,
        )
        self._blinker_state = 0
        self.warning_manager = PersistentWarningManager(
            self.panel_fonts["small_label"], self.dim
        )
        # NEW: cache for Font objects to prevent FD leaks
        self._font_cache = {}  # NEW

        # NEW: prefer CARLA font if present; fallback otherwise
        self._carla_font_path = os.path.join(
            os.getcwd(), 'CarlaUE4', 'Content', 'Carla', 'Fonts', 'tt-supermolot-neue-trl.bd-it.ttf'
        )  # NEW
        if not os.path.exists(self._carla_font_path):  # NEW
            self._carla_font_path = None  # NEW (use pygame default)

        # NEW: build all panel fonts ONCE here (tune sizes to match your HUD)
        self.panel_fonts = {  # NEW
            'title': self._get_font(28, bold=True),      # NEW
            'main_score': self._get_font(64, bold=True), # NEW
            'sub_label': self._get_font(18),             # NEW
            'sub_value': self._get_font(18, bold=True),  # NEW
            'large_val': self._get_font(48, bold=True),  # NEW
            'small_label': self._get_font(16),           # NEW
            'critical_center': self._get_font(48)
        }

        # NEW: notification debounce state
        self._notify_cooldown = 1.0  # seconds  # NEW
        self._last_notify_ts = 0.0  # NEW
        self._notification_duration = 0.0

        self.reset()

        # NEW (08.21.25): Add Flags and MP4 writer
        self.vision_compare = getattr(args, "vision_compare", False)
        self.perception = None
        self._vision_writer = None
        rec_path = getattr(args, "record_vision_demo", None)
        if rec_path:
            try:
                import imageio.v2 as imageio
                self._vision_writer = imageio.get_writer(rec_path, fps=30, codec="libx264", quality=8)
                logging.info(f"[VisionDemo] Recording to {rec_path}")
            except Exception as e:
                logging.error(f"[VisionDemo] Failed to open writer: {e}")
                self._vision_writer = None

    
    def _get_font(self, size: int, bold: bool = False):  # NEW
        key = (self._carla_font_path or 'default', size, bold)  # NEW
        if key in self._font_cache:  # NEW
            return self._font_cache[key]  # NEW
        if self._carla_font_path:  # NEW
            font = pygame.font.Font(self._carla_font_path, size)  # NEW
        else:  # NEW
            font = pygame.font.Font(None, size)  # NEW
        if bold:  # NEW
            try:  # NEW
                font.set_bold(True)  # NEW
            except Exception:  # NEW
                pass  # NEW
        self._font_cache[key] = font  # NEW
        return font  # NEW
    
    def get_dim(self):
        return self.dim
    
    def _render_text(self, surf, x, y, text, font, color, spacing):
        """Draw a single line of text and return the new y-offset."""
        t_surf = font.render(str(text), True, color)
        surf.blit(t_surf, (x, y))
        return y + t_surf.get_height() + spacing


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
        surf.blit(gear_label_surf, (gear_x, y + 40))

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

    def update_mvd_scores_for_display(self, dp_score, mbi, lmi):
        self.overall_dp_score_display, self.mbi_display, self.lmi_display = (
            dp_score,
            mbi,
            lmi,
        )

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
                    t_surf = self.panel_fonts['main_score'].render(text, True, color)
                    x_pos = center_x - (t_surf.get_width() // 2)
                    y_pos = y_off - t_surf.get_height()
                    if y_pos < self.dim[1] * 0.15:
                        break
                    surf.blit(t_surf, (x_pos, y_pos))
                    y_off -= t_surf.get_height() + getattr(self, '_notification_spacing', 8)
                keep.append((expires_at, text, color, is_center))
        self._messages = keep
    """
    def notification(
        self,
        text,
        seconds=2.0,
        text_color=(255, 255, 255),
        is_blinking=False,
        is_critical_center=False,
    ):
        font_sz = 48 if is_critical_center else 36
        try:
            txt_font = (
                pygame.font.Font(self.custom_font_path, font_sz)
                if self._use_custom_font
                else pygame.font.Font(None, font_sz)
            )
        except pygame.error:
            txt_font = pygame.font.Font(None, font_sz)

        new_notif = BlinkingAlert(txt_font, self.dim)
        new_notif.set_text(text, text_color, seconds, is_blinking, is_critical_center)
        if not is_critical_center and any(
            n.text == text for n in self._active_notifications
        ):
            return
        self._active_notifications.append(new_notif)
        self._active_notifications.sort(key=lambda n: not n.is_critical_center)
    """
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
        self.notification(f"ERROR: {text.upper()}", 5.0, (255, 50, 50), True, True)
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
            

    def tick(self, world_instance, clock, idling , controller, display_fps):
        self.event_manager.tick()
        self.world = world_instance

        try:
            cm = self.world.camera_manager
            cam_actor = cm.sensors.get('left_dash_cam')  # second tile (“main”)
            if cam_actor:
                if not hasattr(self, 'perception') or self.perception is None:
                    self.perception = Perception(self.world, camera_actor=cam_actor)
                    logging.info(f"[Vision] bound to left_dash FOV={self.perception.fov_deg:.1f} "
                                f"size={self.perception.image_width}x{self.perception.image_height}")
                else:
                    self.perception.set_camera(cam_actor)
        except Exception as e:
            logging.debug(f"[Vision] bind skipped: {e}")
        self.control = controller
        self._blinker_state = controller.get_blinker_state()
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
                "MBI":f"{self.mbi_display:.2f}",
                "LMI":f"{self.lmi_display:.2f}",                
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
                "main_score":f"{self.overall_dp_score_display:.0f}",
                "sub_label": self._sub_label_values,
                "speed_mph":speed_mph,
                "steer": steer,
                "clamped_steer": clamped_steer,
                "gear":gear,
                "debug_info":self._debug_values,
            }
        else:
            self._info_text = {"title": "Player not ready"}
        for name, parameter in vars(ackermann_settings).items():
            self._info_text[name] = parameter

    def render(self, display):
        """Renders cameras and the right-side HUD panel with a debounced notification queue.
        Assumes:
        - self.panel_fonts is prebuilt in __init__ (no per-frame font opens)
        - self.info_renderers dict exists (dynamic rendering of _info_text)
        - self._draw_notifications(display, x, y) is implemented (debounced queue)
        """
        if not hasattr(self, "_logged_render_size"):
            self._logged_render_size = True
            logging.info(f"[HUD] render() surface = {display.get_width()}x{display.get_height()}")
        #--
        # 0) Vision module
        #-- 
        import pygame
# --- Vision compare mode: LEFT = raw front-left cam, RIGHT = same with overlay ---
        if self.vision_compare and self.world and self.world.camera_manager:
            cm = self.world.camera_manager
            W, H = cm.single_monitor_dim
            left_arr = cm.get_latest_array('main')
            if left_arr is not None:
                # Left (raw)
                left_surf = pygame.surfarray.make_surface(left_arr.swapaxes(0,1)).convert()
                display.blit(left_surf, (0, 0))

                # Right (overlay)
                right_surf = left_surf.copy()
                if self.perception:
                    objs = self.perception.compute(max_objects=24, include_2d=True)
                    font = self.get_font(16, bold=True) if hasattr(self, "get_font") else pygame.font.Font(None, 16)
                    for o in objs:
                        bb = o.get("bbox_xyxy")
                        if not bb: 
                            continue
                        x1,y1,x2,y2 = map(int, bb)
                        pygame.draw.rect(right_surf, (0,255,0), pygame.Rect(x1,y1,x2-x1,y2-y1), 2)
                        label = f"{o['cls']}  {o['distance_m']:.1f}m  {o['rel_speed_mps']:+.1f}m/s"
                        txt = font.render(label, True, (255,255,255))
                        right_surf.blit(txt, (x1, max(0, y1-18)))

                display.blit(right_surf, (W, 0))  # right panel sits in the next column

                # Optional: write a side-by-side MP4 frame
                if self._vision_writer is not None:
                    try:
                        right_np = pygame.surfarray.array3d(right_surf).swapaxes(0,1)
                        frame = np.concatenate([left_arr, right_np], axis=1)
                        self._vision_writer.append_data(frame)
                    except Exception as e:
                        logging.error(f"[VisionDemo] frame write failed: {e}")

                # Skip normal panoramic draw when comparing
    


        # -------------------------------------------------------------
        # 1) Panoramic camera views + 3D bounding box overlays (unchanged)
        # -------------------------------------------------------------
        """
        if hasattr(self, 'world') and self.world and self.world.player and self.world.camera_manager:
            player = self.world.player
            camera_manager = self.world.camera_manager

            panoramic_cameras = [
                (camera_manager.sensors.get('left_side_cam'), 0),
                (camera_manager.sensors.get('left_dash_cam'), camera_manager.single_monitor_dim[0]),
                (camera_manager.sensors.get('right_dash_cam'), camera_manager.single_monitor_dim[0] * 2),
                (camera_manager.sensors.get('right_side_cam'), camera_manager.single_monitor_dim[0] * 3),
            ]

            for cam, offset_x in panoramic_cameras:
                if cam:
                    cam_surface = pygame.Surface(camera_manager.single_monitor_dim, pygame.SRCALPHA)
                    # Draw the player's bounding box into this camera's perspective
                    self.draw_3d_bounding_box(
                        display=cam_surface,
                        camera=cam,
                        bounding_box=player.bounding_box,
                        world_transform=player.get_transform()
                    )
                    display.blit(cam_surface, (offset_x, 0))
        """
        # 1) Panoramic camera views + overlays
        # 1) Panoramic camera views + overlay on LEFT-DASH only
        if hasattr(self, 'world') and self.world and self.world.player and self.world.camera_manager:
            cm = self.world.camera_manager
            W = cm.single_monitor_dim[0]

            panels = [
                ('left_side_cam',   0),
                ('left_dash_cam',   W),        # <-- overlay will go here
                ('right_dash_cam',  2*W),
                ('right_side_cam',  3*W),
            ]

            # Compute detections ONCE (uses left-dash camera actor from HUD.tick)
            objs = []
            font = None
            if getattr(self, 'perception', None):
                objs = self.perception.compute(max_objects=24, include_2d=True)
                font = pygame.font.Font(None, 18)

            for name, xoff in panels:
                cam_actor = cm.sensors.get(name)
                if not cam_actor:
                    continue

                # get latest frame for this camera's queue
                try:
                    queue_key = cm.config['panoramic_setup'][name]['queue']
                except Exception:
                    queue_key = 'main'
                with cm.array_lock:
                    arr = cm.processed_arrays.get(queue_key)

                # build the pygame surface for this tile
                if arr is None:
                    surf = pygame.Surface(cm.single_monitor_dim).convert()
                    surf.fill((10, 10, 10))
                else:
                    surf = pygame.surfarray.make_surface(arr.swapaxes(0, 1)).convert()

                # --- ONLY for left_dash_cam: draw overlay ON THE SURF before blitting ---
                if name == 'left_dash_cam' and objs:
                    for o in objs:
                        bb = o.get("bbox_xyxy")
                        if not bb: 
                            continue
                        x1, y1, x2, y2 = map(int, bb)
                        pygame.draw.rect(surf, (0, 255, 0), pygame.Rect(x1, y1, x2 - x1, y2 - y1), 2)
                        label = o.get("label") or f"{o['cls']} {o['distance_m']:.1f}m {o['rel_speed_mps']:+.1f}m/s"
                        surf.blit(font.render(label, True, (255, 255, 255)), (x1, max(0, y1 - 18)))

                # now blit the (possibly annotated) surf; nothing will overwrite it afterward
                display.blit(surf, (xoff, 0))
                # --- OPTIONAL: Vision overlay (vehicles/pedestrians) ---

                # --- Vision overlay on LEFT-DASH tile (column 2) ---
                if getattr(self, 'perception', None) and self.world and self.world.camera_manager:
                    cm = self.world.camera_manager
                    W = cm.single_monitor_dim[0]   # x offset of left-dash tile
                    objs = self.perception.compute(max_objects=24, include_2d=True)

                    import pygame
                    font = pygame.font.Font(None, 18)
                    # yellow heartbeat so you know this branch ran
                    pygame.draw.rect(display, (255,255,0), pygame.Rect(W+10, 10, 120, 24), 2)

                    for o in objs:
                        bb = o.get("bbox_xyxy")
                        if not bb:
                            continue
                        x1, y1, x2, y2 = map(int, bb)
                        rect = pygame.Rect(x1 + W, y1, (x2 - x1), (y2 - y1))  # shift by W
                        pygame.draw.rect(display, (0, 255, 0), rect, 2)
                        label = o.get("label") or f"{o['cls']} {o['distance_m']:.1f}m {o['rel_speed_mps']:+.1f}m/s"
                        display.blit(font.render(label, True, (255,255,255)), (rect.x, max(0, rect.y - 18)))


                # (Optional) draw the player's own 3D bbox on top of the video
                # self.draw_3d_bounding_box(surf, cam_actor, self.world.player.bounding_box, self.world.player.get_transform())

                display.blit(surf, (xoff, 0))
        # -------------------------------------------------------------
        # 2) Right-side HUD panel (glassmorphic) + dynamic info rendering
        # -------------------------------------------------------------
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
            padding = 20

            info_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
            pygame.draw.rect(info_surf, panel_bg_color, info_surf.get_rect(), border_radius=15)
            pygame.draw.rect(info_surf, panel_border_color, info_surf.get_rect(), width=2, border_radius=15)
            display.blit(info_surf, (panel_x, panel_y))

            v_offset = panel_y + padding
            h_offset = panel_x + padding

            # -------- Dynamic info block rendering --------
            for key, value in self._info_text.items():
                renderer = self.info_renderers.get(key)
                if renderer:
                    # Pass panel geometry for right-aligned values (sub_label, etc.)
                    v_offset = renderer(display, h_offset, v_offset, value, panel_x, panel_w, padding)
                else:
                    # Fallback: simple key:value line
                    v_offset = self._render_text(
                        display, h_offset, v_offset, f"{key}: {value}", self.panel_fonts["small_label"], (200, 200, 200), 10
                    )

            # -------- Debounced, panel-local notifications (NEW) --------
            # Draw small queued messages below the panel content using cached fonts.
            v_offset = self._draw_notifications(display, h_offset, v_offset)  # NEW
            self._draw_center_notifications(display)

        # -------------------------------------------------------------
        # 3) (Optional) Legacy fullscreen notifications
        # -------------------------------------------------------------
        # If you want to keep the legacy _active_notifications (center overlays),
        # leave the original loop here. If you prefer ONLY the debounced queue
        # inside the panel, remove/comment the legacy block.
        #
        # Example of preserving legacy behavior:
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

    def _surf_from_queue(self, cm, queue_key):
        with cm.array_lock:
            arr = cm.processed_arrays.get(queue_key)
        if arr is None:
            s = pygame.Surface(cm.single_monitor_dim).convert()
            s.fill((10,10,10))
            return s
        return pygame.surfarray.make_surface(arr.swapaxes(0,1)).convert()

    def _render_blinker_indicator(self, display):
        if self._blinker_state == 0 or not self._blinker_left_img:
            return
        if (pygame.time.get_ticks() // 750) % 2 == 0:
            return

        main_screen_offset_x = self.dim[0]*0.5 // 2
        y_pos = self.dim[1] - 80
        left_x = main_screen_offset_x + (self.dim[0]*0.5 // 2) * 0.35
        right_x = main_screen_offset_x + (self.dim[0]*0.5 // 2) * 0.65
        if self._blinker_state == 1:

            display.blit(
                self._blinker_left_img,
                self._blinker_left_img.get_rect(center=(left_x, y_pos)),
            )
        elif self._blinker_state == 2:
            display.blit(
                self._blinker_right_img,
                self._blinker_right_img.get_rect(center=(right_x, y_pos)),
            )
        elif self._blinker_state == 3:
            spacing = (self.dim[0] // 2) * 0.05
            center_x = main_screen_offset_x + (self.dim[0]*0.5 // 4)
            display.blit(
                self._blinker_left_img,
                self._blinker_left_img.get_rect(center=(center_x - spacing, y_pos)),
            )
            display.blit(
                self._blinker_right_img,
                self._blinker_right_img.get_rect(center=(center_x + spacing, y_pos)),
            )



class CameraManager(object):
    def __init__(self, parent_actor, hud, fov):
        # --- Standard Initialization ---
        self._parent = parent_actor
        self._actor_model = self._parent.attributes.get('ros_name')
        self.hud = hud
        self.dim = hud.get_dim()
        self.single_monitor_dim = (self.dim[0] // 4, self.dim[1])
        
        # --- Threading and Queue Setup ---
        self.array_lock = threading.Lock()
        self.stop_threads = threading.Event()
        self.threads = []
        self.image_queues = { 'main': queue.Queue(maxsize=1), 'left_side': queue.Queue(maxsize=1), 'right_dash': queue.Queue(maxsize=1), 'right_side': queue.Queue(maxsize=1), 'rearview': queue.Queue(maxsize=1) }
        self.processed_arrays = {k: None for k in self.image_queues}
        
        # --- Get the vehicle-specific configuration ---
        self.config = self._get_vehicle_camera_config()
        self.view_sets = self.config['view_sets']
        self.view_index = 0
        self.rearview_res_w= self.config['rearview_setup'].get('rearview_res_w')
        self.rearview_res_h= self.config['rearview_setup'].get('rearview_res_h')

        self.vehicle_dict = {}

        # --- Create All Panoramic Cameras from the Initial View Set ---
        bp_library = parent_actor.get_world().get_blueprint_library()
        self.sensors = {}
        initial_view_set = self.view_sets[self.view_index] # Get the 'Driver' view set
        
        for name, settings in self.config['panoramic_setup'].items():
            cam_bp = bp_library.find("sensor.camera.rgb")
            cam_bp.set_attribute("image_size_x", str(self.single_monitor_dim[0]))
            cam_bp.set_attribute("image_size_y", str(self.single_monitor_dim[1]))
            cam_bp.set_attribute("fov", str(settings['fov']))

            
            ## -- Modifying post-processing for individual cameras to improve performance. -- ##

            if 'side' in name:
                cam_bp.set_attribute("enable_postprocess_effects",'false')
            
            #
            #cam_bp.set_attribute("enable_postprocess_effects",'false')
            
            ## ----   END MODIFICATIONS ---------------#                                   -- ##

            # Spawn the camera with its specific transform for the initial view
            camera_actor = self._parent.get_world().spawn_actor(
                cam_bp,
                initial_view_set[name], # Get the transform from the view set
                attach_to=self._parent # Attach DIRECTLY to the vehicle
            )
            
            queue_key = settings['queue']
            camera_actor.listen(lambda image, key=queue_key: self._add_to_queue(key, image))
            self.sensors[name] = camera_actor
            self.threads.append(threading.Thread(target=self._image_processor, args=(queue_key,)))
        
        # Map camera names to their target queue ("main", "left_side", etc.)
        self.name_to_queue = {name: s['queue'] for name, s in self.config['panoramic_setup'].items()}
        self.queue_to_name = {}
        for n, q in self.name_to_queue.items():
            self.queue_to_name.setdefault(q, n)    
        
        # --- Rearview Camera Setup ---
        self.rearview_cam = None
        rear_config = self.config.get('rearview_setup')
        if rear_config:
            rear_bp = bp_library.find("sensor.camera.rgb")
            rear_bp.set_attribute("image_size_x", str(rear_config['rearview_res_w']))
            rear_bp.set_attribute("image_size_y", str(rear_config['rearview_res_h']))
            rear_bp.set_attribute("fov", str(rear_config['fov']))
            rear_bp.set_attribute("enable_postprocess_effects",'false')
            self.rearview_cam = self._parent.get_world().spawn_actor(rear_bp, rear_config['transform'], attach_to=self._parent)
            self.rearview_cam.listen(lambda image: self._add_to_queue('rearview', image))
            self.threads.append(threading.Thread(target=self._image_processor, args=('rearview',)))

        # --- Start all threads ---
        
        # --- Rearview Camera Setup ---
        self.rearview_cam = None
        rear_config = self.config.get('rearview_setup')
        if rear_config:
            rear_bp = bp_library.find("sensor.camera.rgb")
            rear_bp.set_attribute("image_size_x", str(rear_config['rearview_res_w']))
            # CORRECTED LINE: Changed the second image_size_x to image_size_y
            rear_bp.set_attribute("image_size_y", str(rear_config['rearview_res_h']))
            rear_bp.set_attribute("fov", str(rear_config['fov']))
            self.rearview_cam = self._parent.get_world().spawn_actor(rear_bp, rear_config['transform'], attach_to=self._parent)
            self.rearview_cam.listen(lambda image: self._add_to_queue('rearview', image))
            self.threads.append(threading.Thread(target=self._image_processor, args=('rearview',)))
        for t in self.threads:
            t.daemon = True
            t.start()
    
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
            driver_loc_x = bounding_box.location.x + (extent.x*.63)
            driver_loc_y = bounding_box.location.y + (extent.y*-0.52)
            driver_loc_z = bounding_box.location.z + (extent.z*0.33)
        
        logging.info(f'DRIVER LOCATION: X: {driver_loc_x}, Y:{driver_loc_y}, Z:{driver_loc_z}')
        passenger_loc_x = bounding_box.location.x + (extent.x*0.47)
        passenger_loc_y = bounding_box.location.y + (extent.y*-0.45)
        passenger_loc_z = bounding_box.location.z + (extent.z*0.38)

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
                    'transform': carla.Transform(carla.Location(x=3.2, z=3.75), carla.Rotation(yaw=190, pitch=-17))
                }
            }
        }
        return VEHICLE_CONFIGS.get(self._actor_model, VEHICLE_CONFIGS['default'])



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
            x_pos = self.dim[0] - (self.single_monitor_dim[0]*1.75) - padding
            y_pos = padding
            display.blit(final_surface, (x_pos, y_pos))
            self.hud._render_blinker_indicator(display)

    def destroy(self):
        """Properly destroys all sensors."""
        self.stop_threads.set()
        for t in self.threads:
            t.join()
        
        # Combine all sensors into one list for cleanup
        all_sensors = list(self.sensors.values())
        if self.rearview_cam:
            all_sensors.append(self.rearview_cam)

        for sensor in all_sensors:
            if sensor and sensor.is_alive:
                sensor.destroy()
    

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
