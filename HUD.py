import carla
import pygame
import os
import math
import time
import logging
import weakref
import numpy as np
import Sensors

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

        # Define colors
        self.title_color = (220, 38, 38)
        self.score_color = (252, 211, 77)
        self.text_color = (229, 231, 235)
        self.button_color = (55, 65, 81)
        self.button_hover_color = (75, 85, 99)
        self.button_text_color = (255, 255, 255)

        # --- Corrected Positioning for Multi-Monitor ---
        main_screen_offset_x = self.dim[0] // 2
        single_screen_width = self.dim[0] // 2
        center_x = main_screen_offset_x + (single_screen_width / 2)
        
        button_w, button_h = 350, 60
        button_y_start = self.dim[1] * 0.75
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
            height = self.dim[1]
            for y in range(height):
                r = top_color[0] + (bottom_color[0] - top_color[0]) * y // height
                g = top_color[1] + (bottom_color[1] - top_color[1]) * y // height
                b = top_color[2] + (bottom_color[2] - top_color[2]) * y // height
                pygame.draw.line(self._background_surface, (r, g, b), (0, y), (self.dim[0], y))
        self.surface.blit(self._background_surface, (0, 0))

        # --- Positioning for Multi-Monitor ---
        main_screen_offset_x = self.dim[0] // 2
        single_screen_width = self.dim[0] // 2
        center_x = main_screen_offset_x + (single_screen_width / 2)

        # Draw the main title, using the correct font key
        title_font = self.fonts.get("main_score", pygame.font.Font(None, 82))
        title_surf = title_font.render("SESSION ENDED", True, self.title_color)
        title_rect = title_surf.get_rect(center=(center_x, self.dim[1] * 0.15))
        self.surface.blit(title_surf, title_rect)
        
        # --- Draw Final Scores using corrected font keys ---
        # CORRECTED: Mapped to the actual keys in the self.fonts dictionary
        score_index_font = self.fonts.get('title', pygame.font.Font(None, 40))
        score_label_font = self.fonts.get('sub_label', pygame.font.Font(None, 24))
        score_value_font = self.fonts.get('sub_value', pygame.font.Font(None, 32))
        
        score_area_width = single_screen_width * 0.6
        score_area_x_start = center_x - (score_area_width / 2)
        y_pos = self.dim[1] * 0.30

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

    def run(self):
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
        main_screen_offset_x = self.screen_dim[0] // 2
        single_screen_width = self.screen_dim[0] // 2
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
            "MVD Controls:",
            "W/S: Throttle/Brake",
            "A/D: Steer",
            "Q: Toggle Reverse",
            "SPACE: Handbrake",
            "P: Toggle Park",
            "M: Toggle Manual",
            "ESC: Quit",
            "TAB: Change Camera",
            "H: Toggle Help",
        ]
        self.font, self.dim, self._render = font, (width, height), False
        max_w = max(self.font.size(l)[0] for l in lines) if lines else 0
        surf_w, surf_h = (
            min(max_w + 44, int(width * 0.8)),
            min(len(lines) * self.font.get_linesize() + 24, int(height * 0.8)),
        )
        self.pos = (width / 2 - surf_w / 2, height / 2 - surf_h / 2)
        self.surface = pygame.Surface((surf_w, surf_h), pygame.SRCALPHA)
        pygame.draw.rect(
            self.surface, (0, 0, 0, 200), self.surface.get_rect(), border_radius=15
        )
        y = 12
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

        self.panel_fonts = (
            {
                "title": pygame.font.Font(font_path, 24),
                "main_score": pygame.font.Font(font_path, 80),
                "sub_label": pygame.font.Font(font_path, 20),
                "sub_value": pygame.font.Font(font_path, 28),
                "large_val": pygame.font.Font(font_path, 56),
                "small_label": pygame.font.Font(font_path, 18),
            }
            if self._use_custom_font
            else {
                "title": pygame.font.Font(None, 24),
                "main_score": pygame.font.Font(None, 80),
                "sub_label": pygame.font.Font(None, 20),
                "sub_value": pygame.font.Font(None, 28),
                "large_val": pygame.font.Font(None, 56),
                "small_label": pygame.font.Font(None, 18),
            }
        )

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
        self._show_info, self._info_text, self._active_notifications = True, [], []
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
        self.reset()

    def get_dim(self):
        return self.dim

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

    def error(self, text):
        self.notification(f"ERROR: {text.upper()}", 5.0, (255, 50, 50), True, True)
        self.play_sound_for_event("error", force_play=True)

    def tick(self, world_instance, clock, idling, controller):
        self.event_manager.tick()
        self._blinker_state = controller.get_blinker_state()
        self._active_notifications = [
            n for n in self._active_notifications if n.tick(clock)
        ]

        self._info_text = []
        if self._show_info and world_instance.player and world_instance.player.is_alive:
            v = world_instance.player.get_velocity()
            c = world_instance.player.get_control()
            speed_mph = int(2.237 * v.length())

            if controller.is_parked():
                gear = "P"
            else:
                gear = (
                    {-1: "R", 0: "N"}.get(c.gear, str(c.gear))
                    if isinstance(c, carla.VehicleControl)
                    else "N/A"
                )

            self._info_text = [
                ("title", "Overall Score"),
                ("main_score", f"{self.overall_dp_score_display:.0f}"),
                ("sub_label", "MBI", f"{self.mbi_display:.2f}"),
                ("sub_label", "LMI", f"{self.lmi_display:.2f}"),
                ("speed_gear", speed_mph, gear),
            ]
        else:
            self._info_text = [("title", "Player not ready")]

    def render(self, display):
        """Renders the main glassmorphic HUD panel on the RIGHT screen."""
        if self._show_info and self._info_text:
            main_screen_offset_x = self.dim[0] // 2
            single_screen_width = self.dim[0] // 2

            panel_h = self.dim[1] * 0.4
            panel_w = single_screen_width * 0.25
            panel_x = main_screen_offset_x + (single_screen_width - panel_w) / 2
            panel_y = self.dim[1] * 0.58
            panel_bg_color = (20, 20, 20, 150)
            panel_border_color = (0, 150, 255, 200)
            padding = 20

            info_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
            pygame.draw.rect(
                info_surf, panel_bg_color, info_surf.get_rect(), border_radius=15
            )
            pygame.draw.rect(
                info_surf,
                panel_border_color,
                info_surf.get_rect(),
                width=2,
                border_radius=15,
            )
            display.blit(info_surf, (panel_x, panel_y))

            v_offset = panel_y + padding
            h_offset = panel_x + padding

            for item in self._info_text:
                item_type = item[0]
                if item_type == "title":
                    text_surf = self.panel_fonts["title"].render(
                        item[1], True, (200, 200, 200)
                    )
                    display.blit(text_surf, (h_offset, v_offset))
                    v_offset += text_surf.get_height() - 5
                elif item_type == "main_score":
                    text_surf = self.panel_fonts["main_score"].render(
                        item[1], True, (255, 255, 255)
                    )
                    display.blit(text_surf, (h_offset, v_offset))
                    v_offset += text_surf.get_height() + 15
                elif item_type == "sub_label":
                    label_surf = self.panel_fonts["sub_label"].render(
                        item[1], True, (200, 200, 200)
                    )
                    value_surf = self.panel_fonts["sub_value"].render(
                        item[2], True, (255, 255, 255)
                    )
                    display.blit(label_surf, (h_offset, v_offset))
                    value_x_pos = panel_x + panel_w - value_surf.get_width() - padding
                    display.blit(value_surf, (value_x_pos, v_offset))
                    v_offset += value_surf.get_height() + 10
                elif item_type == "speed_gear":
                    separator_y = v_offset + 5
                    pygame.draw.line(
                        display,
                        (100, 100, 100),
                        (panel_x + 15, separator_y),
                        (panel_x + panel_w - 15, separator_y),
                        1,
                    )
                    v_offset += 20
                    speed_val, gear_val = str(item[1]), str(item[2])
                    speed_surf = self.panel_fonts["large_val"].render(
                        speed_val, True, (255, 255, 255)
                    )
                    mph_surf = self.panel_fonts["small_label"].render(
                        "MPH", True, (200, 200, 200)
                    )
                    display.blit(speed_surf, (h_offset, v_offset))
                    display.blit(
                        mph_surf, (h_offset + speed_surf.get_width() + 5, v_offset + 30)
                    )
                    gear_surf = self.panel_fonts["large_val"].render(
                        gear_val, True, (255, 255, 255)
                    )
                    gear_label_surf = self.panel_fonts["small_label"].render(
                        "GEAR", True, (200, 200, 200)
                    )
                    gear_x_pos = panel_x + panel_w - gear_surf.get_width() - padding - 5
                    display.blit(gear_surf, (gear_x_pos, v_offset))
                    display.blit(gear_label_surf, (gear_x_pos, v_offset + 40))
                    v_offset += gear_surf.get_height() + 10

            if self.warning_manager:
                panel_rect = pygame.Rect(panel_x, panel_y, panel_w, panel_h)
                self.warning_manager.render(display, panel_rect, v_offset)

        y_off = self._notification_base_pos_y
        for notif in reversed(self._active_notifications):
            if not notif.seconds_left > 0 and notif.surface.get_alpha() == 0:
                continue
            if notif.is_critical_center:
                notif.render(display)
            else:
                main_screen_offset_x = self.dim[0] // 2
                single_screen_width = self.dim[0] // 2
                x_pos = (
                    main_screen_offset_x
                    + (single_screen_width - notif.surface.get_width()) // 2
                )
                y_pos = y_off - notif.surface.get_height()
                if y_pos < self.dim[1] * 0.15:
                    break
                display.blit(notif.surface, (x_pos, y_pos))
                y_off -= notif.surface.get_height() + self._notification_spacing

        self._render_blinker_indicator(display)

        if self.help:
            self.help.render(display)

    def _render_blinker_indicator(self, display):
        if self._blinker_state == 0 or not self._blinker_left_img:
            return
        if (pygame.time.get_ticks() // 500) % 2 == 0:
            return

        main_screen_offset_x = self.dim[0] // 2
        y_pos = self.dim[1] - 80
        left_x = main_screen_offset_x + (self.dim[0] // 2) * 0.35
        right_x = main_screen_offset_x + (self.dim[0] // 2) * 0.65

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
            center_x = main_screen_offset_x + (self.dim[0] // 4)
            display.blit(
                self._blinker_left_img,
                self._blinker_left_img.get_rect(center=(center_x - spacing, y_pos)),
            )
            display.blit(
                self._blinker_right_img,
                self._blinker_right_img.get_rect(center=(center_x + spacing, y_pos)),
            )


class CameraManager(object):
    def __init__(self, parent_actor, hud, fov=85):
        self.sensor = None
        self._parent = parent_actor
        self._actor_model = self._parent.attributes.get('ros_name')
        self.hud = hud
        self.fov = fov
        self.recording = False

        self.dim = hud.get_dim()
        self.single_monitor_dim = (self.dim[0] // 2, self.dim[1])

        self.surface = None  # Main sensor surface
        self.left_surface = None  # Left panoramic camera surface

        # Restore original camera logic for driver POV and chase cam
        driver_view_x = (
            1.2 if fov > 120 else 1.075 if fov > 110 else 0.95 if fov > 100 else 1.0
        )
        
        if self._actor_model=='vehicle.mercedes.sprinter':
            logging.info('Vehicle likely at default (debug statement)')
            self._camera_transforms = [
                carla.Transform(carla.Location(x=driver_view_x, y=-0.5, z=1.82)),
                carla.Transform(carla.Location(x=-10, z=7), carla.Rotation(pitch=-20)),
            ]
        elif self._actor_model=='vehicle.mitsubishi.fusorosa':
            logging.info(f'vehicle type set to {self._actor_model}')
            self._camera_transforms = [
                carla.Transform(carla.Location(x=3.4, y=-0.9, z=2.75)),
                carla.Transform(carla.Location(x=-10, z=7), carla.Rotation(pitch=-20)),
                carla.Transform(carla.Location(x=15, z=7), carla.Rotation(pitch=-20, yaw=180)),
            ]
        else:
            logging.info('Actor - no filter - return to default')
            self._camera_transforms = [
                carla.Transform(carla.Location(x=driver_view_x, y=-0.5, z=1.82)),
                carla.Transform(carla.Location(x=-10, z=7), carla.Rotation(pitch=-20)),
            ]
        self.transform_index = 0

        # Restore sensor list for cycling
        self.sensors = [
            ["sensor.camera.rgb", carla.ColorConverter.Raw, "Camera RGB"],
            ["sensor.camera.depth", carla.ColorConverter.Raw, "Camera Depth (Raw)"],
            [
                "sensor.camera.depth",
                carla.ColorConverter.Depth,
                "Camera Depth (Gray Scale)",
            ],
            [
                "sensor.camera.depth",
                carla.ColorConverter.LogarithmicDepth,
                "Camera Depth (Logarithmic Gray Scale)",
            ],
            [
                "sensor.camera.semantic_segmentation",
                carla.ColorConverter.Raw,
                "Camera Semantic Segmentation (Raw)",
            ],
            [
                "sensor.camera.semantic_segmentation",
                carla.ColorConverter.CityScapesPalette,
                "Camera Semantic Segmentation (CityScapes Palette)",
            ],
            ["sensor.lidar.ray_cast", None, "Lidar (Ray-Cast)"],
            ["sensor.other.radar", None, "Radar (Object Detection)"],
        ]
        bp_library = parent_actor.get_world().get_blueprint_library()
        self.blueprints = [bp_library.find(item[0]) for item in self.sensors]
        self.index = None
        self.set_sensor(0, notify=False)  # Set initial sensor (main RGB camera)

        # Left camera (90 degrees left view)
        self.left_camera_sensor = None
        left_camera_bp = bp_library.find("sensor.camera.rgb")
        left_camera_bp.set_attribute("image_size_x", str(self.single_monitor_dim[0]))
        left_camera_bp.set_attribute("image_size_y", str(self.single_monitor_dim[1]))
        left_camera_bp.set_attribute("fov", str(self.fov))
        left_transform = carla.Transform(
            self._camera_transforms[0].location, carla.Rotation(yaw=-90)
        )
        self.left_camera_sensor = self._parent.get_world().spawn_actor(
            left_camera_bp, left_transform, attach_to=self._parent
        )
        self.left_camera_sensor.listen(lambda image: self._parse_left_image(image))

# --- REARVIEW CAMERA SETUP ---
        # This section correctly sets different sizes based on the vehicle model.
        self.rearview_cam = None
        self.rearview_surface = None
        
        if self._actor_model=='vehicle.mitsubishi.fusorosa':
            self.rearview_res_w = (
                self.single_monitor_dim[0] // 2
            )
        else:
            self.rearview_res_w = (
                self.single_monitor_dim[0] // 3
            )
        self.rearview_res_h = (
            self.single_monitor_dim[1] // 6
        )

        # The oval mask is created for the default case (non-van vehicles).
        # It will be ignored when the van is active.
        self.oval_mask = pygame.Surface(
            (self.rearview_res_w, self.rearview_res_h), pygame.SRCALPHA
        )
        pygame.draw.ellipse(
            self.oval_mask,
            (255, 255, 255, 255),
            (0, 0, self.rearview_res_w, self.rearview_res_h),
        )
        
        # The sheen effect is used for both mirror types.
        self.sheen_surface = pygame.Surface(
            (self.rearview_res_w, self.rearview_res_h), pygame.SRCALPHA
        )
        w, h = self.rearview_res_w, self.rearview_res_h
        sheen_points = [(w * 0.1, 0), (w * 0.4, 0), (w * 0.2, h), (-w * 0.1, h)]
        pygame.draw.polygon(self.sheen_surface, (255, 255, 255, 20), sheen_points)
        
        self._spawn_rearview_camera()

    
    def _spawn_rearview_camera(self):
        rear_bp = (
            self._parent.get_world().get_blueprint_library().find("sensor.camera.rgb")
        )
        rear_bp.set_attribute("image_size_x", str(self.rearview_res_w))
        rear_bp.set_attribute("image_size_y", str(self.rearview_res_h))

        rear_bp.set_attribute("fov","110")
        
        if self._actor_model=='vehicle.mitsubishi.fusorosa':
            #transform = carla.Transform(carla.Location(x=-3.0, z=3.5), carla.Rotation(yaw=180), carla.#Rotation(pitch=-20))
            transform = carla.Transform(carla.Location(x=3.2, z=3.75), carla.Rotation(yaw=190,pitch=-17))
        else:
            transform = carla.Transform(
                carla.Location(x=-4.0, z=2.5), carla.Rotation(yaw=180)
            )
        try:
            self.rearview_cam = self._parent.get_world().spawn_actor(
                rear_bp, transform, attach_to=self._parent
            )
            self.rearview_cam.listen(lambda image: self._parse_rearview_image(image))
        except Exception as e:
            logging.error(f"Failed to spawn rearview camera: {e}")

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.sensor.set_transform(self._camera_transforms[self.transform_index])
        new_left_transform = carla.Transform(
            self._camera_transforms[self.transform_index].location,
            carla.Rotation(yaw=-90),
        )
        self.left_camera_sensor.set_transform(new_left_transform)

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
                    self.fov if self.sensors[index][0] == "sensor.camera.rgb" else "85"
                ),
            )

        self.sensor = self._parent.get_world().spawn_actor(
            bp, self._camera_transforms[self.transform_index], attach_to=self._parent
        )

        if self.sensors[index][0].startswith("sensor.other.radar"):
            self.sensor.listen(lambda data: self._parse_radar_data(data))
        else:
            self.sensor.listen(lambda image: self._parse_image(image))

        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1 if self.index is not None else 0)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification("Recording %s" % ("On" if self.recording else "Off"))

    def render(self, display):
        if self.left_surface:
            display.blit(self.left_surface, (0, 0))
        if self.surface:
            display.blit(self.surface, (self.single_monitor_dim[0], 0))

        # --- REARVIEW CAMERA FIX: Positioning ---
        # Place the mirror in the top-right corner of the main (right) screen.
        if self.rearview_surface:
            padding = 20
            # The x position is the total width, minus the mirror width, minus padding.
            x_pos = self.dim[0] - self.rearview_res_w - padding
            y_pos = padding
            display.blit(self.rearview_surface, (x_pos, y_pos))

    def _parse_image(self, image):
        if self.index is None or self.blueprints[self.index] is None:
            return
        sensor_type, converter = (
            self.sensors[self.index][0],
            self.sensors[self.index][1],
        )
        target_dim = self.single_monitor_dim

        if sensor_type.startswith("sensor.lidar"):
            points = np.frombuffer(image.raw_data, dtype=np.dtype("f4")).reshape(-1, 4)
            lidar_data = (points[:, :2] * min(target_dim) / 100.0) + (
                target_dim[0] / 2.0,
                target_dim[1] / 2.0,
            )
            valid_points = lidar_data[
                (lidar_data[:, 0] < target_dim[0]) & (lidar_data[:, 1] < target_dim[1])
            ].astype(np.int32)
            lidar_img = np.zeros((target_dim[1], target_dim[0], 3), dtype=np.uint8)
            if valid_points.shape[0] > 0:
                lidar_img[valid_points[:, 1], valid_points[:, 0]] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img.swapaxes(0, 1))
        elif sensor_type.startswith("sensor.camera"):
            if converter:
                image.convert(converter)
            array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
                image.height, image.width, 4
            )[:, :, :3][:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk("_out/%08d" % image.frame)

    def _parse_left_image(self, image):
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
            (image.height, image.width, 4)
        )
        array = array[:, :, :3][:, :, ::-1]
        self.left_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    """
    def _parse_rearview_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
            image.height, image.width, 4
        )[:, :, :3][:, :, ::-1]
        image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        flipped_surface = pygame.transform.flip(image_surface, True, False)
        flipped_surface.blit(self.sheen_surface, (0, 0))
        masked_surface = self.oval_mask.copy()
        masked_surface.blit(flipped_surface, (0, 0), None, pygame.BLEND_RGBA_MULT)
        pygame.draw.ellipse(
            masked_surface,
            (20, 20, 20, 255),
            (0, 0, self.rearview_res_w, self.rearview_res_h),
            4,
        )
        self.rearview_surface = masked_surface
    """
    def _parse_rearview_image(self, image):
            # Common steps for both mirror shapes
            array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
                image.height, image.width, 4
            )[:, :, :3][:, :, ::-1]
            image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            flipped_surface = pygame.transform.flip(image_surface, True, False)

            # Conditional rendering based on vehicle model
            if self._parent.attributes.get('ros_name') == 'vehicle.mitsubishi.fusorosa':
                # --- NEW "3D" RECTANGLE LOGIC for the Mitsubishi Van ---
                
                bezel_thickness = 8
                shadow_offset = 4 # How far the shadow juts out

                # Create a new, larger surface to hold the shadow, bezel, and mirror
                final_width = self.rearview_res_w + (bezel_thickness * 2) + shadow_offset
                final_height = self.rearview_res_h + (bezel_thickness * 2) + shadow_offset
                final_surface = pygame.Surface((final_width, final_height), pygame.SRCALPHA)

                # 1. Draw the drop shadow first (dark, semi-transparent, offset)
                shadow_color = (0, 0, 0, 70)
                shadow_rect = pygame.Rect(shadow_offset, shadow_offset, self.rearview_res_w + (bezel_thickness * 2), self.rearview_res_h + (bezel_thickness * 2))
                pygame.draw.rect(final_surface, shadow_color, shadow_rect, border_radius=20)

                # 2. Draw the main plastic bezel on top of the shadow
                bezel_color = (20, 20, 20, 220)
                bezel_rect = pygame.Rect(0, 0, self.rearview_res_w + (bezel_thickness * 2), self.rearview_res_h + (bezel_thickness * 2))
                pygame.draw.rect(final_surface, bezel_color, bezel_rect, border_radius=10)

                # 3. Draw a bright highlight on the top and left edges to simulate light
                highlight_color = (100, 100, 100, 100)
                pygame.draw.rect(final_surface, highlight_color, bezel_rect, 1, border_radius=10)

                # 4. Blit the actual mirror image onto the bezel
                final_surface.blit(flipped_surface, (bezel_thickness, bezel_thickness))

                # 5. Draw a subtle inner border for the glass
                inner_rect = pygame.Rect(bezel_thickness, bezel_thickness, self.rearview_res_w, self.rearview_res_h)
                pygame.draw.rect(final_surface, (0, 0, 0, 150), inner_rect, 1)

                self.rearview_surface = final_surface

            else:
                # --- ELLIPSE LOGIC for all other vehicles (your existing code) ---
                flipped_surface.blit(self.sheen_surface, (0, 0))
                masked_surface = self.oval_mask.copy()
                masked_surface.blit(flipped_surface, (0, 0), None, pygame.BLEND_RGBA_MULT)
                pygame.draw.ellipse(
                    masked_surface,
                    (20, 20, 20, 255),
                    (0, 0, self.rearview_res_w, self.rearview_res_h),
                    4,
                )
                self.rearview_surface = masked_surface
            
    def _parse_radar_data(self, radar_data):
        radar_surface = pygame.Surface(self.single_monitor_dim, pygame.SRCALPHA)
        center_x, center_y, max_range = (
            self.single_monitor_dim[0] / 2,
            self.single_monitor_dim[1] / 2,
            50.0,
        )
        for detect in radar_data:
            x = center_x + detect.depth * math.cos(detect.azimuth) * (
                self.single_monitor_dim[0] / (2 * max_range)
            )
            y = center_y + detect.depth * math.sin(detect.azimuth) * (
                self.single_monitor_dim[1] / (2 * max_range)
            )
            color = (
                (255, 0, 0)
                if detect.velocity < -0.1
                else (0, 255, 0)
                if detect.velocity > 0.1
                else (255, 255, 0)
            )
            pygame.draw.rect(radar_surface, color, (int(x), int(y), 3, 3))
        self.surface = radar_surface

    def destroy(self):
        sensors_to_destroy = [self.sensor, self.left_camera_sensor, self.rearview_cam]
        for sensor in sensors_to_destroy:
            if sensor and sensor.is_alive:
                sensor.destroy()
