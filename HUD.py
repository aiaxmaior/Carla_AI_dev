import carla 
import pygame
import os
import math
import time
import logging
import weakref
import numpy as np

# This import is kept as it might be part of your project structure.
try:
    import Sensors
except ImportError:
    logging.warning("Sensors module not found. Some functionality might be limited.")


class EndScreen(object):
    """
    Manages the display and interaction of the end-of-session screen.
    This screen is shown after a catastrophic failure.
    This version features a gradient background and logo watermark.
    """

    def __init__(self, display_surface, final_scores: dict, hud_fonts: dict):
        """
        Initializes the EndScreen.
        Args:
            display_surface: The main pygame display to draw on.
            final_scores: A dictionary containing the final calculated scores.
            hud_fonts: A dictionary of pre-loaded fonts from the main HUD class.
        """
        self.surface = display_surface
        self.dim = display_surface.get_size()
        self.final_scores = final_scores
        self.fonts = hud_fonts

        # --- ADDED: Caching for the gradient background ---
        self._background_surface = None

        # --- Logo Loading and Scaling ---
        self.logo_img = None
        try:
            logo_surface = pygame.image.load("./images/logo-qryde.png").convert_alpha()
            original_size = logo_surface.get_size()
            scaled_size = (int(original_size[0] * 0.2), int(original_size[1] * 0.2))
            self.logo_img = pygame.transform.smoothscale(logo_surface, scaled_size)
        except pygame.error as e:
            logging.warning(f"Could not load logo image for EndScreen: {e}.")

        # Define colors for the UI elements
        self.title_color = (220, 38, 38)
        self.score_color = (252, 211, 77)
        self.text_color = (229, 231, 235)
        self.button_color = (55, 65, 81)
        self.button_hover_color = (75, 85, 99)
        self.button_text_color = (255, 255, 255)

        # Define button layout
        button_w, button_h = 350, 60
        button_y_start = self.dim[1] * 0.65
        button_spacing = button_h + 20
        center_x = self.dim[0] / 2

        self.buttons = {
            "restart": pygame.Rect(
                center_x - button_w / 2, button_y_start, button_w, button_h
            ),
            "exit": pygame.Rect(
                center_x - button_w / 2,
                button_y_start + button_spacing,
                button_w,
                button_h,
            ),
        }
        self.button_labels = {
            "restart": "Restart Simulation",
            "exit": "Exit to Desktop",
        }

    def draw(self):
        """Draws all visual elements of the end screen for one frame."""
        # --- MODIFIED: Draw and cache the gradient background ---
        if (
            self._background_surface is None
            or self._background_surface.get_size() != self.dim
        ):
            self._background_surface = pygame.Surface(self.dim)
            top_color = (44, 62, 80)
            bottom_color = (27, 38, 49)
            height = self.dim[1]
            for y in range(height):
                r = top_color[0] + (bottom_color[0] - top_color[0]) * y // height
                g = top_color[1] + (bottom_color[1] - top_color[1]) * y // height
                b = top_color[2] + (bottom_color[2] - top_color[2]) * y // height
                pygame.draw.line(
                    self._background_surface, (r, g, b), (0, y), (self.dim[0], y)
                )
 
        self.surface.blit(self._background_surface, (0, 0))

        # Draw the main title
        title_font = self.fonts.get("_font_score_hud", pygame.font.Font(None, 82))
        title_surf = title_font.render("SESSION ENDED", True, self.title_color)
        title_rect = title_surf.get_rect(center=(self.dim[0] / 2, self.dim[1] * 0.25))
        self.surface.blit(title_surf, title_rect)

        # Draw scores
        score_font = self.fonts.get("_font_primary_hud", pygame.font.Font(None, 64))
        score_text_font = self.fonts.get(
            "_font_secondary_hud", pygame.font.Font(None, 48)
        )
        score_title_surf = score_text_font.render(
            "Final Overall Score:", True, self.text_color
        )
        score_title_rect = score_title_surf.get_rect(
            center=(self.dim[0] / 2, self.dim[1] * 0.45)
        )
        self.surface.blit(score_title_surf, score_title_rect)
        overall_score = self.final_scores.get("overall", 0)
        score_surf = score_font.render(f"{overall_score:.1f}", True, self.score_color)
        score_rect = score_surf.get_rect(
            center=(self.dim[0] / 2, self.dim[1] * 0.45 + 55)
        )
        self.surface.blit(score_surf, score_rect)

        # Draw buttons
        mouse_pos = pygame.mouse.get_pos()
        button_font = self.fonts.get("_font_primary_hud", pygame.font.Font(None, 32))
        for key, rect in self.buttons.items():
            color = (
                self.button_hover_color
                if rect.collidepoint(mouse_pos)
                else self.button_color
            )
            pygame.draw.rect(self.surface, color, rect, border_radius=12)
            label_surf = button_font.render(
                self.button_labels[key], True, self.button_text_color
            )
            label_rect = label_surf.get_rect(center=rect.center)
            self.surface.blit(label_surf, label_rect)

        # Draw logo watermark
        if self.logo_img:
            logo_rect = self.logo_img.get_rect()
            logo_rect.topright = (self.dim[0] - 20, 20)
            self.surface.blit(self.logo_img, logo_rect)

        pygame.display.flip()

    def run(self):
        """Runs the event loop for the end screen, waiting for user action."""
        logging.info("EndScreen is now active, waiting for user input.")
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    return "exit"
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for key, rect in self.buttons.items():
                        if rect.collidepoint(event.pos):
                            logging.info(f"User clicked '{self.button_labels[key]}'.")
                            return key
            self.draw()


class HUD(object):
    """
    Manages the heads-up display (HUD) for the simulation.
    """

    def __init__(self, width, height, args):
        self.dim = (width, height)
        self.idle_throttle = None
        self.idle_brake = None
        self.custom_notification_font_path = os.path.join(
            args.carla_root,
            "CarlaUE4",
            "Content",
            "Carla",
            "Fonts",
            "tt-supermolot-neue-trl.bd-it.ttf",
        )
        self._use_custom_notification_font = False
        self._server_clock = pygame.time.Clock()

        try:
            if os.path.exists(self.custom_notification_font_path):
                pygame.font.Font(self.custom_notification_font_path, 1)
                self._use_custom_notification_font = True
                logging.info(f"Custom font found: {self.custom_notification_font_path}")
            else:
                logging.warning(
                    "Custom font not found. Falling back to system defaults."
                )
        except Exception as e:
            logging.warning(
                f"Error loading custom font: {e}. Falling back to system defaults."
            )

        # --- BUG FIX & ROBUSTNESS IMPROVEMENT ---
        # Using a robust font loading method to prevent silent failures.
        font_path = (
            self.custom_notification_font_path
            if self._use_custom_notification_font
            else None
        )

        self._font_title_hud = (
            pygame.font.Font(font_path, 34) if font_path else pygame.font.Font(None, 50)
        )
        self._font_primary_hud = (
            pygame.font.Font(font_path, 28) if font_path else pygame.font.Font(None, 40)
        )
        self._font_secondary_hud = (
            pygame.font.Font(font_path, 20) if font_path else pygame.font.Font(None, 40)
        )
        self._font_score_hud = (
            pygame.font.Font(font_path, 36) if font_path else pygame.font.Font(None, 40)
        )

        # --- NEW: Specific fonts for the glassmorphic panel ---
        self.panel_fonts = {
            "title": (
                pygame.font.Font(font_path, 42)
                if font_path
                else pygame.font.Font(None, 24)
            ),
            "main_score": (
                pygame.font.Font(font_path, 80)
                if font_path
                else pygame.font.Font(None, 80)
            ),
            "sub_label": (
                pygame.font.Font(font_path, 40)
                if font_path
                else pygame.font.Font(None, 40)
            ),
            "sub_value": (
                pygame.font.Font(font_path, 40)
                if font_path
                else pygame.font.Font(None, 40)
            ),
            "large_val": (
                pygame.font.Font(font_path, 56)
                if font_path
                else pygame.font.Font(None, 56)
            ),
            "small_label": (
                pygame.font.Font(font_path, 18)
                if font_path
                else pygame.font.Font(None, 18)
            ),
        }

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

        pygame.mixer.init()
        self.sounds = {}
        self.sound_cooldowns = {
            "lane_drift": 3.0,
            "solid_line_crossing": 2.0,
            "oncoming_traffic_violation": 1.0,
            "speeding": 5.0,
            "collision": 0.0,  # Collision sound typically plays immediately
            "error": 0.0,  # Error sounds typically play immediately
            "default_notification": 0.0,
            "off_road": 2.0,  # Cooldown for off-road sound
            "over_the_median": 3.0,  # Crossing the median
        }
        self._last_sound_time = {sound_type: 0.0 for sound_type in self.sound_cooldowns}

        sound_files = {
            "collision": "./audio/alerts/collision_alert_sound.wav",
            "lane_drift": "./audio/alerts/lane_deviation_sound.wav",
            "solid_line_crossing": "./audio/alerts/solid_line_sound.wav",
            "oncoming_traffic_violation": "./audio/alerts/oncoming_alert_sound.wav",
            "speeding": "./audio/alerts/speed_violation_sound.wav",
            "error": "./audio/alerts/error_encountered_sound.wav",
            "default_notification": "./audio/alerts/alert_sound.wav",
            "off_road": "./audio/alerts/off_road_sound.wav",
            "over_the_median": "./audio/alerts/over_the_median.wav",
        }
        for sound_type, filename in sound_files.items():
            try:
                if os.path.exists(filename):
                    self.sounds[sound_type] = pygame.mixer.Sound(filename)
                    self.sounds[sound_type].set_volume(0.5)
                else:
                    self.sounds[sound_type] = None
                    logging.warning(
                        f"Sound file '{filename}' for '{sound_type}' not found."
                    )
            except pygame.error as e:
                self.sounds[sound_type] = None
                logging.warning(f"Could not load sound '{filename}': {e}")

        # UI helper elements
        self._persistent_warning = PersistentWarning(
            self._font_secondary_hud, self.dim, (0, 0)
        )
        self.help = HelpText(self._font_secondary_hud, width, height)

        # HUD state variables
        # HUD state variables
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []  # Will be populated in tick method
        self._server_clock = pygame.time.Clock()
        self._active_notifications = []  # List of active BlinkingAlerts
        self._notification_base_pos_y = int(
            self.dim[1] * 0.85
        )  # Base Y position for notifications
        self._notification_spacing = 8  # Spacing between stacked notifications
        self._last_speed_warning_frame_warned = 0  # Added: Initialize this attribute

        # MVD Score displays (updated by game_loop)
        self.overall_dp_score_display = 100  # Initial score
        self.mbi_display = 1.0
        self.lmi_display = 1.0
        self.mvd_event_message_display = (
            ""  # For showing "Collision!", "Lane Violation!" etc.
        )
        self.mvd_event_message_lifetime = 0.0
        self.mvd_event_message_start_time = 0.0

        self._blinker_state = 0

        self.reset()
    def get_font_dictionary(self):
        """Returns a dictionary of the HUD's fonts for use in other UI classes."""
        return {
            "_font_title_hud": self._font_title_hud,
            "_font_primary_hud": self._font_primary_hud,
            "_font_secondary_hud": self._font_secondary_hud,
            "_font_score_hud": self._font_score_hud,
        }

    def reset(self):
        self.overall_dp_score_display = 100
        self.mbi_display = 1.0
        self.lmi_display = 1.0
        # ... (rest of reset is unchanged) ...

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def play_sound_for_event(self, event_type, force_play=False):
        """
        Plays a sound for a given event type, respecting cooldowns.
        :param event_type: Key for the sound in self.sounds.
        :param force_play: If True, plays sound regardless of cooldown.
        """
        sound_to_play = self.sounds.get(event_type) or self.sounds.get(
            "default_notification"
        )
        if not sound_to_play:
            return

        current_time = time.time()
        cooldown = self.sound_cooldowns.get(event_type, 0.0)  # Default to no cooldown

        if force_play or (
            current_time > self._last_sound_time.get(event_type, 0.0) + cooldown
        ):
            try:
                sound_to_play.play()
                self._last_sound_time[event_type] = current_time
            except pygame.error as e:
                logging.warning(f"Error playing sound '{event_type}': {e}")

    def update_mvd_scores_for_display(self, dp_score, mbi, lmi):
        """Updates the MVD scores to be displayed on the HUD."""
        self.overall_dp_score_display = dp_score
        self.mbi_display = mbi
        self.lmi_display = lmi

    def show_mvd_event_message(self, message, duration=2.0):
        """Displays a temporary MVD-related event message on the HUD."""
        self.mvd_event_message_display = message
        self.mvd_event_message_lifetime = duration
        self.mvd_event_message_start_time = time.time()

    def toggle_info(self):
        """Toggles the display of the main info panel."""
        self._show_info = not self._show_info

    def notification(
        self,
        text,
        seconds=2.0,
        text_color=(255, 255, 255),
        symbol_enabled=False,
        symbol_color=(255, 0, 0),
        is_blinking=False,
        is_critical_center=False,
    ):
        """
        Displays a notification message on the HUD.
        :param text: The message to display.
        :param seconds: How long the notification should be visible.
        :param text_color: RGB tuple for text color.
        :param symbol_enabled: Whether to show a warning symbol.
        :param symbol_color: Color of the warning symbol.
        :param is_blinking: If the text should blink.
        :param is_critical_center: If the notification should be large and centered.
        """
        font_sz, sym_sz = (96, 56) if is_critical_center else (96, 42)
        font_path = (
            self.custom_notification_font_path
            if self._use_custom_notification_font
            else None
        )

        try:
            # Using Font(font_path, size) or Font(None, size) for system defaults
            txt_font = (
                pygame.font.Font(font_path, font_sz)
                if font_path
                else pygame.font.Font(None, font_sz)
            )
            sym_font = (
                pygame.font.Font(font_path, sym_sz)
                if font_path
                else pygame.font.Font(None, sym_sz)
            )
        except pygame.error as e:
            logging.warning(
                f"HUD Font Error for notification: {e}. Falling back to default system fonts."
            )
            txt_font = pygame.font.Font(None, font_sz)
            sym_font = pygame.font.Font(None, sym_sz)

        notif_w, notif_h = (self.dim[0] * 0.5, 70) if is_critical_center else (400, 50)
        new_notif = BlinkingAlert(txt_font, self.dim, (notif_w, notif_h), sym_font)
        new_notif.set_text(
            text,
            text_color,
            seconds,
            symbol_enabled,
            symbol_color,
            is_blinking,
            is_critical_center,
        )

        # Prevent duplicate non-critical notifications stacking rapidly
        if not is_critical_center:
            for n in self._active_notifications:
                if (
                    n.text == text and n.seconds_left > 0.5
                ):  # If same text and still visible
                    n.seconds_left = seconds  # Just refresh its duration
                    return
        self._active_notifications.append(new_notif)
        self._active_notifications.sort(
            key=lambda n: not n.is_critical_center
        )  # Critical alerts on top

    def error(self, text):
        """Displays a critical error notification on the HUD."""
        self.notification(
            f"ERROR: {text.upper()}", 5.0, (255, 50, 50), True, (255, 0, 0), True, True
        )
        self.play_sound_for_event("error", force_play=True)    # ... (notification, error, tick methods are largely unchanged but now rely on the robustly loaded fonts)

    def tick(self, world_instance, clock, idling, controller):
        self._blinker_state = controller.get_blinker_state()
        self._active_notifications = [
            n for n in self._active_notifications if n.tick(world_instance, clock)
        ]

        self._info_text = []
        if self._show_info and world_instance.player and world_instance.player.is_alive:
            v = world_instance.player.get_velocity()
            c = world_instance.player.get_control()
            speed_mph = int(2.237 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
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
            # ... (speeding warning logic unchanged)
        else:
            self._info_text = [("title", "Player not ready")]

    def render(self, display):
        """Renders the main glassmorphic HUD panel."""
        if self._show_info and self._info_text:
            panel_h = display.get_height() * 0.2
            panel_w = display.get_width() * 0.15
            panel_x = display.get_width() * 0.4
            panel_y = display.get_height() * 0.58
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
                    gear_x_pos = panel_x + panel_w - gear_surf.get_width() - padding - 15
                    display.blit(gear_surf, (gear_x_pos, v_offset))
                    display.blit(gear_label_surf, (gear_x_pos, v_offset + 40))

        # --- ALL OTHER ORIGINAL RENDERING LOGIC IS PRESERVED BELOW ---

        if self.mvd_event_message_display and time.time() < self.mvd_event_message_start_time + self.mvd_event_message_lifetime:
            event_font = pygame.font.Font(self.font_path, 48) if self._use_custom_notification_font else pygame.font.Font(None, 48)
            text_surface = event_font.render(self.mvd_event_message_display, True, (255, 50, 50))
            text_rect = text_surface.get_rect(center=(self.dim[0] // 2, self.dim[1] * 0.3))

            bg_rect = text_rect.inflate(20, 10)
            bg_surface = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            bg_surface.fill((0, 0, 0, 180))
            display.blit(bg_surface, bg_rect.topleft)
            display.blit(text_surface, text_rect.topleft)
        elif self.mvd_event_message_display:
            self.mvd_event_message_display = ""

        y_off = self._notification_base_pos_y
        for notif in reversed(self._active_notifications):
            if not notif.seconds_left > 0 and notif.surface.get_alpha() == 0:
                continue
            if notif.is_critical_center:
                notif.render(display)
            else:
                x_pos = (self.dim[0] - notif.surface.get_width()) // 2
                y_pos = y_off - notif.surface.get_height()
                if y_pos < self.dim[1] * 0.15:
                    break
                display.blit(notif.surface, (x_pos, y_pos))
                y_off -= notif.surface.get_height() + self._notification_spacing

        self._render_blinker_indicator(display)

        if self.help:
            self.help.render(display)
        if self._persistent_warning:
            self._persistent_warning.render(display)

    def _render_blinker_indicator(self, display):
        if self._blinker_state == 0 or (self._blinker_left_img is None and self._blinker_right_img is None):
            return
        if (pygame.time.get_ticks() // 500) % 2 == 0:
            return
        y_pos = self.dim[1] - 80
        if self._blinker_state == 1 and self._blinker_left_img:
            rect = self._blinker_left_img.get_rect(center=(self.dim[0] // 2, y_pos))
            display.blit(self._blinker_left_img, rect)
        elif self._blinker_state == 2 and self._blinker_right_img:
            rect = self._blinker_right_img.get_rect(center=(self.dim[0] // 2, y_pos))
            display.blit(self._blinker_right_img, rect)
        elif self._blinker_state == 3 and self._blinker_left_img and self._blinker_right_img:
            left_rect = self._blinker_left_img.get_rect(midright=(self.dim[0] // 2 - 10, y_pos))
            right_rect = self._blinker_right_img.get_rect(midleft=(self.dim[0] // 2 + 10, y_pos))
            display.blit(self._blinker_left_img, left_rect)
            display.blit(self._blinker_right_img, right_rect)


class BlinkingAlert(object):
    """
    A UI element for displaying temporary, blinking, and/or critical alerts.
    """
    def __init__(self, font, screen_dim, initial_dim, symbol_font=None):
        self.font = font
        self.screen_dim = screen_dim
        self.initial_dim = initial_dim
        self.current_pos = [0, 0]
        self.seconds_left = 0
        self.surface = pygame.Surface(self.initial_dim, pygame.SRCALPHA)
        self.start_time = 0.0
        self.duration = 0.0
        self.text = ""
        self.is_blinking = False
        self.is_critical_center = False
        self.symbol_font = symbol_font if symbol_font else pygame.font.Font(pygame.font.get_default_font(), int(self.initial_dim[1] * 0.7))
        self.bounce_height = 30.0
        self.bounce_frequency = 2.5
        self.num_bounces = 2
        self.outline_color = (0, 0, 0)
        self.outline_thickness = 3
        self.vertical_bar_width = 12
        self.vertical_bar_color = (128, 0, 128)

    def set_text(self, text, text_color=(255, 255, 255), seconds=2.0, symbol_enabled=True, symbol_color=(255, 255, 0), is_blinking=False, is_critical_center=False):
        self.text = text
        self.seconds_left = seconds
        self.start_time = pygame.time.get_ticks() / 1000.0
        self.duration = seconds if seconds > 0 else 0.001
        self.is_blinking = is_blinking
        self.is_critical_center = is_critical_center
        self.vertical_bar_color = symbol_color
        
        symbol_texture_main, symbol_texture_outline = None, None
        if symbol_enabled:
            symbol_text_str = "⚠"
            symbol_texture_main = self.symbol_font.render(symbol_text_str, True, symbol_color)
            symbol_texture_outline = self.symbol_font.render(symbol_text_str, True, self.outline_color)

        display_text_str = text.upper() if self.is_critical_center or self.is_blinking else text
        text_texture_main = self.font.render(display_text_str, True, text_color)
        text_texture_outline = self.font.render(display_text_str, True, self.outline_color)

        padding_horizontal = 20 if self.is_critical_center else 15
        padding_vertical = 15 if self.is_critical_center else 10

        content_main_height = max(symbol_texture_main.get_height() if symbol_texture_main else 0, text_texture_main.get_height())
        box_height = content_main_height + 2 * padding_vertical
        content_main_width = text_texture_main.get_width()
        if symbol_texture_main:
            content_main_width += symbol_texture_main.get_width() + (padding_horizontal // 2)
        box_width = content_main_width + (padding_horizontal * 2) + self.vertical_bar_width + (padding_horizontal // 3)

        if not self.is_critical_center:
            box_width = max(box_width, self.initial_dim[0])
            box_height = max(box_height, self.initial_dim[1])
        self.surface = pygame.Surface((box_width, box_height), pygame.SRCALPHA)

        base_rgb = (0, 0, 0)
        gradient_area_width = box_width - self.vertical_bar_width - (padding_horizontal // 3)
        alpha_left, alpha_right = int(255 * 0.1), 220
        for x_col in range(gradient_area_width):
            ratio = x_col / (gradient_area_width - 1) if gradient_area_width > 1 else 0
            current_alpha = int(alpha_left + (alpha_right - alpha_left) * ratio)
            line_color = (base_rgb[0], base_rgb[1], base_rgb[2], max(0, min(255, current_alpha)))
            pygame.draw.line(self.surface, line_color, (x_col, 0), (x_col, box_height))

        bar_x = gradient_area_width
        bar_rect = pygame.Rect(bar_x, 0, self.vertical_bar_width, box_height)
        pygame.draw.rect(self.surface, self.vertical_bar_color, bar_rect)

        blit_x_start_content = padding_horizontal
        current_blit_x_main = blit_x_start_content
        symbol_y_main = (box_height - (symbol_texture_main.get_height() if symbol_texture_main else 0)) // 2
        text_y_main = (box_height - text_texture_main.get_height()) // 2
        
        offsets = [p for p in ((i, j) for i in range(-self.outline_thickness, self.outline_thickness + 1) for j in range(-self.outline_thickness, self.outline_thickness + 1)) if p != (0, 0)]
        
        temp_blit_x_outline = blit_x_start_content
        if symbol_texture_outline:
            for dx, dy in offsets:
                self.surface.blit(symbol_texture_outline, (temp_blit_x_outline + dx, symbol_y_main + dy))
            temp_blit_x_outline += symbol_texture_outline.get_width() + (padding_horizontal // 2)
        for dx, dy in offsets:
            self.surface.blit(text_texture_outline, (temp_blit_x_outline + dx, text_y_main + dy))

        if symbol_texture_main:
            self.surface.blit(symbol_texture_main, (current_blit_x_main, symbol_y_main))
            current_blit_x_main += symbol_texture_main.get_width() + (padding_horizontal // 2)
        self.surface.blit(text_texture_main, (current_blit_x_main, text_y_main))
        
        if self.is_critical_center:
            target_center_y = int(self.screen_dim[1] * 0.40)
            self.initial_pos = [(self.screen_dim[0] - box_width) // 2, target_center_y - box_height // 2]
        else:
            self.initial_pos = [(self.screen_dim[0] - box_width) // 2, self.screen_dim[1]]
        self.current_pos = list(self.initial_pos)

    def tick(self, world_instance, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        alpha = 0
        if self.seconds_left > 0 and self.duration > 0:
            if self.is_blinking:
                elapsed_time = (pygame.time.get_ticks() / 1000.0) - self.start_time
                blink_freq = 3.0 if self.is_critical_center else 2.0
                alpha = int(abs(math.sin(elapsed_time * math.pi * blink_freq)) * 255)
            else:
                alpha = int(255 * (self.seconds_left / self.duration))
            self.surface.set_alpha(max(0, min(255, alpha)))
            if self.is_critical_center and (self.seconds_left > (self.duration - self.num_bounces / self.bounce_frequency)):
                elapsed_bounce_time = self.duration - self.seconds_left
                bounce_offset_y = self.bounce_height * (0.5 * (1 - math.cos(elapsed_bounce_time * math.pi * self.bounce_frequency * 2)))
                if elapsed_bounce_time < (1 / (self.bounce_frequency * 2)):
                    bounce_offset_y = self.bounce_height * math.sin(elapsed_bounce_time * math.pi * self.bounce_frequency * 2)
                self.current_pos[1] = self.initial_pos[1] - bounce_offset_y
            elif self.is_critical_center:
                self.current_pos[1] = self.initial_pos[1]
        else:
            self.surface.set_alpha(0)
        return self.seconds_left > 0

    def render(self, display):
        if self.surface.get_alpha() > 0:
            display.blit(self.surface, self.current_pos)

class HelpText(object):
    """
    Displays multi-line help text on the HUD.
    """
    def __init__(self, font_object, width, height):
        doc_lines = [
            "MVD Controls:", "W/S: Throttle/Brake", "A/D: Steer", "Q: Toggle Reverse",
            "SPACE: Handbrake", "P: Toggle Autopilot", "ESC: Quit", "TAB: Change Camera",
            "H: Toggle Help", "---", "Real-time Driving Score Demo",
        ]
        self.font = font_object
        self.dim = (width, height)
        self._render = False
        line_height = self.font.get_linesize()
        max_line_width = max(self.font.size(line)[0] for line in doc_lines) if doc_lines else 0
        surf_width = min(max_line_width + 44, int(width * 0.8))
        surf_height = min(len(doc_lines) * line_height + 24, int(height * 0.8))
        self.surface_dim = (surf_width, surf_height)
        self.pos = (0.5 * width - 0.5 * surf_width, 0.5 * height - 0.5 * surf_height)
        self.surface = pygame.Surface(self.surface_dim, pygame.SRCALPHA)
        self.surface.fill((0, 0, 0, 0))
        pygame.draw.rect(self.surface, (0, 0, 0, 200), self.surface.get_rect(), border_radius=15)
        current_y = 12
        for line in doc_lines:
            if current_y + line_height > self.surface_dim[1] - 12: break
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, current_y))
            current_y += line_height

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)

class PersistentWarning(object):
    """
    Displays a persistent warning message in a corner of the screen.
    """
    def __init__(self, font_object, dim, pos):
        self.font = font_object
        self.screen_dim = dim
        self.text_surface = None
        self.background_color = (80, 80, 0, 180)
        self.text_color = (255, 255, 200)
        self.symbol_color = (255, 255, 0)
        self.is_active = False
        self.symbol_font = pygame.font.Font(pygame.font.get_default_font(), 48)

    def set_warning_status(self, text="", active=False):
        self.is_active = active
        if self.is_active:
            symbol_text = "⚠"
            symbol_texture = self.symbol_font.render(symbol_text, True, self.symbol_color)
            display_text = text.upper()
            text_texture = self.font.render(display_text, True, self.text_color)
            padding = 8
            content_height = max(symbol_texture.get_height(), text_texture.get_height())
            box_height = content_height + 2 * padding
            content_width = text_texture.get_width() + symbol_texture.get_width() + (padding // 2)
            box_width = content_width + 2 * padding
            self.text_surface = pygame.Surface((box_width, box_height), pygame.SRCALPHA)
            pygame.draw.rect(self.text_surface, self.background_color, self.text_surface.get_rect(), border_radius=5)
            current_x_pos = padding
            symbol_y_pos = (box_height - symbol_texture.get_height()) // 2
            text_y_pos = (box_height - text_texture.get_height()) // 2
            self.text_surface.blit(symbol_texture, (current_x_pos, symbol_y_pos))
            current_x_pos += symbol_texture.get_width() + (padding // 2)
            self.text_surface.blit(text_texture, (current_x_pos, text_y_pos))
        else:
            self.text_surface = None

    def tick(self, world, clock):
        pass

    def render(self, display):
        if self.is_active and self.text_surface:
            render_pos_x = self.screen_dim[0] - self.text_surface.get_width() - 10
            render_pos_y = 10
            display.blit(self.text_surface, (render_pos_x, render_pos_y))

class CameraManager(object):
    """
    Manages camera sensors and their display on the Pygame surface.
    """
    def __init__(self, parent_actor, hud_instance, fov=120.0):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud_instance
        self.fov = fov
        self.recording = False
        self.driver_view = None
        self.rearview_cam = None
        self.rearview_surface = None
        self.display_dim = hud_instance.dim
        self.rearview_res_w = self.display_dim[0] // 3.5
        self.rearview_res_h = self.display_dim[1] // 7
        self.oval_mask = pygame.Surface((self.rearview_res_w, self.rearview_res_h), pygame.SRCALPHA)
        pygame.draw.ellipse(self.oval_mask, (255, 255, 255, 255), (0, 0, self.rearview_res_w, self.rearview_res_h))
        self.sheen_surface = pygame.Surface((self.rearview_res_w, self.rearview_res_h), pygame.SRCALPHA)
        pygame.draw.polygon(self.sheen_surface, (255, 255, 255, 20), [(-50, 0), (150, 0), (100, self.rearview_res_h), (-100, self.rearview_res_h)])

        self.rear_view = carla.Transform(carla.Location(x=-10, z=7), carla.Rotation(pitch=-20))
        if fov > 120: self.driver_view = carla.Transform(carla.Location(x=1.2, y=-0.5, z=1.82), carla.Rotation(pitch=0))
        elif fov > 110: self.driver_view = carla.Transform(carla.Location(x=1.075, y=-0.5, z=1.82), carla.Rotation(pitch=0))
        elif fov >100: self.driver_view = carla.Transform(carla.Location(x=0.95, y=-0.5, z=1.82), carla.Rotation(pitch=0))
        else: self.driver_view = carla.Transform(carla.Location(x=0.80, y=-0.5, z=1.82), carla.Rotation(pitch=0))
        self._camera_transforms = [self.driver_view, self.rear_view]
        self.transform_index = 1
        
        self.sensors_config = [
            ["sensor.camera.rgb", carla.ColorConverter.Raw, "Camera RGB"],
            ["sensor.camera.depth", carla.ColorConverter.LogarithmicDepth, "Camera Depth (Logarithmic Gray Scale)"],
            ["sensor.camera.semantic_segmentation", carla.ColorConverter.CityScapesPalette, "Camera Semantic Segmentation (CityScapes Palette)"],
            ["sensor.lidar.ray_cast", None, "Lidar (Ray-Cast)"],
        ]
        self.sensor_blueprints = []
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item_config in self.sensors_config:
            bp = bp_library.find(item_config[0])
            if item_config[0].startswith("sensor.camera"):
                bp.set_attribute("image_size_x", str(self.hud.dim[0]))
                bp.set_attribute("image_size_y", str(self.hud.dim[1]))
                if bp.has_attribute("fov"): bp.set_attribute("fov", str(self.fov))
            elif item_config[0].startswith("sensor.lidar"):
                if bp.has_attribute("range"): bp.set_attribute("range", "50")
            self.sensor_blueprints.append(bp)
        self.index = None
        self._spawn_rearview_camera()

    def _spawn_rearview_camera(self):
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        rear_cam_bp = bp_library.find('sensor.camera.rgb')
        rear_cam_bp.set_attribute('image_size_x', str(self.rearview_res_w))
        rear_cam_bp.set_attribute('image_size_y', str(self.rearview_res_h))
        transform = carla.Transform(carla.Location(x=-2.5, z=1.2), carla.Rotation(yaw=180))
        try:
            self.rearview_cam = world.spawn_actor(rear_cam_bp, transform, attach_to=self._parent)
            weak_self = weakref.ref(self)
            self.rearview_cam.listen(lambda image: CameraManager._parse_rearview_image(weak_self, image))
        except Exception as e:
            logging.error(f"Failed to spawn rearview camera: {e}")
            self.rearview_cam = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        if self.sensor and self.sensor.is_alive:
            try:
                self.sensor.set_transform(self._camera_transforms[self.transform_index])
            except RuntimeError as e:
                logging.error(f"Error setting camera transform: {e}")

    def set_sensor(self, index, notify=True):
        index = index % len(self.sensors_config)
        needs_respawn = True
        if self.index is not None and self.sensors_config[index][0] == self.sensors_config[self.index][0]:
            needs_respawn = False
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            try:
                self.sensor = self._parent.get_world().spawn_actor(self.sensor_blueprints[index], self._camera_transforms[self.transform_index], attach_to=self._parent)
            except RuntimeError as e:
                logging.error(f"Error spawning sensor: {e}")
                self.sensor = None
                return
            if self.sensor:
                weak_self = weakref.ref(self)
                self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors_config[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1 if self.index is not None else 0)

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))
        if self.rearview_surface is not None:
            x_pos = int(display.get_width() // 2 - self.rearview_res_w // 2)
            y_pos = int(display.get_height() * 0.10 - self.rearview_res_h // 2)
            display.blit(self.rearview_surface, (x_pos, y_pos))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self or self.index is None: return
        sensor_type, color_converter, _ = self.sensors_config[self.index]
        if sensor_type.startswith("sensor.lidar"):
            points = np.frombuffer(image.raw_data, dtype=np.dtype("f4"))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2 * 50.0)
            lidar_data[:, 0] += self.hud.dim[0] / 2.0
            lidar_data[:, 1] += self.hud.dim[1] / 2.0
            lidar_data = np.fabs(lidar_data).astype(np.int32)
            valid_points = (lidar_data[:, 0] < self.hud.dim[0]) & (lidar_data[:, 1] < self.hud.dim[1])
            lidar_data = lidar_data[valid_points]
            lidar_img = np.zeros((self.hud.dim[1], self.hud.dim[0], 3), dtype=np.uint8)
            lidar_img[lidar_data[:, 1], lidar_data[:, 0]] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img.swapaxes(0, 1))
        elif sensor_type.startswith("sensor.camera"):
            if color_converter is not None:
                image.convert(color_converter)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))[:, :, :3][:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    @staticmethod
    def _parse_rearview_image(weak_self, image):
        self = weak_self()
        if not self: return
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))[:, :, :3][:, :, ::-1]
        image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        flipped_surface = pygame.transform.flip(image_surface, True, False)
        flipped_surface.blit(self.sheen_surface, (0,0), None, pygame.BLEND_RGBA_ADD)
        masked_surface = self.oval_mask.copy()
        masked_surface.blit(flipped_surface, (0, 0), None, pygame.BLEND_RGBA_MULT)
        pygame.draw.ellipse(masked_surface, (20, 20, 20, 255), (0, 0, self.rearview_res_w, self.rearview_res_h), 4)
        self.rearview_surface = masked_surface
