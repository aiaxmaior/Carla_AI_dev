# ============================================================================
# PERF CHECK (file-level):
# ============================================================================
# [X] | Role: UI helpers (EndScreen, warnings, alerts, help text)
# [X] | Hot-path functions: PersistentWarningManager.render() in HUD loop
# [X] |- Heavy allocs in hot path? Moderate - font renders, surface blits
# [ ] |- pandas/pyarrow/json/disk/net in hot path? No
# [X] | Graphics here? YES - EndScreen rendering, warnings
# [ ] | Data produced (tick schema?): UI state only
# [ ] | Storage (Parquet/Arrow/CSV/none): None
# [ ] | Queue/buffer used?: No
# [ ] | Session-aware? No
# [X] | Debug-only heavy features?: HelpText overlay
# Top 3 perf risks:
# 1. [PERF_HOT] EndScreen.run() blocks main loop (acceptable - end of session)
# 2. [PERF_HOT] PersistentWarningManager font renders if called every frame
# 3. [PERF_OK] Most helpers NOT in hot path (EndScreen, HelpText)
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
from Core.Vision.VisionPerception import Perception
from EventManager import EventManager
from Utility.Font.FontIconLibrary import FontLibrary,IconLibrary

iLib = IconLibrary()
fLib = FontLibrary()
SCALE_FACTOR_GLOBAL = 1.0

class EndScreen(object):
    def __init__(self, display_surface, final_scores: dict, hud_fonts: dict, data_ingestor):
        self.surface = display_surface
        self.data_ingestor = data_ingestor
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

        # --- Center on full ultrawide display ---
        center_x = self.dim[0] // 2

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

        # --- Center on full ultrawide display ---
        center_x = self.dim[0] // 2

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

        # Score area width based on full display, not single panel
        score_area_width = self.dim[0] * 0.3
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
    def __init__(self, dim, scale_factor=1.0):
        self.scale_factor = scale_factor
        self.panel_fonts = fLib.get_loaded_fonts(font="tt-supermolot-neue-trl.bd-it", type="panel_fonts", scale=self.scale_factor)
        self.font = self.panel_fonts['small_label']
        self.screen_dim = dim
        self.active_warnings = {}

    def add_warning(self, key, text, is_critical_center=False):
        """Add a warning with backward compatibility - is_critical_center defaults to False"""
        entry = {'warning': text, 'is_critical_center': is_critical_center}
        self.active_warnings[key] = entry

    def remove_warning(self, key):
        if key in self.active_warnings:
            del self.active_warnings[key]

    def get_warnings(self):
        return list(self.active_warnings.values())

    def render(self, display, panel_rect, start_y):
        if not self.active_warnings:
            return
        y_offset = start_y

        # Split warnings into center and panel categories
        center_warnings = [data['warning'] for data in self.active_warnings.values() if data.get('is_critical_center', False)]
        panel_warnings = [data['warning'] for data in self.active_warnings.values() if not data.get('is_critical_center', False)]

        # Render center warnings (critical) - centered on full ultrawide display
        if center_warnings:
            font = self.panel_fonts['large_val']
            # Center on full ultrawide display
            center_x = self.screen_dim[0] // 2
            y_pos = int(self.screen_dim[1] * 0.40)

            for text in center_warnings:
                # Render text with black border for visibility
                text_surf = font.render(text, True, (255, 50, 50))
                bg_rect = text_surf.get_rect(center=(center_x, y_pos + text_surf.get_height()))
                bg_rect.inflate_ip(20, 15)
                # 80% opacity background (204 = 0.8 * 255)
                bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
                bg_surf.fill((20, 20, 20, 204))

                display.blit(bg_surf, bg_rect.topleft)
                display.blit(text_surf, text_surf.get_rect(center=bg_rect.center))
                y_pos += bg_rect.height + 10

        # Render panel warnings (non-critical)
        for warning_text in panel_warnings:
            try:
                symbol_font = pygame.font.Font(pygame.font.get_default_font(), 18)
                symbol_texture = symbol_font.render("⚠", True, (255, 255, 0))
                text_texture = self.font.render(warning_text, True, (255, 255, 200))
                padding = 8
                box_height = text_texture.get_height() + padding
                content_width = text_texture.get_width() + symbol_texture.get_width() + 5
                box_width = content_width + padding * 2
                surface = pygame.Surface((box_width, box_height), pygame.SRCALPHA)
                # 80% opacity (204 = 0.8 * 255)
                surface.fill((80, 80, 0, 204), surface.get_rect())
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
            except Exception as e:
                logging.error(f"Error rendering warning: {e}")


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
        # 80% opacity (204 = 0.8 * 255)
        self.surface.fill((20, 20, 20, 204))
        self.surface.blit(
            text_surf,
            ((box_w - text_surf.get_width()) / 2, (box_h - text_surf.get_height()) / 2),
        )

        # --- Center on full ultrawide display ---
        center_x = (self.screen_dim[0] - box_w) / 2

        self.initial_pos = (
            [center_x, int(self.screen_dim[1] * 0.4) - box_h / 2]
            if is_critical_center
            else [center_x, self.screen_dim[1]]
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

