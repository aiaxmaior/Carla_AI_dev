# ============================================================================
# PERF CHECK (file-level):
# ============================================================================
# [X] | Role: Joystick configuration UI (NOT in hot path - init/menu only)
# [ ] | Hot-path functions: None (runs at startup or when remapping controls)
# [X] |- Heavy allocs in hot path? N/A - not in hot path
# [X] |- pandas/pyarrow/json/disk/net in hot path? Disk I/O at config save only
# [X] | Graphics here? YES - pygame rendering (gradient fill, text, UI)
# [ ] | Data produced (tick schema?): None (config JSON only)
# [X] | Storage (Parquet/Arrow/CSV/none): JSON (joystick_mappings.json)
# [ ] | Queue/buffer used?: No
# [ ] | Session-aware? No
# [ ] | Debug-only heavy features?: None
# Top 3 perf risks:
# 1. [PERF_OK] NOT in hot path - only runs during controller config
# 2. [PERF_OK] Gradient fill loop (L210) acceptable - not in game loop
# 3. [PERF_OK] Font rendering acceptable - UI only
# ============================================================================

import pygame
import sys
import os
import json
import time
import logging
import collections
import threading
#from Steering import SteerParams
from Utility.Font.FontIconLibrary import FontLibrary, IconLibrary
from pygame.locals import K_RETURN, K_KP_ENTER, K_ESCAPE, KMOD_CTRL, K_q, K_r
import input_utils
fLib=FontLibrary()
iLib=IconLibrary()


class DynamicMapping:
    """
    Handles the dynamic configuration and mapping of joystick controls.
    """

    def __init__(self, args, display, world, joysticks, joystick_capabilities, map_keys = None, display_height = 1080):
        
        self._args = args
        self._display = display
        self._H=display_height
        self.world = world
        self._joysticks = joysticks
        self._joystick_capabilities = joystick_capabilities
        self.mapped_controls = {}
        self.JOYSTICK_MAPPING_FILE = "./configs/joystick_mappings/joystick_mappings.json"
        self._map_keys = map_keys
        logging.debug(f"Dynamic Mapping init, {self._map_keys}")
        self._H
        scale_factor = self._H / 1080.0
        panel_fonts = fLib.get_loaded_fonts(
        font="tt-supermolot-neue-trl.bd-it", type="mapping_screen", scale=scale_factor
    )
        self.primary_font = panel_fonts['sub']
        pygame.display.get_surface()
        

    # NEW

    def run_configuration(self):
        """
        Start the controller configuration process.
        1) Try loading saved mappings.
        2) If none, run the interactive UI (if HUD/surface available).
        3) If a MOZA (or other) accessory is present, inject defaults for common actions
        unless they are already mapped.
        Always returns a dict (possibly empty).
        """
        # --- Resolve display + font safely ---
        display_surface = None
        primary_font = self.primary_font
        try:
            display_surface = pygame.display.get_surface()
        except Exception:
            pass

        hud = getattr(self.world, "hud", None)


        try:
            if not pygame.font.get_init():
                pygame.font.init()
            if not primary_font:
                primary_font = pygame.font.Font(None, 36)
            else:
                primary_font = pygame.font.Font(None, 36)
        except Exception as e:
            logging.error(f"Could not load default font for mapping: {e}")
            primary_font = pygame.font.Font(None,36)

        # --- 1) Load saved mappings if they exist ---
        try:
            loaded_mappings = self.check_for_saved_mappings(display_surface, primary_font)
        except Exception as e:
            logging.error(f"Failed checking saved mappings: {e}", exc_info=True)
            loaded_mappings = None

        if isinstance(loaded_mappings, dict) and loaded_mappings:
            self.mapped_controls = loaded_mappings
            self._fold_persistent_ui_into_mapped()
            logging.info("Using saved controller mappings.")
            return self.mapped_controls

        # --- 2) Run interactive mapping if UI is available ---
        if display_surface is not None and primary_font is not None:
            logging.info("Starting interactive controller configuration...")
            try:
                self._configure_controls_dynamically(display_surface, primary_font)
                # Ensure accessory defaults are present (no overwrite of user choices)
                try:
                    self._fold_persistent_ui_into_mapped()
                    self._save_mappings_to_file()
                except Exception as e:
                    logging.error(f"Failed to save mappings: {e}", exc_info=True)
                return self.mapped_controls or {}
            except Exception as e:
                logging.error(f"ERROR during dynamic control configuration: {e}", exc_info=True)
                if hud is not None and hasattr(hud, "error"):
                    hud.error(f"Mapping Error: {e}")
                return self.mapped_controls or {}

        # --- 3) No HUD/Surface => cannot run UI; return empty mapping with HUD message ---
        err_msg = "HUD display surface or primary font not available for dynamic control configuration."
        logging.error(err_msg)
        if hud is not None and hasattr(hud, "error"):
            hud.error(err_msg)
        return {}





    def check_for_saved_mappings(self, display_surface, primary_font):
        """
        Checks for a saved mapping file and asks the user if they want to reuse it.
        """
        loaded_mappings = self._load_mappings_from_file()
        if not loaded_mappings:
            return None

        self._display_mapping_message(
            display_surface,
            font_object=primary_font,
            primary_message="Device Mapping:",
            instruction_message="Reuse Mapped Controls?",
            sub_message="Press ENTER to Reuse\nor\nESC to Map Controls Again",
            mapping_message=True,
        )
        timeout_seconds = 10.0
        start_time = time.time()
        pygame.event.clear()  # Clear event queue to avoid stale events
        while time.time() - start_time < timeout_seconds:
            for event in pygame.event.get():
                # Handle Keyboard
                if event.type == pygame.KEYDOWN:
                    if event.key == K_ESCAPE:
                        return None # Don't reuse
                    elif event.key in [K_RETURN, K_KP_ENTER]:
                        return loaded_mappings # Reuse

                # Handle Mapped Joystick Buttons
                elif event.type in (pygame.JOYBUTTONDOWN, pygame.JOYAXISMOTION, pygame.JOYHATMOTION):
                    if self._check_mapped_button(event, 'ESCAPE'):
                        return None # Don't reuse
                    elif self._check_mapped_button(event, 'ENTER'):
                        return loaded_mappings # Reuse

    def _load_mappings_from_file(self):
        """Loads joystick mappings from a JSON file."""
        if os.path.exists(self.JOYSTICK_MAPPING_FILE):
            try:
                with open(self.JOYSTICK_MAPPING_FILE, "r") as f:
                    mappings = json.load(f)
                if isinstance(mappings, dict):
                    return mappings
                else:
                    logging.warning(
                        f"Invalid format in {self.JOYSTICK_MAPPING_FILE}. Expected a dictionary."
                    )
            except json.JSONDecodeError as e:
                logging.warning(
                    f"Error reading {self.JOYSTICK_MAPPING_FILE}: {e}. Remapping required."
                )
            except Exception as e:
                logging.warning(
                    f"Unexpected error loading {self.JOYSTICK_MAPPING_FILE}: {e}"
                )
        return None

    def _save_mappings_to_file(self):
        """Saves the current joystick mappings to a JSON file."""
        try:
            with open(self.JOYSTICK_MAPPING_FILE, "w") as f:
                json.dump(self.mapped_controls, f, indent=4)
        except Exception as e:
            logging.error(f'Joystick Mapping was not Saved:{e}')

    def _fold_persistent_ui_into_mapped(self):
        try:
            if isinstance(self._map_keys, dict):
                for k in ('ENTER', 'ESCAPE'):
                    if k in self._map_keys and self._map_keys[k]:
                        self.mapped_controls.setdefault(k, self._map_keys[k])
        except Exception:
            pass

            logging.info(f"Joystick mappings saved to {self.JOYSTICK_MAPPING_FILE}")
        except Exception as e:
            logging.error(f"Could not save mappings: {e}")

    def _display_mapping_message(
        self,
        args,
        surface,
        font_object,
        primary_message,
        instruction_message="",
        sub_message="",
        detected_value_str="",
        mapping_message=False,
    ):
        """Helper to display messages during dynamic mapping."""
        # --- Gradient Background ---
        # This will correctly cover the entire double-wide window
        self._args = args
        top_color = (44, 62, 80)  # Dark Slate Blue
        bottom_color = (27, 38, 49)  # Very Dark Blue/Charcoal
        screen_height = self._display.get_height()
        for y in range(screen_height):
            r = top_color[0] + (bottom_color[0] - top_color[0]) * y // screen_height
            g = top_color[1] + (bottom_color[1] - top_color[1]) * y // screen_height
            b = top_color[2] + (bottom_color[2] - top_color[2]) * y // screen_height
            pygame.draw.line(surface, (r, g, b), (0, y), (self._display.get_width(), y))

        # --- Font Setup (no changes needed here) ---
        if not hasattr(font_object, "render"):
            try:
                font_object = pygame.font.Font(None, 36)
            except Exception:
                return

        panel_fonts = fLib.get_loaded_fonts(
            font="tt-supermolot-neue-trl.bd-it", type="mapping_screen", scale=self._H/1080
        )

        title_font = panel_fonts['title']
        main_font = self.primary_font
        instr_font = panel_fonts['instructions']
        sub_font = panel_fonts['sub']
        detected_font = panel_fonts['detected']
        # --- Corrected Positioning for Multi-Monitor ---
        center_x = self._display.get_width() // 2


        sizes = [(self._display.get_width(), screen_height)]
        self._num_panels = max(
            1, sizes[0][0] / self._args.width
        )  # estimate from window width
        # assume all panels same height; take width from first (good enough for uniform setup)
        total_w, self._panel_h = sizes[0][0], sizes[0][1]
        self._panel_w = total_w // self._num_panels

        def _scale_cover(img, dst_w, dst_h):
            iw, ih = img.get_size()
            sx, sy = dst_w / iw, dst_h / ih
            scale = max(sx, sy)  # cover (crop as needed), preserves aspect
            sw, sh = int(iw * scale), int(ih * scale)
            scaled = pygame.transform.smoothscale(img, (sw, sh))
            # center inside the dst panel
            surf = pygame.Surface((dst_w, dst_h)).convert()
            surf.blit(scaled, ((dst_w - sw) // 2, (dst_h - sh) // 2))
            return surf

        try:
            base = pygame.image.load("./images/logo_duhd.png").convert_alpha()
            self._splash_img = _scale_cover(base, self._panel_w, self._panel_h)
        except pygame.error as e:
            logging.warning(f"Could not load splash image: {e}")
            self._splash_img = None

        # --- Render and Blit all text elements, centered on the main screen ---
        title_surf = title_font.render("CONTROLLER CONFIGURATION", True, (169, 204, 227))
        surface.blit(title_surf, title_surf.get_rect(center=(center_x, surface.get_height() // 6)))

        text_surface = main_font.render(primary_message, True, (189, 195, 199))
        surface.blit(text_surface, text_surface.get_rect(center=(center_x, surface.get_height() // 3)))

        if instruction_message:
            instr_surf = instr_font.render(instruction_message, True, (169, 204, 227))
            surface.blit(instr_surf, instr_surf.get_rect(center=(center_x, surface.get_height() // 2)))

        if sub_message:
            lines = sub_message.split("\n")
            y_offset = 0
            for line in lines:
                sub_surf = sub_font.render(line, True, (189, 195, 199))
                rect = sub_surf.get_rect(center=(center_x, surface.get_height() * 2 // 3 + y_offset))
                surface.blit(sub_surf, rect)
                y_offset += sub_font.get_height()

        if detected_value_str:
            det_surf = detected_font.render(detected_value_str, True, (100, 255, 100))
            surface.blit(det_surf, det_surf.get_rect(center=(center_x, surface.get_height() * 3 // 4)))
        
        pygame.display.flip()

    def _get_axis_value_confirmed(
        self, surface, font, joy_idx_internal, axis_idx, prompt_main, prompt_instr
    ):
        """
        Prompts user to confirm an axis value.
        Used for pedal released/pressed calibration and steering center.
        """
        timeout_seconds = 15.0
        start_time = time.time()
        pygame.event.clear()

        while time.time() - start_time < timeout_seconds:
            current_raw_val = 0.0
            detected_value_str = f"Joy {joy_idx_internal} Axis {axis_idx}: ERROR"
            if (
                joy_idx_internal < len(self._joysticks)
                and self._joysticks[joy_idx_internal]
            ):
                try:
                    current_raw_val = self._joysticks[joy_idx_internal].get_axis(
                        axis_idx
                    )
                    detected_value_str = (
                        f"Joy {joy_idx_internal} Axis {axis_idx}: {current_raw_val:.2f}"
                    )
                except pygame.error:
                    pass

            self._display_mapping_message(
                surface,
                font,
                prompt_main,
                prompt_instr,
                "Press ENTER to confirm. ESC to re-detect axis.\nPress R to restart entire config.",
                detected_value_str,
            )

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == K_r:
                        return "RESTART_CONFIG"
                    if event.key in [K_RETURN, K_KP_ENTER]:
                        final_val = 0.0
                        if (
                            joy_idx_internal < len(self._joysticks)
                            and self._joysticks[joy_idx_internal]
                        ):
                            try:
                                final_val = self._joysticks[joy_idx_internal].get_axis(
                                    axis_idx
                                )
                            except pygame.error:
                                pass
                        return final_val
                    if event.key == K_ESCAPE:
                        return "REDO_AXIS_DETECTION"
                elif event.type in (pygame.JOYBUTTONDOWN, pygame.JOYAXISMOTION, pygame.JOYHATMOTION):
                    if self._check_mapped_button(event, 'UI_RESTART'):
                        return "RESTART_CONFIG"                    
                    if self._check_mapped_button(event, 'ENTER'):
                        # This logic is copied from the KEYDOWN Enter check
                        final_val = 0.0
                        if (joy_idx_internal < len(self._joysticks) and self._joysticks[joy_idx_internal]):
                            try:
                                final_val = self._joysticks[joy_idx_internal].get_axis(axis_idx)
                            except pygame.error:
                                pass
                        return final_val
                    
                    elif self._check_mapped_button(event, 'LEFT'):
                        return "REDO_AXIS_DETECTION" # Treat mapped Escape as "go back"
            pygame.display.flip()
            pygame.time.wait(10)
        return None

    def _prompt_for_input(
        self,
        surface,
        font,
        action_id,
        input_type_expected,
        action_prompt_name,
        mapping_message=False,
    ):
        """
        Guides the user through mapping a single control (axis, button, or hat).
        """
        prompt_message = f"Map: {action_prompt_name.upper()}"
        sub_instr = "Press ESC to SKIP. Press R to RESTART config."
        detected_value_str = ""
        timeout_seconds = 30.0
        instruction = ""
        is_center_blinker_mapped = False

        if input_type_expected == "axis_steer":
            instruction = (
                "Turn STEERING fully LEFT, then RIGHT, then CENTER. Press ENTER."
            )
        elif input_type_expected == "button":
            instruction = "Press the desired BUTTON."
            _accaps = []
        elif input_type_expected == "axis_pedal":
            self._display_mapping_message(
                surface,
                font,
                prompt_message,
                f"Gently touch/move {action_prompt_name} pedal to identify it.",
                sub_instr,
            )
            active_pedal_joy_idx, active_pedal_axis_idx = None, None
            start_time_pedal_detect = time.time()
            pedal_detect_timeout = 15.0

            initial_axis_values_per_joystick = {}
            for joy_obj in self._joysticks:
                if joy_obj and joy_obj.get_init():
                    try:
                        initial_axis_values_per_joystick[joy_obj.get_instance_id()] = [
                            joy_obj.get_axis(ax_idx)
                            for ax_idx in range(joy_obj.get_numaxes())
                        ]
                    except pygame.error:
                        pass

            AXIS_DETECTION_THRESHOLD = 0.2
            pygame.event.clear()

            while time.time() - start_time_pedal_detect < pedal_detect_timeout:
                self._display_mapping_message(
                    surface,
                    font,
                    prompt_message,
                    f"Gently touch/move {action_prompt_name} pedal to identify it.",
                    sub_instr,
                    detected_value_str,
                )
                # still inside the while loop of _prompt_for_input(...)
                # --- Accessory poll (non-blocking) ---

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.KEYDOWN:
                        if event.key == K_ESCAPE:
                            return None
                        if event.key == K_r:
                            return "RESTART_CONFIG"
                    elif event.type in (pygame.JOYBUTTONDOWN, pygame.JOYAXISMOTION, pygame.JOYHATMOTION):
                        if self._check_mapped_button(event, 'ESCAPE'):
                            return None # Skip this control
                    elif self._check_mapped_button(event, 'UI_RESTART'):
                        return "RESTART_CONFIG"

                    if event.type == pygame.JOYAXISMOTION:
                        joy_id_event = event.instance_id
                        internal_joy_idx = next(
                            (
                                j_cap["index"]
                                for j_cap in self._joystick_capabilities
                                if j_cap["id"] == joy_id_event
                            ),
                            None,
                        )
                        if internal_joy_idx is None:
                            continue

                        axis_idx = event.axis
                        current_val = event.value

                        if (
                            joy_id_event in initial_axis_values_per_joystick
                            and axis_idx
                            < len(initial_axis_values_per_joystick[joy_id_event])
                        ):
                            initial_val_for_this_axis = (
                                initial_axis_values_per_joystick[joy_id_event][axis_idx]
                            )
                            detected_value_str = f"Joy {internal_joy_idx} Axis {axis_idx}: {current_val:.2f}"

                            if (
                                abs(current_val - initial_val_for_this_axis)
                                > AXIS_DETECTION_THRESHOLD
                            ):
                                active_pedal_joy_idx, active_pedal_axis_idx = (
                                    internal_joy_idx,
                                    axis_idx,
                                )
                                break
                if active_pedal_joy_idx is not None:
                    break
                pygame.time.wait(10)

            if active_pedal_joy_idx is None:
                self._display_mapping_message(
                    surface,
                    font,
                    prompt_message,
                    "No pedal movement detected.",
                    "Skipping...",
                    "(Timeout)",
                )
                pygame.time.wait(1500)
                return None

            released_val_result = self._get_axis_value_confirmed(
                surface,
                font,
                active_pedal_joy_idx,
                active_pedal_axis_idx,
                f"Map: {action_prompt_name.upper()}",
                f"RELEASE {action_prompt_name} fully, then press ENTER.",
            )
            if released_val_result == "RESTART_CONFIG":
                return "RESTART_CONFIG"
            if released_val_result == "REDO_AXIS_DETECTION":
                return self._prompt_for_input(
                    surface, font, action_id, input_type_expected, action_prompt_name
                )
            if released_val_result is None:
                return None

            pressed_val_result = self._get_axis_value_confirmed(
                surface,
                font,
                active_pedal_joy_idx,
                active_pedal_axis_idx,
                f"Map: {action_prompt_name.upper()}",
                f"PRESS {action_prompt_name} fully, then press ENTER.",
            )
            if pressed_val_result == "RESTART_CONFIG":
                return "RESTART_CONFIG"
            if pressed_val_result == "REDO_AXIS_DETECTION":
                return self._prompt_for_input(
                    surface, font, action_id, input_type_expected, action_prompt_name
                )
            if pressed_val_result is None:
                return None

            self._display_mapping_message(
                surface,
                font,
                f"{action_prompt_name.upper()} MAPPED!",
                f"Released: {released_val_result:.2f}, Pressed: {pressed_val_result:.2f}",
                "OK!",
            )
            pygame.time.wait(1000)
            return {
                "type": "axis_pedal",
                "joy_idx": active_pedal_joy_idx,
                "id": active_pedal_axis_idx,
                "raw_released_val": released_val_result,
                "raw_pressed_val": pressed_val_result,
            }

        axis_steer_data = {
            "min": 1.0,
            "max": -1.0,
            "center": None,
            "moved_significantly": False,
            "joy_idx": None,
            "axis_idx": None,
        }
        initial_axis_values_per_joystick_steer = {}
        if input_type_expected == "axis_steer":
            for joy_obj in self._joysticks:
                if joy_obj and joy_obj.get_init():
                    try:
                        initial_axis_values_per_joystick_steer[
                            joy_obj.get_instance_id()
                        ] = [
                            joy_obj.get_axis(ax_idx)
                            for ax_idx in range(joy_obj.get_numaxes())
                        ]
                    except pygame.error:
                        pass

        AXIS_ACTIVATION_THRESHOLD = 0.3
        start_time = time.time()
        pygame.event.clear()

        while time.time() - start_time < timeout_seconds:
            self._display_mapping_message(
                surface,
                font,
                prompt_message,
                instruction,
                sub_instr,
                detected_value_str,
            )
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                # --- Handle Keyboard Input ---
                if event.type == pygame.KEYDOWN:
                    if event.key == K_ESCAPE:
                        return None
                    if event.key == K_r:
                        return "RESTART_CONFIG"

                    if event.key in [K_RETURN, K_KP_ENTER]:
                        if (
                            input_type_expected == "axis_steer"
                            and axis_steer_data["moved_significantly"]
                        ):
                            current_joy_idx, current_axis_idx = (
                                axis_steer_data["joy_idx"],
                                axis_steer_data["axis_idx"],
                            )
                            if (
                                current_joy_idx is not None
                                and current_axis_idx is not None
                                and 0 <= current_joy_idx < len(self._joysticks)
                                and self._joysticks[current_joy_idx]
                            ):
                                try:
                                    axis_steer_data["center"] = self._joysticks[
                                        current_joy_idx
                                    ].get_axis(current_axis_idx)
                                except pygame.error:
                                    continue
                                self._display_mapping_message(
                                    surface,
                                    font,
                                    f"{action_prompt_name.upper()} MAPPED!",
                                    f"Min: {axis_steer_data['min']:.2f}, Max: {axis_steer_data['max']:.2f}, Center: {axis_steer_data['center']:.2f}",
                                    "OK!",
                                )
                                pygame.time.wait(1000)
                                return {
                                    "type": "axis_steer",
                                    "joy_idx": current_joy_idx,
                                    "id": current_axis_idx,
                                    "raw_calibrated_min": axis_steer_data["min"],
                                    "raw_calibrated_max": axis_steer_data["max"],
                                    "raw_calibrated_center": axis_steer_data["center"],
                                }

                # --- Handle mapped joystick UI controls (Enter/Escape/Restart) for button/axis/hat ---
                elif event.type in (pygame.JOYBUTTONDOWN, pygame.JOYAXISMOTION, pygame.JOYHATMOTION):
                    if self._check_mapped_button(event, 'UI_RESTART'):
                        return "RESTART_CONFIG"
                    if self._check_mapped_button(event, 'ENTER'):
                        if (input_type_expected == "axis_steer" and axis_steer_data.get("moved_significantly")):
                            current_joy_idx = axis_steer_data.get("joy_idx")
                            current_axis_idx = axis_steer_data.get("axis_idx")
                            if (isinstance(current_joy_idx, int) and isinstance(current_axis_idx, int) and
                                0 <= current_joy_idx < len(self._joysticks) and self._joysticks[current_joy_idx]):
                                try:
                                    center_val = self._joysticks[current_joy_idx].get_axis(current_axis_idx)
                                except pygame.error:
                                    center_val = 0.0
                                axis_steer_data["center"] = center_val
                                self._display_mapping_message(
                                    surface,
                                    font,
                                    f"{action_prompt_name.upper()} MAPPED!",
                                    (f"Min: {axis_steer_data.get('min', 0.0):.2f}, "
                                     f"Max: {axis_steer_data.get('max', 0.0):.2f}, "
                                     f"Center: {axis_steer_data.get('center', 0.0):.2f}"),
                                    "OK!",
                                )
                                pygame.time.wait(1000)
                                return {
                                    "type": "axis_steer",
                                    "joy_idx": current_joy_idx,
                                    "id": current_axis_idx,
                                    "raw_calibrated_min": float(axis_steer_data.get("min", 0.0)),
                                    "raw_calibrated_max": float(axis_steer_data.get("max", 0.0)),
                                    "raw_calibrated_center": float(axis_steer_data.get("center", 0.0)),
                                }
                    if self._check_mapped_button(event, 'ESCAPE'):
                        return None
                
                # --- Handle Mapped Joystick UI Buttons ---
                    elif event.type in (pygame.JOYBUTTONDOWN, pygame.JOYAXISMOTION, pygame.JOYHATMOTION):
                        # Resolve mapped "Enter" once
                        enter_map = (self._map_keys or {}).get('ENTER') or {}
                        enter_joy_id = enter_map.get('joy_idx')
                        enter_btn_id = enter_map.get('id')
                        enter_type_id = enter_map.get('type')

                        # Only proceed if this JOYBUTTONDOWN matches the configured Enter button
                        if (event.type == pygame.JOYBUTTONDOWN and
                            (getattr(event, "instance_id", getattr(event, "joy", None)) == enter_joy_id) and
                            (getattr(event, "button", None) == enter_btn_id)):
                            return {"type": "continue"}

                            # Confirm steer axis only when we've seen meaningful motion
                            if input_type_expected == "axis_steer" and axis_steer_data.get("moved_significantly"):
                                current_joy_idx = axis_steer_data.get("joy_idx")
                                current_axis_idx = axis_steer_data.get("axis_idx")

                                if (isinstance(current_joy_idx, int) and isinstance(current_axis_idx, int) and
                                    0 <= current_joy_idx < len(self._joysticks) and self._joysticks[current_joy_idx]):

                                    # Try to capture the current axis position as 'center'
                                    try:
                                        center_val = self._joysticks[current_joy_idx].get_axis(current_axis_idx)
                                    except pygame.error:
                                        center_val = 0.0  # safe fallback

                                    axis_steer_data["center"] = center_val

                                    self._display_mapping_message(
                                        surface,
                                        font,
                                        f"{action_prompt_name.upper()} MAPPED!",
                                        (
                                            f"Min: {axis_steer_data.get('min', 0.0):.2f}, "
                                            f"Max: {axis_steer_data.get('max', 0.0):.2f}, "
                                            f"Center: {axis_steer_data.get('center', 0.0):.2f}"
                                        ),
                                        "OK!",
                                    )
                                    pygame.time.wait(1000)

                                    return {
                                        "type": "axis_steer",
                                        "joy_idx": current_joy_idx,
                                        "id": current_axis_idx,
                                        "raw_calibrated_min": float(axis_steer_data.get("min", 0.0)),
                                        "raw_calibrated_max": float(axis_steer_data.get("max", 0.0)),
                                        "raw_calibrated_center": float(axis_steer_data.get("center", 0.0)),
                                    }
                    # Check for mapped Restart button
                    elif self._check_mapped_button(event, 'UI_RESTART'):
                        return "RESTART_CONFIG"
                    
                    # Check for mapped Escape button
                    elif self._check_mapped_button(event, 'ESCAPE'):
                        return None               
                # --- CORRECTED LOGIC: Prioritize mapping events over skip events ---

                # 1. Check if the event is the one we are actively trying to map
                if (
                    input_type_expected == "axis_steer"
                    and event.type == pygame.JOYAXISMOTION
                ):
                    joy_id_event = event.instance_id
                    internal_joy_idx = next(
                        (
                            j_cap["index"]
                            for j_cap in self._joystick_capabilities
                            if j_cap["id"] == joy_id_event
                        ),
                        None,
                    )
                    if internal_joy_idx is None:
                        continue
                    axis_idx = event.axis
                    current_val = event.value
                    detected_value_str = (
                        f"Joy {internal_joy_idx} Axis {axis_idx}: {current_val:.2f}"
                    )
                    if not axis_steer_data["moved_significantly"]:
                        if joy_id_event not in initial_axis_values_per_joystick_steer:
                            continue
                        if axis_idx >= len(
                            initial_axis_values_per_joystick_steer[joy_id_event]
                        ):
                            continue
                        initial_val = initial_axis_values_per_joystick_steer[
                            joy_id_event
                        ][axis_idx]
                        if abs(current_val - initial_val) > AXIS_ACTIVATION_THRESHOLD:
                            axis_steer_data.update(
                                {
                                    "moved_significantly": True,
                                    "joy_idx": internal_joy_idx,
                                    "axis_idx": axis_idx,
                                    "min": current_val,
                                    "max": current_val,
                                }
                            )
                            instruction = (
                                "Move through full range, center, then press ENTER."
                            )
                    elif (
                        axis_steer_data["joy_idx"] == internal_joy_idx
                        and axis_steer_data["axis_idx"] == axis_idx
                    ):
                        axis_steer_data["min"] = min(
                            axis_steer_data["min"], current_val
                        )
                        axis_steer_data["max"] = max(
                            axis_steer_data["max"], current_val
                        )
                    pygame.display.flip()

                elif (
                    input_type_expected == "button"
                    and event.type == pygame.JOYBUTTONDOWN
                ):
                    joy_id_event = event.instance_id
                    internal_joy_idx = next(
                        (
                            j_cap["index"]
                            for j_cap in self._joystick_capabilities
                            if j_cap["id"] == joy_id_event
                        ),
                        None,
                    )
                    if internal_joy_idx is None:
                        continue
                    button_idx = event.button
                    self._display_mapping_message(
                        surface,
                        font,
                        f"{action_prompt_name.upper()} MAPPED!",
                        f"Joy {internal_joy_idx}, Button {button_idx}",
                        "OK!",
                    )
                    pygame.time.wait(750)
                    return {
                        "type": "button",
                        "joy_idx": internal_joy_idx,
                        "id": button_idx,
                    }
                
                # 2. If it wasn't a mapping event, check if it was the mapped Skip button
                elif event.type in (pygame.JOYBUTTONDOWN, pygame.JOYAXISMOTION, pygame.JOYHATMOTION):
                    if self._check_mapped_button(event, 'ESCAPE'):
                        return None # Skip with joystick

            pygame.time.wait(10)
        return None

    def _check_mapped_button(self, event, mapping_name):
        """Helper to check if an event matches a mapped button/axis/hat.
        Uses self._map_keys for 'Enter'/'Escape' (persistent UI keys), otherwise self.mapped_controls.
        """
        mapping = (self._map_keys.get(mapping_name, {}) if mapping_name in ['ENTER', 'ESCAPE','UP','DOWN','LEFT','RIGHT']
                else self.mapped_controls.get(mapping_name, {})) or {}
        mtype = mapping.get('type', 'button')
        
        # Get the event's instance ID and internal index
        event_instance_id = getattr(event, 'instance_id', getattr(event, 'joy', None))
        event_internal_idx = None
        
        # Find the internal index for this instance ID
        for cap in self._joystick_capabilities:
            if cap['id'] == event_instance_id:
                event_internal_idx = cap['index']
                break
        
        # Get the mapped joystick identifier
        mapped_joy_idx = mapping.get('joy_idx')
        
        if mapped_joy_idx is not None:
            # Compare using internal index
            if event_internal_idx != mapped_joy_idx:
                return False

        if mtype == 'button':
            return (event.type == pygame.JOYBUTTONDOWN and
                    event.button == mapping.get('id'))
        elif mtype == 'axis':
            if event.type != pygame.JOYAXISMOTION:
                return False
            if event.axis != mapping.get('axis_id'):
                return False
            direction = mapping.get('direction', 1) or 1
            threshold = mapping.get('threshold', 0.6)
            val = getattr(event, 'value', 0.0)
            return (val >= threshold) if direction > 0 else (val <= -threshold)
        elif mtype == 'hat':
            if event.type != pygame.JOYHATMOTION:
                return False
            if event.hat != mapping.get('hat_id'):
                return False
            target = mapping.get('value')
            return tuple(target) == tuple(getattr(event, 'value', (0, 0))) if target is not None else False
        else:
            return False

    def _configure_controls_dynamically(self, surface, font):
            """Main loop to guide user through mapping all necessary controls."""
            actions_to_map = [
                {"id": "STEERING", "type": "axis_steer", "prompt": "Steering"},
                {"id": "THROTTLE", "type": "axis_pedal", "prompt": "Throttle Pedal"},
                {"id": "BRAKE", "type": "axis_pedal", "prompt": "Brake Pedal"},
                {"id": "HELP","type":"button","prompt":"Show Help Menu"},
                {"id": "PARK","type":"button","prompt":"Park Button"},
                {"id": "REVERSE", "type": "button", "prompt": "Reverse Gear Button"},
                {"id": "HANDBRAKE", "type": "button", "prompt": "Handbrake Button"},
                {
                    "id": "TOGGLE_MANUAL_GEAR",
                    "type": "button",
                    "prompt": "Toggle Manual Transmission Button",
                },
                {"id": "GEAR_UP", "type": "button", "prompt": "Gear Up Button (Manual)"},
                {
                    "id": "GEAR_DOWN",
                    "type": "button",
                    "prompt": "Gear Down Button (Manual)",
                },
                {"id": "TOGGLE_VIEW", "type": "button", "prompt": "Toggle Camera view"},
                {"id": "BLINKER_LEFT", "type": "button", "prompt": "Left Blinker"},
                {'id': "BLINKER_LEFT_OFF",'type':'button','prompt':"Reset Blinker to Middle"},
                {"id": "BLINKER_RIGHT", "type": "button", "prompt": "Right Blinker"},
                {'id': "BLINKER_RIGHT_OFF",'type':'button','prompt':"Reset Blinker to Middle"},
                {"id": "HAZARD", "type": "button", "prompt": "Hazard Lights"},
                {"id": "NEXT_WEATHER","type": "button","prompt":"Next Weather Setting"},
                {"id": "NEXT_WEATHER_REVERSE","type": "button","prompt":"Previous Weather Setting"},
            ]
            self._display_mapping_message(
                surface,
                font,
                "Controller Setup",
                "Follow prompts. Press ENTER to begin.",
                "Press ESC to skip entire setup.\nPress R to restart at any time.",
            )
            wait_start = True
            pygame.event.clear()
            while wait_start:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.KEYDOWN:
                        if event.key in [pygame.K_RETURN, pygame.K_KP_ENTER]:
                            wait_start = False
                        if event.key ==pygame.K_ESCAPE:
                            if hasattr(self.world, "hud"):
                                self.world.hud.notification("Config skipped.", 3)
                            return
                    elif event.type == pygame.JOYBUTTONDOWN:
                        if self._check_mapped_button(event, 'ENTER'):
                            logging.critical("ENTER pressed")
                            return # Skip entire setup
                        elif self._check_mapped_button(event, 'ESCAPE'):
                            logging.critical("Enter Pressed")
                            wait_start = False
                        
                pygame.time.wait(10)
            # This outer loop allows restarting the whole process
            while True:
                self.mapped_controls.clear()  # Start fresh

                # Inner loop iterates through each control to map
                for action_cfg in actions_to_map:
                    res = self._prompt_for_input(
                        surface,
                        font,
                        action_cfg["id"],
                        action_cfg["type"],
                        action_cfg["prompt"],
                    )

                    if res == "RESTART_CONFIG":
                        self._display_mapping_message(
                            surface,
                            font,
                            "Configuration Restarting...",
                            "Press ENTER to begin again.",
                        )
                        wait_for_restart = True
                        pygame.event.clear()
                        while wait_for_restart:
                            for event in pygame.event.get():
                                if event.type == pygame.QUIT:
                                    pygame.quit()
                                    sys.exit()
                                if event.type == pygame.KEYDOWN and event.key in [
                                    K_RETURN,
                                    K_KP_ENTER,
                                ]:
                                    wait_for_restart = False

                                elif event.type == pygame.JOYBUTTONDOWN and self._check_mapped_button(event,'ESCAPE'):
                                    logging.critical("ESC pressed")
                                    return # Skip entire setup
                                elif event.type == pygame.JOYBUTTONDOWN and self._check_mapped_button(event, 'ENTER'):
                                    logging.critical("Enter Pressed")
                                    wait_start = False



                        break

                    self.mapped_controls[action_cfg["id"]] = (
                        res if res else {"type": "unmapped"}
                    )

                else:
                    if hasattr(self.world, "hud"):
                        self.world.hud.notification(
                            "Controller mapping complete!", 3, (0, 200, 0)
                        )
                    if surface:
                        surface.fill((0, 0, 0))
                        pygame.display.flip()
                        pygame.time.wait(500)
                    return