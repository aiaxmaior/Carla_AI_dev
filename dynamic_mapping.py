import pygame
import sys
import os
import json
import time
import logging
#from Steering import SteerParams
from pygame.locals import K_RETURN, K_KP_ENTER, K_ESCAPE, KMOD_CTRL, K_q, K_r


class DynamicMapping:
    """
    Handles the dynamic configuration and mapping of joystick controls.
    """

    def __init__(self, world, joysticks, joystick_capabilities, map_keys = None):
        self.world = world
        self._joysticks = joysticks
        self._joystick_capabilities = joystick_capabilities
        self.mapped_controls = {}
        self.JOYSTICK_MAPPING_FILE = "joystick_mappings.json"
        self._map_keys = map_keys

    def run_configuration(self):
        """
        Public method to start the entire joystick configuration process.
        It checks for saved mappings and runs the dynamic configuration if needed.
        """
        display_surface = pygame.display.get_surface()
        primary_font = None
        if hasattr(self.world.hud, "_font_primary_hud"):
            primary_font = self.world.hud._font_primary_hud
        else:
            try:
                primary_font = pygame.font.Font(None, 36)
            except Exception as e:
                logging.error(f"Could not load default font for mapping: {e}")

        loaded_mappings = self.check_for_saved_mappings(display_surface, primary_font)

        if loaded_mappings:
            self.mapped_controls = loaded_mappings
            logging.info("Using saved joystick mappings.")
            return self.mapped_controls

        if display_surface and primary_font:
            logging.info("Starting interactive controller configuration...")
            try:
                self._configure_controls_dynamically(display_surface, primary_font)
                self._save_mappings_to_file()
                return self.mapped_controls
            except Exception as e:
                logging.error(
                    f"ERROR during dynamic control configuration: {e}", exc_info=True
                )
                if hasattr(self.world, "hud"):
                    self.world.hud.error(f"Mapping Error: {e}")
        else:
            err_msg = "HUD display surface or primary font not available for dynamic control configuration."
            logging.error(f"ERROR: {err_msg}")
            if hasattr(self.world, "hud"):
                self.world.hud.error(err_msg)
        return None

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
                elif event.type == pygame.JOYBUTTONDOWN:
                    if (event.instance_id == self._map_keys.get('Escape', {}).get('joy_id') and
                        event.button == self._map_keys.get('Escape', {}).get('button_id')):
                        return None # Don't reuse
                    elif (event.instance_id == self._map_keys.get('Enter', {}).get('joy_id') and
                        event.button == self._map_keys.get('Enter', {}).get('button_id')):
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
            logging.info(f"Joystick mappings saved to {self.JOYSTICK_MAPPING_FILE}")
        except Exception as e:
            logging.error(f"Could not save mappings: {e}")

    def _display_mapping_message(
        self,
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
        top_color = (44, 62, 80)  # Dark Slate Blue
        bottom_color = (27, 38, 49)  # Very Dark Blue/Charcoal
        screen_height = surface.get_height()
        for y in range(screen_height):
            r = top_color[0] + (bottom_color[0] - top_color[0]) * y // screen_height
            g = top_color[1] + (bottom_color[1] - top_color[1]) * y // screen_height
            b = top_color[2] + (bottom_color[2] - top_color[2]) * y // screen_height
            pygame.draw.line(surface, (r, g, b), (0, y), (surface.get_width(), y))

        # --- Font Setup (no changes needed here) ---
        if not hasattr(font_object, "render"):
            try:
                font_object = pygame.font.Font(None, 36)
            except Exception:
                return

        font_path = (
            self.world.hud.custom_font_path
            if hasattr(self.world, "hud") and self.world.hud._use_custom_font
            else None
        )
        title_font = (pygame.font.Font(font_path, 48) if font_path else pygame.font.Font(None, 48))
        main_font = (pygame.font.Font(font_path, 48) if font_path else pygame.font.Font(None, 48))
        instr_font = (pygame.font.Font(font_path, 72) if font_path else pygame.font.Font(None, 72))
        sub_font = (pygame.font.Font(font_path, 36) if font_path else pygame.font.Font(None, 36))
        detected_font = (pygame.font.Font(font_path, 36) if font_path else pygame.font.Font(None, 36))

        # --- Corrected Positioning for Multi-Monitor ---
        main_screen_offset_x = surface.get_width() // 4
        single_screen_width = surface.get_width() // 4
        center_x = main_screen_offset_x + (single_screen_width / 2)

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
                elif event.type == pygame.JOYBUTTONDOWN:
                    if (event.instance_id == self.mapped_controls.get('UI_RESTART', {}).get('joy_id') and
                        event.button == self.mapped_controls.get('UI_RESTART', {}).get('button_id')):
                        return "RESTART_CONFIG"                    
                    if (event.instance_id == self._map_keys.get('Enter', {}).get('joy_id') and
                        event.button == self._map_keys.get('Enter', {}).get('button_id')):
                        # This logic is copied from the KEYDOWN Enter check
                        final_val = 0.0
                        if (joy_idx_internal < len(self._joysticks) and self._joysticks[joy_idx_internal]):
                            try:
                                final_val = self._joysticks[joy_idx_internal].get_axis(axis_idx)
                            except pygame.error:
                                pass
                        return final_val
                    
                    elif (event.instance_id == self._map_keys.get('Escape', {}).get('joy_id') and
                        event.button == self._map_keys.get('Escape', {}).get('button_id')):
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

        if input_type_expected == "axis_steer":
            instruction = (
                "Turn STEERING fully LEFT, then RIGHT, then CENTER. Press ENTER."
            )
        elif input_type_expected == "button":
            instruction = "Press the desired BUTTON."

        if input_type_expected == "axis_pedal":
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
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.KEYDOWN:
                        if event.key == K_ESCAPE:
                            return None
                        if event.key == K_r:
                            return "RESTART_CONFIG"
                    elif event.type == pygame.JOYBUTTONDOWN:
                        if (event.instance_id == self._map_keys.get('Escape', {}).get('joy_id') and
                            event.button == self._map_keys.get('Escape', {}).get('button_id')):
                            return None # Skip this control
                    elif (event.instance_id == self.mapped_controls.get('UI_RESTART', {}).get('joy_id') and
                          event.button == self.mapped_controls.get('UI_RESTART', {}).get('button_id')):
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
                
                # --- Handle Mapped Joystick UI Buttons ---
                elif event.type == pygame.JOYBUTTONDOWN:
                    # Check for mapped Enter button
                    if (event.instance_id == self._map_keys.get('Enter', {}).get('joy_id') and
                        event.button == self._map_keys.get('Enter', {}).get('button_id')):
                        
                        if (input_type_expected == "axis_steer" and axis_steer_data["moved_significantly"]):
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

                    # Check for mapped Restart button
                    elif (event.instance_id == self.mapped_controls.get('UI_RESTART', {}).get('joy_id') and
                          event.button == self.mapped_controls.get('UI_RESTART', {}).get('button_id')):
                        return "RESTART_CONFIG"
                    
                    # Check for mapped Escape button
                    elif (event.instance_id == self._map_keys.get('Escape', {}).get('joy_id') and
                          event.button == self._map_keys.get('Escape', {}).get('button_id')):
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
                elif event.type == pygame.JOYBUTTONDOWN:
                    if (event.instance_id == self._map_keys.get('Escape', {}).get('joy_id') and
                        event.button == self._map_keys.get('Escape', {}).get('button_id')):
                        return None # Skip with joystick

            pygame.time.wait(10)
        return None

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
            {"id": "BLINKER_RIGHT", "type": "button", "prompt": "Right Blinker"},
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
                    if event.key == K_ESCAPE:
                        if hasattr(self.world, "hud"):
                            self.world.hud.notification("Config skipped.", 3)
                        return
                    if event.key in [K_RETURN, K_KP_ENTER]:
                        wait_start = False

                elif event.type == pygame.JOYBUTTONDOWN:
                    if (event.instance_id == self._map_keys.get('Escape', {}).get('joy_id') and
                        event.button == self._map_keys.get('Escape', {}).get('button_id')):
                        return # Skip entire setup
                    elif (event.instance_id == self._map_keys.get('Enter', {}).get('joy_id') and
                        event.button == self._map_keys.get('Enter', {}).get('button_id')):
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