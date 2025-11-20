# ============================================================================
# PERF CHECK (file-level):
# ============================================================================
# [X] | Role: DualControl input handler (LEGACY/DUPLICATE? - see Core/Controls/controls_queue.py)
# [X] | Hot-path functions: parse_events(), process_commands() - EVERY FRAME
# [X] |- Heavy allocs in hot path? Moderate - command queue, dict creation per frame
# [ ] |- pandas/pyarrow/json/disk/net in hot path? No
# [ ] | Graphics here? No
# [X] | Data produced (tick schema?): Control state dict, datalog dict
# [ ] | Storage (Parquet/Arrow/CSV/none): None (consumed by DataIngestion)
# [X] | Queue/buffer used?: YES - command queue (deque)
# [ ] | Session-aware? No
# [ ] | Debug-only heavy features?: logging.info() on L750 every frame
# Top 3 perf risks:
# 1. [PERF_HOT] parse_events() + process_commands() run EVERY FRAME
# 2. [PERF_SPLIT] NOTE: This appears to be DUPLICATE/LEGACY version
# 3. [PERF_OK] Command queue pattern good - decouples parsing from execution
# NOTE: Check if this file is still used - may be superseded by Core/Controls/controls_queue.py
# ============================================================================

import collections
import logging
import math

import carla
import pygame

from dynamic_mapping import DynamicMapping

try:
    import pygame

    #  Keyboard (pygame.locals): Keep from old script for HUD
    from pygame.locals import (
        K_BACKQUOTE,
        K_BACKSPACE,
        K_COMMA,
#        K_DOWN,
        K_ESCAPE,
#        K_LEFT,
        K_PERIOD,
#        K_RETURN,
#        K_RIGHT,
#        K_SLASH,
        K_SPACE,
        K_TAB,
        K_UP,
        KMOD_CTRL,
        KMOD_SHIFT,
        K_a,
        K_c,
#        K_d,
        K_h,
        K_m,
        K_p,
        K_q,
#        K_s,
#        K_w,
        K_z,
        K_x,
        K_n
    )
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")

STEER_DEADZONE_DEFAULT = 0.05
STEER_LINEARITY_DEFAULT = 0.75
PEDAL_DEADZONE_DEFAULT = 0.05


class DualControl(object):
    """
    Handles input from keyboard and joysticks by decoupling event parsing
    from control logic execution using a command queue.
    """

    def __init__(self, world_instance, args, existing_mappings=None, map_keys=None):
        self.world = world_instance
        self.args = args
        self._autopilot_enabled = self.args.autopilot
        self.last_ackermann = None
        self._hud = None  # set by World during finalize
        self.args_steer_degrees = args.steer
        # --- Queue and State Variables ---
        self._command_queue = collections.deque()
        self._current_steer = 0.0
        self._clamped_steer = 0.0
        self._current_throttle = 0.0
        self._current_brake = 0.0
        self._handbrake_state = True
        self._driver_view = True
        self._is_resetting = False
        self._blinker_state = 0
        self._idling_nobrake = None
        self._collision_lockout_active = False
        self._park_engaged = False
        self.data_log = {}
        self._map_keys = map_keys


        # CARLA control objects
        self._ackermann_disabled = self.args.noackermann
        self._ackermann_control = carla.VehicleAckermannControl()
        self._control = carla.VehicleControl()

        # --- MODIFIED: Player-specific setup is moved to finalize_setup() ---
        # This prevents the crash on initialization.

        # --- Joystick Initialization ---
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        self._joysticks = []
        self._joystick_capabilities = []
        if joystick_count == 0:
            self._joysticks_present = False
        else:
            self._joysticks_present = True
            logging.info(f"Found {joystick_count} joystick(s).")
            for i in range(joystick_count):
                joystick = pygame.joystick.Joystick(i)
                joystick.init()
                self._joysticks.append(joystick)
                caps = {
                    "index": i,
                    "id": joystick.get_instance_id(),
                    "name": joystick.get_name(),
                    "guid": joystick.get_guid(),
                    "num_axes": joystick.get_numaxes(),
                    "num_buttons": joystick.get_numbuttons(),
                    "num_hats": joystick.get_numhats(),
                }
                self._joystick_capabilities.append(caps)
                logging.info(
                    f"  Joystick {caps['index']} (ID: {caps['id']}): {caps['name']}"
                )

        # --- Mapping Initialization ---
        self.mapped_controls = {}
        self._steer_joy_obj, self._steer_axis_idx = None, None
        self._steer_axis_props = {
            "deadzone": self.args.steer_deadzone,
            "linearity": self.args.steer_linearity,
        }
        self._throttle_joy_obj, self._throttle_axis_idx = None, None
        self._throttle_axis_props = {"deadzone": self.args.pedal_deadzone}
        self._brake_joy_obj, self._brake_axis_idx = None, None
        self._brake_axis_props = {"deadzone": self.args.pedal_deadzone}
        button_attrs = [
            "_ui_enter_joy_idx",
            "_ui_enter_button_idx",
            "_ui_escape_joy_idx",
            "_ui_escape_button_idx",
            "_reverse_button_joy_idx",
            "_reverse_button_idx",
            "_handbrake_button_joy_idx",
            "_handbrake_button_idx",
            "_toggle_manual_button_joy_idx",
            "_toggle_manual_button_idx",
            "_gear_up_button_joy_idx",
            "_gear_up_button_idx",
            "_gear_down_button_joy_idx",
            "_gear_down_button_idx",
            "_toggle_camera_view_joy_idx",
            "_toggle_camera_view_button_idx",
            "_blinker_left_button_joy_idx",
            "_blinker_left_button_idx",
            "_blinker_right_button_joy_idx",
            "_blinker_right_button_idx",
            "_hazard_button_joy_idx",
            "_hazard_button_idx",
            "_cycle_weather_joy_idx",
            "_cycle_weather_button_idx",
            "_cycle_weather_reverse_joy_idx",
            "_cycle_weather_reverse_button_idx"
            "_help_joy_idx",
            "_help_button_idx",
            "_park_joy_idx",
            "_park_btn_idx",

        ]
        for attr in button_attrs:
            setattr(self, attr, None)
        if self._map_keys:
            # Map the Enter button
            enter_map = self._map_keys.get("Enter")
            if enter_map and isinstance(enter_map, dict):
                # This is where you find the joystick's internal index
                for caps in self._joystick_capabilities:
                    if caps['id'] == enter_map.get("joy_id"):
                        self._ui_enter_joy_idx = caps['index']
                        self._ui_enter_button_idx = enter_map.get("button_id")
                        break

            # Map the Escape button
            escape_map = self._map_keys.get("Escape")
            if escape_map and isinstance(escape_map, dict):
                # Find the joystick's internal index
                for caps in self._joystick_capabilities:
                    if caps['id'] == escape_map.get("joy_id"):
                        self._ui_escape_joy_idx = caps['index']
                        self._ui_escape_button_idx = escape_map.get("button_id")
                        break
        if existing_mappings:
            self.mapped_controls = existing_mappings
            self._apply_loaded_mappings()
        elif self._joysticks_present:
            mapper = DynamicMapping(
                self.world, self._joysticks, self._joystick_capabilities, self._map_keys
            )
            self.mapped_controls = mapper.run_configuration()
            if self.mapped_controls:
                self._apply_loaded_mappings()
            else:
                logging.warning("No joystick mappings loaded or configured.")
        else:
            logging.info("No joysticks. Keyboard only.")


    def populate_datalog(self, raw_inputs, is_joystick_mode):
        """Helper method to fill the data_log dictionary."""
        self.data_log["input_mode"] = "joystick" if is_joystick_mode else "keyboard"
        self.data_log["raw_inputs"] = {
            "steer": raw_inputs.get("steer"),
            "clamped_steer": self._clamped_steer,
            "throttle": raw_inputs.get("throttle"),
            "brake": raw_inputs.get("brake"),
        }
        self.data_log["normalized_outputs"] = {
            "steer": self._current_steer,
            "clamped_steer": self._clamped_steer,
            "throttle": self._current_throttle,
            "brake": self._current_brake,
        }
        self.data_log["vehicle_state"] = {
            "handbrake": self._control.hand_brake,
            "reverse": self._control.reverse,
            "manual_gear_shift": self._control.manual_gear_shift,
            "gear": self._control.gear,
            "blinker_state": self._blinker_state,
            "autopilot": self._autopilot_enabled,
        }
        self.data_log["ackermann_targets"] = {
            "enabled": not self._ackermann_disabled,
            "target_steer": self._ackermann_control.steer,
            "target_speed_ms": self._ackermann_control.speed,
            "target_accel_ms2": self._ackermann_control.acceleration,
            "target_jerk_ms3": self._ackermann_control.jerk,
        }
        self.data_log["static_params"] = {
            "steer_deadzone": self.args.steer_deadzone,
            "steer_linearity": self.args.steer_linearity,
            "pedal_deadzone": self.args.pedal_deadzone,
        }

    def get_datalog(self):
        """Public method to retrieve the comprehensive data log for the current tick."""
        return self.data_log

    def finalize_setup(self):
        """
        --- NEW METHOD ---
        Initializes physics settings on the player vehicle after it has been spawned.
        This must be called from Main.py after the World has been finalized.
        """
        if self.world and self.world.player:
            # Apply Ackermann PID settings
            ackermann_settings = carla.AckermannControllerSettings()
            ackermann_settings.speed_kp = 0.6
            ackermann_settings.speed_ki = 0
            ackermann_settings.speed_kd = 0.2
            ackermann_settings.accel_kp = 0.6
            ackermann_settings.accel_ki = 0
            ackermann_settings.accel_kd = 0.2

            self.world.player.set_simulate_physics(True)
            self.world.player.apply_ackermann_controller_settings(ackermann_settings)
            self.world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)
        else:
            logging.error("DualControl.finalize_setup(): world.player not available.")

    def register_collision(self):
        """Called from Main.py to engage the post-collision lockout."""
        if not self._collision_lockout_active:
            self._collision_lockout_active = True
            self._control.hand_brake = True
            self.world.hud.warning_manager.add_warning("handbrake", "HANDBRAKE ON")

    def _apply_loaded_mappings(self):
        """Applies loaded or configured joystick mappings to internal attributes."""
        logging.info("MAPPER: Applying loaded/configured mappings...")
        # (This method remains unchanged)
        # Steering Axis Mapping
        steer_map = self.mapped_controls.get("STEERING")
        if (
            steer_map
            and steer_map["type"] == "axis_steer"
            and steer_map.get("joy_idx") is not None
        ):
            try:
                joy_index = steer_map["joy_idx"]
                if 0 <= joy_index < len(self._joysticks):
                    self._steer_joy_obj = self._joysticks[joy_index]
                    self._steer_axis_idx = steer_map["id"]
                    self._steer_axis_props.update(steer_map)
                    self._steer_axis_props["deadzone"] = self.args.steer_deadzone
                    self._steer_axis_props["linearity"] = self.args.steer_linearity
                else:
                    self._steer_joy_obj = None
            except Exception as e:
                logging.error(f"  ERROR applying steering map: {e}")
                self._steer_joy_obj = None

        # Throttle Pedal Mapping
        throttle_map = self.mapped_controls.get("THROTTLE")
        if (
            throttle_map
            and throttle_map["type"] == "axis_pedal"
            and throttle_map.get("joy_idx") is not None
        ):
            try:
                joy_index = throttle_map["joy_idx"]
                if 0 <= joy_index < len(self._joysticks):
                    self._throttle_joy_obj = self._joysticks[joy_index]
                    self._throttle_axis_idx = throttle_map["id"]
                    self._throttle_axis_props.update(throttle_map)
                    self._throttle_axis_props["deadzone"] = self.args.pedal_deadzone
                else:
                    self._throttle_joy_obj = None
            except Exception as e:
                logging.error(f"  ERROR applying throttle map: {e}")
                self._throttle_joy_obj = None

        # Brake Pedal Mapping
        brake_map = self.mapped_controls.get("BRAKE")
        if (
            brake_map
            and brake_map["type"] == "axis_pedal"
            and brake_map.get("joy_idx") is not None
        ):
            try:
                joy_index = brake_map["joy_idx"]
                if 0 <= joy_index < len(self._joysticks):
                    self._brake_joy_obj = self._joysticks[joy_index]
                    self._brake_axis_idx = brake_map["id"]
                    self._brake_axis_props.update(brake_map)
                    self._brake_axis_props["deadzone"] = self.args.pedal_deadzone
                else:
                    self._brake_joy_obj = None
            except Exception as e:
                logging.error(f"  ERROR applying brake map: {e}")
                self._brake_joy_obj = None
        # ... and so on for the rest of the method
        # Button Mappings
        button_actions = [
            "UI_ENTER",
            "UI_ESCAPE",
            "REVERSE",
            "HANDBRAKE",
            "TOGGLE_MANUAL_GEAR",
            "GEAR_UP",
            "GEAR_DOWN",
            "TOGGLE_VIEW",
            "BLINKER_LEFT",
            "BLINKER_RIGHT",
            "HAZARD",
            "NEXT_WEATHER",
            "NEXT_WEATHER_REVERSE",
            "HELP",
            "PARK"
        ]
        attr_map = {
            "UI_ENTER": ("_ui_enter_joy_idx", "_ui_enter_button_idx"),
            "UI_ESCAPE": ("_ui_escape_joy_idx", "_ui_escape_button_idx"),
            "REVERSE": ("_reverse_button_joy_idx", "_reverse_button_idx"),
            "HANDBRAKE": ("_handbrake_button_joy_idx", "_handbrake_button_idx"),
            "TOGGLE_MANUAL_GEAR": (
                "_toggle_manual_button_joy_idx",
                "_toggle_manual_button_idx",
            ),
            "GEAR_UP": ("_gear_up_button_joy_idx", "_gear_up_button_idx"),
            "GEAR_DOWN": ("_gear_down_button_joy_idx", "_gear_down_button_idx"),
            "TOGGLE_VIEW": (
                "_toggle_camera_view_joy_idx",
                "_toggle_camera_view_button_idx",
            ),
            "BLINKER_LEFT": (
                "_blinker_left_button_joy_idx",
                "_blinker_left_button_idx",
            ),
            "BLINKER_RIGHT": (
                "_blinker_right_button_joy_idx",
                "_blinker_right_button_idx",
            ),
            "HAZARD": (
                "_hazard_button_joy_idx", 
                "_hazard_button_idx"
            ),
            "NEXT_WEATHER": (
                "_cycle_weather_joy_idx",
                "_cycle_weather_button_idx"
            ),
            "NEXT_WEATHER_REVERSE": (
                "_cycle_weather_joy_reverse_idx",
                "_cycle_weather_button_reverse_idx"
            ),
            "HELP": (
                "_help_joy_idx",
                "_help_button_idx"
            ),
            "PARK": (
                "_park_joy_idx",
                "_park_button_idx"
            )
        }
        for action_id in button_actions:
            btn_map = self.mapped_controls.get(action_id)
            if (
                btn_map
                and btn_map["type"] == "button"
                and btn_map.get("joy_idx") is not None
            ):
                try:
                    joy_attr, btn_attr = attr_map[action_id]
                    setattr(self, joy_attr, btn_map["joy_idx"])
                    setattr(self, btn_attr, btn_map["id"])
                except Exception as e:
                    logging.error(f"  ERROR applying button map for {action_id}: {e}")

    def parse_events(self, world, clock):
        """
        PRODUCER: Parses Pygame events and adds them to a command queue.
        """
        joystick_input_active = (
            self._steer_joy_obj or self._throttle_joy_obj or self._brake_joy_obj
        )

        # Always read joystick axes if they are mapped
        if joystick_input_active:
            if self._steer_joy_obj and self._steer_axis_idx is not None:
                steer_val = self._steer_joy_obj.get_axis(self._steer_axis_idx)
                self._command_queue.append(
                    {"type": "axis", "action": "steer", "value": steer_val}
                )

            if self._throttle_joy_obj and self._throttle_axis_idx is not None:
                throttle_val = self._throttle_joy_obj.get_axis(self._throttle_axis_idx)
                self._command_queue.append(
                    {"type": "axis", "action": "throttle", "value": throttle_val}
                )

            if self._brake_joy_obj and self._brake_axis_idx is not None:
                brake_val = self._brake_joy_obj.get_axis(self._brake_axis_idx)
                self._command_queue.append(
                    {"type": "axis", "action": "brake", "value": brake_val}
                )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True

            elif event.type == pygame.JOYBUTTONUP:
                joy_id = event.joy  # In this context, event.joy is the index.
                btn_idx = event.button
                if (
                    joy_id == self._handbrake_button_joy_idx
                    and btn_idx == self._handbrake_button_idx
                ):
                    self._command_queue.append(
                        {"type": "button", "action": "toggle_handbrake"}
                    )
                elif (
                    joy_id == self._reverse_button_joy_idx
                    and btn_idx == self._reverse_button_idx
                ):
                    self._command_queue.append(
                        {"type": "button", "action": "toggle_reverse"}
                    )
                elif (
                    joy_id == self._toggle_manual_button_joy_idx
                    and btn_idx == self._toggle_manual_button_idx
                ):
                    self._command_queue.append(
                        {"type": "button", "action": "toggle_manual"}
                    )
                elif (
                    joy_id == self._gear_up_button_joy_idx
                    and btn_idx == self._gear_up_button_idx
                ):
                    self._command_queue.append({"type": "button", "action": "gear_up"})
                elif (
                    joy_id == self._gear_down_button_joy_idx
                    and btn_idx == self._gear_down_button_idx
                ):
                    self._command_queue.append(
                        {"type": "button", "action": "gear_down"}
                    )
                elif (
                    joy_id == self._toggle_camera_view_joy_idx
                    and btn_idx == self._toggle_camera_view_button_idx
                ):
                    self._command_queue.append(
                        {"type": "button", "action": "toggle_camera"}
                    )
                elif (
                    joy_id == self._blinker_left_button_joy_idx
                    and btn_idx == self._blinker_left_button_idx
                ):
                    self._command_queue.append(
                        {"type": "button", "action": "activate_left_blinker"}
                    )
                elif (
                    joy_id == self._blinker_right_button_joy_idx
                    and btn_idx == self._blinker_right_button_idx
                ):
                    self._command_queue.append(
                        {"type": "button", "action": "activate_right_blinker"}
                    )
                elif (
                    joy_id == self._hazard_button_joy_idx
                    and btn_idx == self._hazard_button_idx
                ):
                    self._command_queue.append(
                        {"type": "button", "action": "toggle_hazard"}
                    )
                
                elif (
                    joy_id == self._cycle_weather_joy_idx
                    and btn_idx == self._cycle_weather_button_idx
                ):
                    self._command_queue.append(
                        {"type": "button", "action": "cycle_weather"}
                    )

                elif (
                    joy_id == self._cycle_weather_reverse_joy_idx
                    and btn_idx == self._cycle_weather_reverse_button_idx
                ):
                    self._command_queue.append(
                        {"type": "button", "action": "cycle_weather_reverse"}
                    )

                elif (
                    joy_id == self._ui_escape_joy_idx
                    and btn_idx == self._ui_escape_button_idx
                ):
                    return True
                
                elif (
                    joy_id == self._help_joy_idx
                    and btn_idx == self._help_button_idx
                ):
                    self._command_queue.append(
                        {"type": "button", "action": "toggle_help"}
                    )
                
                elif (
                    joy_id == self._park_joy_idx
                    and btn_idx == self._park_btn_idx
                ):
                    self._command_queue.append(
                        {"type":"button","action":"toggle_park"}
                    )

            elif event.type == pygame.KEYUP:
                # QUIT
                if (
                    event.key == K_ESCAPE
                    or (event.key == K_c and (pygame.key.get_mods() & KMOD_CTRL))
                    or (event.key == K_q and (pygame.key.get_mods() & KMOD_CTRL))
                ):
                    return True
                # SIMULATION FUNCTIONS
                if event.key == K_q:
                    self._command_queue.append(
                        {"type": "button", "action": "toggle_reverse"}
                    )
                if event.key == K_p:
                    self._command_queue.append(
                        {"type": "button", "action": "toggle_park"}
                    )
                if event.key == K_m:
                    self._command_queue.append(
                        {"type": "button", "action": "toggle_manual"}
                    )
                if event.key == K_SPACE:
                    self._command_queue.append(
                        {"type": "button", "action": "toggle_handbrake"}
                    )
                if event.key == K_COMMA:
                    self._command_queue.append(
                        {"type": "button", "action": "gear_down"}
                    )
                if event.key == K_PERIOD:
                    self._command_queue.append(
                        {"type": "button", "action": "gear_up"}
                    )
                if event.key == K_a:
                    self._command_queue.append(
                        {"type": "button", "action": "toggle_autopilot"}
                    )
                if event.key == K_x:
                    self._command_queue.append(
                        {"type": "button", "action": "activate_right_blinker"}
                    )
                if event.key == K_z and not (pygame.key.get_mods() & KMOD_SHIFT):
                    self._command_queue.append(
                        {"type": "button", "action": "activate_left_blinker"}
                    )

                if event.key == K_z and (pygame.key.get_mods() & KMOD_SHIFT):
                    self._command_queue.append(
                        {"type": "button", "action": "toggle_hazard"}
                    )
                if event.key == K_h:
                    self._command_queue.append(
                        {"type": "button", "action": "toggle_help"}
                    )
                if event.key == K_TAB:
                    self._command_queue.append(
                        {"type": "button", "action": "toggle_camera"}
                    )
                if event.key == K_BACKQUOTE:
                    self._command_queue.append(
                        {"type": "button", "action": "next_sensor"}
                    )
                if event.key == K_n:
                    self._command_queue.append(
                        {"type": "button", "action": "cycle_weather"}
                    )
                if event.key == K_n and (pygame.key.get_mods() & KMOD_SHIFT):
                    self._command_queue.append(
                        {"type": "button", "action": "cycle_weather_reverse"}
                    )
                if event.key == K_BACKSPACE:
                    if event.mod & KMOD_CTRL:
                        self._command_queue.append(
                            {
                                "type": "button",
                                "action": "restart_world_and_reset_scores",
                            }
                        )
                    else:
                        self._command_queue.append(
                            {"type": "button", "action": "restart_world"}
                        )
        return False

    def process_commands(self, player_carla_actor, args):
        """
        CONSUMER: Processes the command queue to update and apply vehicle controls.
        """
        if not player_carla_actor or not player_carla_actor.is_alive:
            return

        raw_axis_values = {"steer": 0.0, "throttle": 0.0, "brake": 0.0}
        self._idling_nobrake = False
        joystick_driving_active = (
            self._steer_joy_obj or self._throttle_joy_obj or self._brake_joy_obj
        )
        while self._command_queue:
            command = self._command_queue.popleft()
            if command["type"] == "axis":
                raw_axis_values[command["action"]] = command["value"]
            elif command["type"] == "button":
                action = command["action"]
                if action == "activate_left_blinker":
                    self._blinker_state = 0 if self._blinker_state == 1 else 1
                elif action == "activate_right_blinker":
                    self._blinker_state = 0 if self._blinker_state == 2 else 2
                elif action == "toggle_hazard":
                    self._blinker_state = 0 if self._blinker_state == 3 else 3
                elif action == "toggle_reverse":
                    if player_carla_actor.get_velocity().length()>5:
                        self.world.hud.warning_manager.add_warning(
                            "reverse","REVERSE_UNAVAILABLE"
                        )
                    else:
                        self._control.reverse = not self._control.reverse
                elif action == 'toggle_park':
                    self._park_engaged = not self._park_engaged
                elif action == "toggle_handbrake":
                    # First, toggle the actual control state
                    self._control.hand_brake = not self._control.hand_brake
                    # Now, update the persistent warning based on the new state
                    if self._control.hand_brake:
                        self.world.hud.warning_manager.add_warning(
                            "handbrake", "HANDBRAKE ON"
                        )
                    else:
                        self.world.hud.warning_manager.remove_warning("handbrake")

                elif action == "toggle_autopilot":
                    self._autopilot_enabled = not self._autopilot_enabled
                    player_carla_actor.set_autopilot(self._autopilot_enabled)
                elif action == "toggle_camera":
                    self.world.camera_manager.toggle_camera()
                elif action == "next_sensor":
                    self.world.camera_manager.next_sensor()
                elif action == "cycle_weather":
                    self.world.next_weather()
                elif action == "cycle_weather_reverse":
                    self.world.next_weather(reverse=True)
                elif action == "restart_world":
                    self.world.is_reset = True
                elif action == "restart_world_and_reset_scores":
                    self.world.is_reset = True
                    self.world.should_reset_scores = True
                elif action == "toggle_manual":
                    self._control.manual_gear_shift = (
                        not self._control.manual_gear_shift
                    )
                    self._control.gear = (
                        player_carla_actor.get_control().gear
                        if self._control.manual_gear_shift
                        else 0
                    )
                    if self._control.manual_gear_shift:
                        self.world.hud.warning_manager.add_warning(
                            "manual_mode", "MANUAL TRANSMISSION"
                        )
                    else:
                        self.world.hud.warning_manager.remove_warning("manual_mode")
                elif self._control.manual_gear_shift:
                    if action == "gear_up":
                        self._control.gear += 1
                    elif action == "gear_down":
                        self._control.gear = max(
                            -1, player_carla_actor.get_control().gear - 1
                        )
                elif action == "toggle_help":
                    if self.world and self.world.hud and self.world.hud.help:
                        self.world.hud.help.toggle()

        self._apply_lights_to_vehicle(player_carla_actor)
        if self._autopilot_enabled:
            return

        if joystick_driving_active:
            self._current_throttle = self._normalize_pedal_value(
                raw_axis_values["throttle"], self._throttle_axis_props
            )
            self._current_brake = self._normalize_pedal_value(
                raw_axis_values["brake"], self._brake_axis_props
            )
            self._current_steer = self._normalize_steer_value(raw_axis_values["steer"])
        
            # 1. Define your desired limit in radians
            desired_steer_limit_rad = 0.3491 # Approx. 20 degrees

            # 2. Get the vehicle's absolute maximum steering angle from its physics
            physics_control = player_carla_actor.get_physics_control()
            # Assumes front wheels have the same max angle
            vehicle_max_steer_deg = physics_control.wheels[0].max_steer_angle


            vehicle_max_steer_rad = math.radians(vehicle_max_steer_deg)

            # 3. Calculate the normalized limit (-1.0 to 1.0)
            # This prevents errors if the desired limit is greater than the car's max
            # Commented out for new logic#
            #if vehicle_max_steer_rad > 0:
            #    normalized_limit = min(1.0, desired_steer_limit_rad / vehicle_max_steer_rad)
            #else:
            #    normalized_limit = 1.0 # Default to full range if max angle is zero

            # 4. Clamp the current steering input to the calculated limit
            normalized_limit, vehicle_max_deg = self._compute_normalized_steer_limit(player_carla_actor, self.args_steer_degrees)
            self._clamped_steer = max(-normalized_limit, min(normalized_limit, self._current_steer))

            logging.info(f"CQ  |Max steering: {vehicle_max_steer_deg} |  CLAMPED_STEERING: {self._clamped_steer} | NORM_LIM: {normalized_limit} | CUR_STR: {self._current_steer}")

        if self._ackermann_disabled:
            self._control.throttle = self._current_throttle
            self._control.brake = self._current_brake
            self._control.steer = self._clamped_steer
            player_carla_actor.apply_control(self._control)
        else:
            # Define vehicle physics parameters
            max_accel, max_brake_decel, max_jerk, max_speed_ms = 4, 10, 6.0, 40.23
            self._ackermann_control.steer = self._clamped_steer
            self._ackermann_control.jerk = max_jerk

            # === Main Control Logic: Pedals / Idle / Coasting ===
            if self._current_brake > 0.05:
                # If lockout is active, pressing the brake clears all lockout states.
                if self._collision_lockout_active:
                    self._collision_lockout_active = False
                    self._control.hand_brake = False
                    self.world.hud.warning_manager.remove_warning("handbrake")

                self._idling_nobrake = False
                self._ackermann_control.speed = 0.0
                self._ackermann_control.acceleration = (
                    self._current_brake * max_brake_decel
                )
            elif self._current_throttle > 0.05:
                # If lockout is active, pressing the throttle clears all lockout states.
                if self._collision_lockout_active:
                    self._collision_lockout_active = False
                    self._control.hand_brake = False
                    self.world.hud.warning_manager.remove_warning("handbrake")

                self._idling_nobrake = False
                self._ackermann_control.speed = max_speed_ms
                self._ackermann_control.acceleration = (
                    self._current_throttle * max_accel
                )
            else:
                # This block handles three states: idle creep, high-speed coasting, and post-collision stop.
                self._idling_nobrake = False
                current_speed_ms = player_carla_actor.get_velocity().length()

                # Condition for normal, slow-speed idle creep
                if (
                    current_speed_ms < 2.0
                    and not self._control.hand_brake
                    and not self._collision_lockout_active
                ):
                    self._idling_nobrake = True
                    self._ackermann_control.speed = 2.5
                    self._ackermann_control.acceleration = 0.5
                else:
                    # This block handles both high-speed coasting and the post-collision stop.
                    if self._collision_lockout_active:
                        # If locked out after a collision, command a hard stop.
                        self._ackermann_control.speed = 0.0
                    else:
                        # Otherwise, allow the car to coast at its current speed.
                        self._ackermann_control.speed = current_speed_ms

                    self._ackermann_control.acceleration = 0.0

            # === Final Overrides: Handbrake and Reverse ===
            if self._park_engaged:
                self._ackermann_control.speed = 0.0
                self._ackermann_control.acceleration = max_brake_decel
                self._idling_nobrake = False
            
            if self._control.hand_brake:
                self._ackermann_control.speed = 0.0
                # Corrected: Use braking deceleration when handbrake is on.
                self._ackermann_control.acceleration = max_brake_decel
                self._idling_nobrake = False

            if self._control.reverse:
                # Apply reverse direction to the determined target speed.
                self._ackermann_control.speed *= -1.0

            player_carla_actor.apply_ackermann_control(self._ackermann_control)
            self.populate_datalog(raw_axis_values, joystick_driving_active)

    def _compute_normalized_steer_limit(self, player, desired_deg=None):
        """
        Returns (normalized_limit in [-1..1], vehicle_max_deg).
        - desired_deg=None -> use full physical range (no software cap)
        - desired_deg=number -> clamp to min(desired_deg, vehicle_max_deg)
        """
        pc = player.get_physics_control()
        steerables = [getattr(w, "max_steer_angle", 0.0) for w in pc.wheels if getattr(w, "max_steer_angle", 0.0) > 0.0]
        vehicle_max_deg = max(steerables) if steerables else 0.0
        logging.info(f'ctrl_fx_args_deg{desired_deg}')
        if vehicle_max_deg <= 0.0:
            return 1.0, 0.0  # fallback: no clamp

        elif not desired_deg:
            effective_deg = vehicle_max_deg 
        else:
            effective_deg=min(desired_deg, vehicle_max_deg)
        logging.info(f'effective_deg: {effective_deg}, desired deg: {desired_deg}, vehicle_max: {vehicle_max_deg}, calc: {effective_deg/vehicle_max_deg}')
        return (effective_deg / vehicle_max_deg), vehicle_max_deg

    def get_blinker_state(self):
        return self._blinker_state

    def _apply_lights_to_vehicle(self, player_carla_actor):
        """Applies the current blinker state to the vehicle actor's light state."""
        if not player_carla_actor or not player_carla_actor.is_alive:
            return
        current_lights = player_carla_actor.get_light_state()
        current_lights &= ~(
            carla.VehicleLightState.LeftBlinker | carla.VehicleLightState.RightBlinker
        )
        if self._blinker_state == 1:
            current_lights |= carla.VehicleLightState.LeftBlinker
        elif self._blinker_state == 2:
            current_lights |= carla.VehicleLightState.RightBlinker
        elif self._blinker_state == 3:
            current_lights |= (
                carla.VehicleLightState.LeftBlinker
                | carla.VehicleLightState.RightBlinker
            )
        player_carla_actor.set_light_state(carla.VehicleLightState(current_lights))

    def updated_hud_information(self):
        return self._idling_nobrake

    def _normalize_pedal_value(self, raw_value, props):
        raw_released = props.get("raw_released_val", 1.0)
        raw_pressed = props.get("raw_pressed_val", -1.0)
        denominator = raw_pressed - raw_released
        if abs(denominator) < 1e-6:
            return 0.0
        normalized = (raw_value - raw_released) / denominator
        normalized = max(0.0, min(normalized, 1.0))
        if normalized < props.get("deadzone", 0.05):
            return 0.0
        return normalized

    def _normalize_steer_value(self, raw_value):
        if raw_value is None:
            return 0.0
        props = self._steer_axis_props
        min_s, max_s, center_s = (
            props.get("raw_calibrated_min", -1.0),
            props.get("raw_calibrated_max", 1.0),
            props.get("raw_calibrated_center", 0.0),
        )
        norm_steer = 0.0
        if raw_value > center_s:
            denominator = max_s - center_s
            if abs(denominator) < 1e-6:
                return 0.0
            norm_steer = (raw_value - center_s) / denominator
        else:
            denominator = center_s - min_s
            if abs(denominator) < 1e-6:
                return 0.0
            norm_steer = (raw_value - center_s) / denominator
        norm_steer = max(-1.0, min(1.0, norm_steer))
        if abs(norm_steer) < props.get("deadzone", 0.05):
            return 0.0
        sign = 1 if norm_steer > 0 else -1
        effective_range = 1.0 - props.get("deadzone", 0.05)
        scaled_input = (abs(norm_steer) - props.get("deadzone", 0.05)) / effective_range
        final_steer = sign * math.pow(scaled_input, props.get("linearity", 0.9))
        if self.args.invert_steer:
            return final_steer * -1.0
        return final_steer
    
    def is_parked(self):
        return self._park_engaged