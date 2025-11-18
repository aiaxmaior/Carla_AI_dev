"""
Project: Q-DRIVE

## About:
CARLA Manual Control Client with Modular Design.
Integrates custom controls, HUD, sensors, and MVD scoring.
Supports dynamic joystick mapping and CARLA server launch.


Author: CARLA-dev, Arjun Joshi (HBSS)
Recent Date: 09.12.2025
Versioning: v0.3.2
"""

# ============================================================================
# PERF CHECK (file-level):
# ============================================================================
# [X] | Role: Main orchestration, game loop, CARLA connection, session mgmt
# [X] | Hot-path functions: game_loop() main tick loop (L331-433)
# [X] |- Heavy allocs in hot path? YES - dict creation every frame (L411-419)
# [X] |- pandas/pyarrow/json/disk/net in hot path? CSV save in finally only
# [ ] | Graphics here? No (delegated to World/HUD)
# [X] | Data produced (tick schema?): metrics dict per frame
# [X] | Storage (Parquet/Arrow/CSV/none): CSV via DataIngestion
# [X] | Queue/buffer used?: DataIngestion handles internally
# [X] | Session-aware? (session_id/tick_index): Yes (frame from snapshot)
# [X] | Debug-only heavy features?: xrandr logging, verbose iLib.ilog
# Top 3 perf risks:
# 1. [PERF_HOT] Logging spam in tick loop (L374 seatbelt every frame, L342, L1005/1008 blinker)
# 2. [PERF_HOT] Dict allocation every frame (L411-419) - should reuse or pool
# 3. [PERF_SPLIT] Heavy imports at module level (carla, pygame, pandas-commented but DataIngestion imports it)
# ============================================================================

import argparse
import logging
import os
import sys
import subprocess
import time
#import json
import carla
import sys_task
#import math
#import re
#import Sensors
import DataIngestion
#import pandas as pd
import PreWelcomeSelect as pws
from TitleScreen import TitleScreen
from Utility.Monitor import DynamicMonitor
#from VehicleLibrary import VehicleLibrary
from World import World
from HUD import HUD
from Core.Controls.controls_queue import DualControl
from Core.Simulation.MVD import MVDFeatureExtractor
from Core.Controls.MozaArduinoVirtualGamepad import HardwareVirtualGamepad    
from evdev import InputDevice, list_devices
from Helpers import EndScreen
from PredictiveManager import PredictiveManager

try:
    import pygame
    from pygame.locals import NOFRAME #K_ESCAPE, K_RETURN, K_KP_ENTER, NOFRAME
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")
from Utility.Font.FontIconLibrary import IconLibrary, FontLibrary
# Global variable to hold the CARLA server process if we launch it
carla_server_process = None
iLib = IconLibrary()
fLib = FontLibrary()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ==============================================================================
# -- Title Screen Function -----------------------------------------------------
# ==============================================================================
dev_keys = {'ENTER': {'type': 'button', 'joy_idx': 0, 'id': 37}, 'UP': {'type': 'button', 'joy_idx': 0, 'id': 47}, 'DOWN': {'type': 'button', 'joy_idx': 0, 'id': 46}, 'LEFT': {'type': 'button', 'joy_idx': 0, 'id': 21}, 'RIGHT': {'type': 'button', 'joy_idx': 0, 'id': 34}, 'ESCAPE': {'type': 'button', 'joy_idx': 0, 'id': 36}}

def ensure_evdev_map(force=False, moza_event=None, arduino_event=None):
    """
    Ensure ./configs/joystick_mappings/input_devices.json exists.
    - If it exists and force=False: print the path and return.
    - Else: run Tools/evdev_autotest.py before we start monitors/accessories.
    """
    root = os.path.dirname(os.path.abspath(__file__))
    cfg_dir = os.path.join(root, "configs", "joystick_mappings")
    cfg_file = os.path.join(cfg_dir, "input_devices.json")

    if os.path.exists(cfg_file) and not force:
        print(f"✅ [evdev] mapping present: {cfg_file}")
        return True

    os.makedirs(cfg_dir, exist_ok=True)
    wizard = os.path.join(root, "Tools", "evdev_autotest.py")
    if not os.path.exists(wizard):
        logging.warning("⚠️evdev_autotest.py not found at Tools/; skipping auto-capture.")
        return False

    # Build the command
    cmd = [sys.executable, wizard]
    if moza_event:
        cmd += ["--moza-event", moza_event]
    else:
        cmd += ["--moza-needle", "Gudsen MOZA Multi-function Stalk"]

    if arduino_event:
        cmd += ["--arduino-event", arduino_event]
    else:
        cmd += ["--arduino-needle", "Arduino"]

    # Make it a bit chatty if you launched with --verbose
    if any(a in sys.argv for a in ("-v", "--verbose")):
        cmd.append("-v")

    print("\n>>> Running input capture (evdev_autotest). "
          "Follow prompts for LEFT/RIGHT/PUSH/PULL and seatbelt…")
    try:
        rc = subprocess.call(cmd)
        if rc != 0:
            logging.warning(f"⚠️evdev_autotest exited with code {rc}")
    except Exception as e:
        logging.exception(f"⚠️Failed to run evdev_autotest: {e}")

    if not os.path.exists(cfg_file):
        logging.warning("⚠️input_devices.json was not created. "
                        "Run Tools/evdev_autotest.py manually if needed.")
        return False

    print(f" ✅🕹️ Captured device mapping → {cfg_file}")
    return True


# [PERF_HOT] Main game loop - runs continuously every tick
def game_loop(args, client, monitors, joystick_mappings=None):
    """
    Main simulation loop. Handles a single session from start to end.
    """
#    logging.info("✅GAME LOOP: Initializing new session.")
    # Yeah,
    os.environ["SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS"] = "0"

    pygame.font.init()

    world_obj = None
    original_settings = None
    carla_world = None
    try:

        """ Display set-up Legacy Code: KEEP FOR BACKUP
        # Multi-monitor setup for panoramic view
        single_monitor_width, single_monitor_height = args.width, args.height
        total_width = single_monitor_width * 4
        total_height = single_monitor_height

        logging.info(
            f"Creating 4 monitor wide window for panoramic view: {total_width}x{total_height}"
        )
        """
#
        try:
            pygame.display.init()
            desktop_sizes = pygame.display.get_desktop_sizes() or []
        except Exception:
            desktop_sizes = []

        use_single = bool(getattr(args, "single", False))
        if not use_single:
            # Fallback: if fewer than 4 displays detected, go single
            use_single = (getattr(args, "screens", 4) < 4) or (len(desktop_sizes) < 4)

        args.layout_mode = "single" if use_single else "quad"
        layout_strategy = monitors.get_layout_strategy()
        total_logical = monitors.get_total_logical_displays()
        if getattr(args, "single", False):
            layout_strategy = "single"
        args.layout_mode = layout_strategy
        print(f"🎯 Using {layout_strategy} layout mode ({total_logical} logical displays)")


        single_monitor_width, single_monitor_height = args.width, args.height
        if args.layout_mode == "single":
            total_width = single_monitor_width
            total_height = single_monitor_height
            logging.info(f" ✅🖥️ Layout: {args.layout_mode}  Window={total_width}x{total_height}")
            iLib.ilog("info", f"Layout: {args.layout_mode}  Window={total_width}x{total_height}", "status_alerts","debug",1 )
        else:
            total_width = single_monitor_width*4
            total_height = single_monitor_height
            logging.info(f" ✅🖥️🖥️🖥️🖥️ Layout: {args.layout_mode}  Window={total_width}x{total_height}")
            iLib.ilog("info", f"Layout: {args.layout_mode}  Window={total_width}x{total_height}", "status_alerts","debug",4)
        
        
            
        pygame.display.set_icon(pygame.image.load("./images/icon.png"))

        display_flags = pygame.HWSURFACE | pygame.DOUBLEBUF | NOFRAME
        #display = pygame.display.set_mode(
        #    (total_width,
         #    total_height),
          #  display_flags,
           # args.display,
            #)
        args.screens = monitors.total_logical_displays 
        display = pygame.display.set_mode(
            (total_width, total_height),
            display_flags,
            display=args.display,
        )

        pygame.display.set_caption("CARLA MVD Demo")

        logging.info(f"width x height = {total_width}x{total_height}")
        xrandr_output = subprocess.check_output(['xrandr', '--query']).decode('utf-8')
#        logging.info(f"🖥️🖥️🖥️🖥️🖥️🖥️xrandr output: {xrandr_output}")

        if args.dev:
            from VehicleLibrary import VehicleLibrary
            
            logging.info("🚀 Developer mode: skipping selection screen, using defaults")
            persistent_keys = dev_keys
            chosen_vehicle_id = "ford_e450_super_duty"
            chosen_map_id = "Town10HD"
            
            # Get the correct blueprint from VehicleLibrary
            vlib = VehicleLibrary()
            try:
                vehicle_info = vlib.get_vehicle("ford_e450_super_duty")
                carla_blueprint = vehicle_info.get("carla_blueprint", "vehicle.ford.ambulance")
            except:
                carla_blueprint = "vehicle.ford.ambulance"  # Fallback
        else:
            # Normal flow
            title = TitleScreen(display, client, args)
            persistent_keys, chosen_vehicle_id, carla_blueprint, chosen_map_id = title.show_title_screen()

        if args.xodr_path:
            if os.path.exists(args.xodr_path):
                with open(args.xodr_path, encoding='utf-8') as od_file:
                    data = od_file.read()
                logging.info(f" 🗺️Loading map from OpenDRIVE file: {os.path.basename(args.xodr_path)}")
                # Parameters for procedural generation
                params = carla.OpendriveGenerationParameters(
                    vertex_distance=15.0,
                    max_road_length=500.0,
                    wall_height=1.0,
                    additional_width=0.6,
                    smooth_junctions=False,
                    enable_mesh_visibility=False
                )
                carla_world = client.generate_opendrive_world(data, params)
            else:
                logging.error(f"XODR file not found at: {args.xodr_path}")
                return "exit", joystick_mappings # Exit if file not found
        else:
            if chosen_map_id:
                logging.info(f" 🗺️Loading chosen map: {chosen_map_id}")
                carla_world = client.load_world(chosen_map_id)
            else:
                logging.info(f"❓No chosen map returned, loading default from cmdline: {args.map}")
                carla_world = client.load_world(args.map)
        
        original_settings = carla_world.get_settings()
        # --- End of Map Loading Logic ---

        if args.sync:
            settings = carla_world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            carla_world.apply_settings(settings)
            logging.info("⏱️ SYNCHRONOUS MODE applied successfully")

        # Object Initialization
        hud = HUD(total_width, total_height, args)
        world_obj = World(
            carla_world, hud, chosen_vehicle_id, carla_blueprint, args.fov, not args.novehicleparams, args
        )

        """
        Vehicle Configurations (below):

        Load Vehicle Configs for Vehicle Dynamics (Steering etc.):
            This is particularly important when 
            - Unreal Assets (the blueprint library) base asset technical values are incorrect
            - A placeholder vehicle is used an requires custom specs to simulate accurate vehicle dynamics

        CONTEXT Note:
            Even if CARLA's package blueprints are close to real-world values, the vehicle still performs
            poorly given the custom mapping of simulation hardware.

        ### See vehicle_library.txt for a list of pre-configured vehicles, steering-vehicle configuration
        ### reasoning and validation
        """
        
        # Load customized vehicle specification configurations from custom library (./configs/vehicle_configs)
        world_obj.load_vehicle_config(chosen_vehicle_id)


        ## (deprecated, will soon be incorporated into vehicle_configs)
        if world_obj.player and args.max_rpm > 0:
            physics_control = world_obj.player.get_physics_control()
            physics_control.max_rpm = args.max_rpm
            world_obj.player.apply_physics_control(physics_control)
            logging.info(f"⚖️ Applied custom max_rpm of {args.max_rpm} to vehicle.")
        
        # Starting main gameloop. Custom configurations loaded.

        hardware_bridge = None

        try:
            hardware_bridge = HardwareVirtualGamepad()
            if hardware_bridge.start():
                iLib.ilog('info',"Hardware Virtual Gamepad bridge started",'alerts','s')
                # pygame detect time
                time.sleep(0.5)
                pygame.joystick.quit()
                pygame.joystick.init()
            else:
                iLib.ilog('warning','Hardware bridge failed to start','alerts','wn')
        except Exception as e:
            iLib.ilog('error',f'Hardware bridge error: {e}','items','js')
            hardware_bridge = None        

        # Instantiate controller
        controller = DualControl(world_obj, args, joystick_mappings, persistent_keys, total_height)

        # After HUD is fully ready
        world_obj.finalize_initialization(controller)


        controller.finalize_setup()
        if getattr(args, "layout_mode","quad") == "single":
            world_obj.enable_single_screen_cameras(window_size=display.get_size())
        mvd_feature_extractor = MVDFeatureExtractor(args.mvd_config)
        mvd_feature_extractor.reset_scores()

        if args.sync:
            logging.info("Performing stabilization ticks...")
            for _ in range(15):
                carla_world.tick()
            logging.info("✅⏱️ Stabilization complete.")

        data_ingestor = DataIngestion.DataIngestion()
        predictive_manager = PredictiveManager(data_ingestor)

        clock = pygame.time.Clock()

        # --- Main Tick Loop for this Session ---
        # [PERF_HOT] Critical path: This loop runs at 50 FPS (every 20ms)
        while True:
            clock.tick(50)

            #Handle reset state
            if world_obj.is_reset:
                if world_obj.should_reset_scores:
                    mvd_feature_extractor.reset_scores()
                    hud.reset()
                    world_obj.should_reset_scores = False
                world_obj.restart()
                controller = DualControl(world_obj, args, joystick_mappings, persistent_keys, total_height)
                logging.info("BLINKER map snapshot: %s", 
                             {k:v for k,v in (controller.mapped_controls or {}).items() if "BLINKER" in k})
                continue

            world_obj.player.show_debug_telemetry(True)
            
            world_snapshot = carla_world.get_snapshot() if args.sync else None
            if args.sync:
                carla_world.tick()

            if not world_snapshot:
                logging.error("❌Failed to get world snapshot.")
                continue

            if controller.parse_events(world_obj, clock):
                logging.info("🏁User requested exit.")
                break

            if mvd_feature_extractor._catastrophic_failure_occurred:
                logging.warning("❌Catastrophic failure detected. Ending session.")
                break
            display_fps = clock.get_fps()
            if world_obj.player and world_obj.player.is_alive:
                controller.process_commands(world_obj.player, args)
                world_obj.tick(clock, controller.updated_hud_information(), controller, display_fps)

                if world_obj.lane_invasion_sensor_instance:
                    world_obj.lane_invasion_sensor_instance.tick()


                # --- Data Gathering for Logging and Scoring ---
                seatbelt_state = hardware_bridge._seatbelt_fastened
                # [PERF_HOT][DEBUG_ONLY] CRITICAL: This logs EVERY FRAME! Should be throttled or debug-only
                # TODO: Move to debug mode or throttle to once per second
                logging.info(f"Seatbelt state: {'ON' if seatbelt_state else 'OFF'}")
                controller._seatbelt_state = seatbelt_state
                velocity = world_obj.player.get_velocity()
                speed_kmh = 3.6 * velocity.length()
                #collision_data = (
                #    world_obj.collision_sensor_instance.get_collision_data_and_reset()
                #)
                collision_data = world_obj.get_collision_data_and_reset()

                if collision_data.get("collided"):
                    controller.register_collision()

                lane_violation_state = (
                    world_obj.lane_invasion_sensor_instance.get_violation_state()
                )
                lane_change_state = world_obj.lane_manager.get_lane_change_state()
                blinker_state = controller.get_blinker_state()

                mvd_feature_extractor.update_scores(
                    collision_data,
                    lane_violation_state,
                    lane_change_state,
                    speed_kmh,
                    world_obj.player,
                    world_obj.world,
                    time.time(),
                    blinker_state,
                )

                

                standardized_indices = mvd_feature_extractor.get_standardized_indices()
                overall_dp_score = mvd_feature_extractor.get_overall_mvd_score()
                
                # --- RESTORED: Per-frame data logging ---
                # [PERF_HOT] Dict allocation every frame - consider object pooling or reuse
                control_datalog = controller.get_datalog()
                mvd_datalog = mvd_feature_extractor.get_mvd_datalog_metrics()
                metrics = {
                    'frame': world_snapshot.frame,
                    'timestamp': world_snapshot.timestamp.elapsed_seconds,
                    'lane_violation':lane_violation_state,
                    'lane_change': lane_change_state,
                    'collision_data': collision_data,
                    'mvd_datalog': mvd_datalog,
                    'controller_datalog':control_datalog,
                    }
                data_ingestor.log_frame(world_obj,metrics)
                if predictive_manager:
                    predictive_manager.tick(world_snapshot.frame)
                    predictive_output = predictive_manager.get_indices()
                    hud.update_predictive_indices(predictive_output)
                hud.update_mvd_scores_for_display(
                    data_ingestor)

                #session_tick_data.append(metrics)
                # --- End of logging block ---

            world_obj.render(display)
            pygame.display.flip()

        # --- End Screen Logic ---
        logging.info("🏁 Session ended. Presenting end screen.")
        final_overall_scores = mvd_feature_extractor.get_mvd_datalog_metrics()
        end_screen = EndScreen(display, final_overall_scores, hud.panel_fonts, data_ingestor)
        action = end_screen.run(persistent_keys)
        return action, joystick_mappings

    except Exception as e:
        logging.critical(f"☠️ Critical error in game loop: {e}", exc_info=True)
        return "exit", joystick_mappings
    finally:
#        if hardware_bridge:
#            hardware_bridge.stop()
            
        if 'data_ingestor' in locals():
            data_ingestor.save_to_csv()
        # --- RESTORED: Write session log to file ---
        #if session_tick_data:
        #    consolidate_and_save_log(session_tick_data)
        # --- End of log writing block ---

        if world_obj:
            world_obj.destroy_all_actors()
        if carla_world and original_settings:
            carla_world.apply_settings(original_settings)
        logging.info("🏁 GAME LOOP: Session cleanup complete.")


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        dest="debug",
        help="print debug information",
    )
    argparser.add_argument(
        "--host", metavar="H", default="localhost", help="IP of the host server"
    )
    argparser.add_argument(
        "-p",
        "--port",
        metavar="P",
        default=2000,
        type=int,
        help="TCP port to listen to",
    )
    argparser.add_argument(
        "-a", "--autopilot", action="store_true", help="enable autopilot"
    )
    argparser.add_argument(
        "--res",
        metavar="WIDTHxHEIGHT",
        default="3840x2160",
        help="rendering resolution of a single monitor",
    )
    argparser.add_argument(
        "--filter",
        metavar="PATTERN",
        default="vehicle.mercedes.sprinter",
        help="actor filter",
    )
    argparser.add_argument(
        "--steer-deadzone", default=0.01, type=float, help="steering deadzone"
    )
    argparser.add_argument(
        "--steer-linearity", default=0.75, type=float, help="steering linearity"
    )
    argparser.add_argument(
        "--pedal-deadzone", default=0.02, type=float, help="pedal deadzone"
    )
    argparser.add_argument(
        "--carla-root",
        metavar="PATH",
        default=os.environ.get("CARLA_ROOT", ""),
        help="Path to CARLA installation",
    )
    argparser.add_argument(
        "--no-launch-carla", action="store_true", help="Do not launch CARLA server."
    )
    argparser.add_argument(
        "--fov", default=85.0, type=float, help="Horizontal field of view"
    )
    argparser.add_argument(
        "--sync", action="store_true", help="Enable synchronous mode"
    )
    argparser.add_argument(
        "--num-vehicles", default=20, type=int, help="Number of traffic vehicles"
    )
    argparser.add_argument(
        "--num-pedestrians", default=30, type=int, help="Number of pedestrians"
    )
    argparser.add_argument(
        "--max-rpm", default=4000.0, type=float, help="Maximum engine RPM"
    )
    argparser.add_argument(
        "--map", metavar="M", default="Town10HD", help="Map")
    argparser.add_argument(
        "--noackermann", action="store_true", help="Disable Ackermann Steering Physics"
    )
    argparser.add_argument(
        "--novehicleparams", action="store_true", help="Disable custom vehicle physics"
    )
    argparser.add_argument(
        "--invert-steer", action="store_true", help="Invert steering input for truck mounts."
    )
    argparser.add_argument(
        "--mvd-config",
        metavar="FILE",
        default="./penalty_config/default_penalties.json",
        help="Path to the JSON file with penalty configurations.",
    )
    argparser.add_argument(
        "--quality",
        metavar="QUALITY",
        default="Epic",
        type = str,
        help="Define CARLA render quality",
    )

    argparser.add_argument(
        "--display",
        metavar="INDEX",
        default=0,
        type=int,
        help="Index of the display to use for the main window (e.g., 0, 1).",
    )

    argparser.add_argument(
        '-x', '--xodr-path',
        metavar='XODR_FILE_PATH',
        help='load a new map with a minimum physical road representation of the provided OpenDRIVE'
    )

    argparser.add_argument(
        "--windowed",
        action='store_true',  # Correct way to handle a boolean flag
        help="Run the CARLA server in a windowed mode.",
    )
    argparser.add_argument(
        "--ResX",
        metavar="X",
        default=None,         # Correct default for an optional integer
        type=int,
        help="Store X resolution for windowed render",
    )
    argparser.add_argument(
        "--ResY",
        metavar="Y",
        default=None,         # Correct default for an optional integer
        type=int,
        help="Set Y resolution for windowed render",
    )

    argparser.add_argument(
        "--screens",
        metavar="SCR",
        default=4,         # Correct default for an optional integer
        type=int,
        help="Define Number of Screens",
    )
    argparser.add_argument(
        "--steer",
        metavar="TRN",
        default= None,         # Correct default for an optional integer
        type=float,
        help="Define steering angle",
    )

    argparser.add_argument(
        "--vision-compare",
        action="store_true",
        help="Show split view: left raw front-left feed, right vision overlay.")
    
    argparser.add_argument(
        "--record-vision-demo",
        metavar="OUT.mp4",
        default=None,
        help="Write the split view video (requires imageio[ffmpeg]).")

    argparser.add_argument(
        "--single",
        action="store_true",
        help="Force single-screen layout (driver view + rear PIP)"
    )
    argparser.add_argument(
    "--title-screen-index",
    type=int,
    default=1,
    help="Which monitor to use for the Title Screen (0-based). Defaults to 0 in single-screen, 1 in quad."
)
    argparser.add_argument(
    "--dev",
        action='store_true',  # Correct way to handle a boolean flag
        help="Skip selection screens and go straight to simulation.",
    )

    args = argparser.parse_args()
    game_loop_reached = False
    prewelcome_options = pws.pre_welcome_select()
    # Get the original state of all monitors before doing anything else
    simulation_resolution = args.res
    monitors = DynamicMonitor(simulation_resolution)
    original_layout = monitors.get_monitor_layout()
    if not original_layout:
        print("⚠️ Could not detect monitor setup. Exiting.")
        return

    try:
        # ----------------------------------------------------------------------
        # Set the desired layout for the simulation
        # ----------------------------------------------------------------------
        print(f"\nConfiguring monitors for simulation with resolution: {simulation_resolution}")
        # This single function call replaces your old logic
        monitors.arrange_monitors_horizontally(simulation_resolution, original_layout)
#        iLib.ilog("warning", f"simulation resolution {simulation_resolution}",'alerts','wn',3)
        # Give the window manager a moment to adjust
        time.sleep(2) 

        # ----------------------------------------------------------------------
        # --- ALL SIMULATION LOGIC IS NOW INSIDE THE TRY BLOCK ---
        # ----------------------------------------------------------------------
        print("\nStarting Pygame application...")
        log_level = logging.DEBUG if args.debug else logging.INFO
        logging.basicConfig(format="%(levelname)s: %(message)s", level=log_level)

        # Set the position for the Pygame window (now reliably at the top-left)
        os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"

        pygame.init()
        pygame.joystick.init()
# --- Pre-Welcome quick setup ---
        """
        def _existing_res_from_args(a):
            # Try args.res "WxH", else fall back to args.width/height, else 1920x1080
            try:
                res_str = getattr(a, "res", f"{getattr(a,'width',1920)}x{getattr(a,'height',1080)}")
                w, h = [int(x) for x in str(res_str).lower().split("x")]
                return (w, h)
            except Exception:
                return (getattr(a, "width", 1920), getattr(a, "height", 1080))
        """

        if prewelcome_options is not None:
            (res_w, res_h), n_peds, n_cars, dev_mode = prewelcome_options
            args.num_vehicles = n_cars
            args.num_pedestrians = n_peds
            args.width, args.height = res_w, res_h
            args.ResX, args.ResY = res_w, res_h
            args.dev = dev_mode

        else:
            # Keep existing CLI values or sensible defaults
            try:
                args.width, args.height = [int(x) for x in getattr(args, "res", "1920x1080").split("x")]
            except Exception:
                args.width, args.height = 1920, 1080
            args.ResX = getattr(args, "ResX", args.width)
            args.ResY = getattr(args, "ResY", args.height)

        client = None
        global carla_server_process

        if not args.no_launch_carla:
            if not args.carla_root:
                logging.error("❓ CARLA_ROOT path not provided.")
                sys.exit(1)
            carla_exe_path = os.path.join(args.carla_root, "CarlaUE4.sh")

            # --- CORRECTED LAUNCH COMMAND LOGIC ---
            command = [
                carla_exe_path,
                f"-carla-rpc-port={args.port}",
                f"-quality-level={args.quality.capitalize()}",
                f"-ResX={int(args.ResX)}",
                f"-ResY={int(args.ResY)}",
            ]

            # Windowing
            if getattr(args, "windowed", False):
                command.append("-windowed")
            else:
                command.append("-RenderOffScreen")

            logging.info(f"Launching CARLA server: {' '.join(map(str, command))}")
            carla_server_process = subprocess.Popen(command)
            time.sleep(10)

        max_retries, retry_delay = 10, 5
        for i in range(max_retries):
            try:
                client = carla.Client(args.host, args.port)
                client.set_timeout(3000.0)
#                logging.info(f"Successfully connected to CARLA Server {client.get_server_version()}")
                break
            except RuntimeError as e:
                logging.warning(f"⚠️Connection failed: {e}. Retrying...")
                if i == max_retries - 1:
                    raise
                time.sleep(retry_delay)

        client.start_recorder(f"session_{time.strftime('%Y%m%d_%H%M%S')}.log", True)

        joystick_mappings = None
        while True:
            if client:
                logging.info(f"Simulation running, CARLA Server {client.get_server_version()}")
            action, new_mappings = game_loop(args, client, monitors, joystick_mappings)
            joystick_mappings = new_mappings
            if action == "exit":
                break
            elif action == "restart":
                continue

    except Exception as e:
        logging.critical(f"☠️Unhandled exception in main: {e}", exc_info=True)
    finally:
        # This block ALWAYS runs, ensuring resolution is restored
        monitors.restore_monitor_layout(original_layout)
        # --- Your existing cleanup logic ---
        if 'client' in locals() and client:
            client.stop_recorder()
        if carla_server_process:
            sys_task.terminate_popen_process_gracefully(carla_server_process, "CARLA Server")
        sys_task.sig_kill_engine(sys_task.get_running_processes())
        pygame.quit()
        logging.info("🏁✅Main script execution finished.")
if __name__ == "__main__":
    main()
