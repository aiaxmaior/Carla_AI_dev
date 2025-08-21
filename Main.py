"""
Project: Q-DRIVE

## About:
CARLA Manual Control Client with Modular Design.
Integrates custom controls, HUD, sensors, and MVD scoring.
Supports dynamic joystick mapping and CARLA server launch.


Author: CARLA-dev, Arjun Joshi (HBSS)
Recent Date: 08.04.2025
Versioning: v0.3.0
"""

import argparse
import logging
import os
import sys
import subprocess
import time
import json
import carla
import sys_task
import math
import re
import Sensors
import DataIngestion
import pandas as pd

from TitleScreen import TitleScreen
from DynamicMonitor import DynamicMonitor
from VehicleLibrary import VehicleLibrary
from World import World
from HUD import HUD, EndScreen
from controls_queue import DualControl
from MVD import MVDFeatureExtractor

try:
    import pygame
    from pygame.locals import K_ESCAPE, K_RETURN, K_KP_ENTER, NOFRAME
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")

# Global variable to hold the CARLA server process if we launch it
carla_server_process = None


# ==============================================================================
# -- Title Screen Function -----------------------------------------------------
# ==============================================================================


def game_loop(args, client, joystick_mappings=None):
    """
    Main simulation loop. Handles a single session from start to end.
    """
    logging.info("GAME LOOP: Initializing new session.")
    # Yeah,
    os.environ["SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS"] = "0"

    pygame.font.init()

    world_obj = None
    session_tick_data = []
    original_settings = None
    carla_world = None

    try:
        # --- MODIFIED: Map Loading Logic ---
        # This now handles both standard maps and custom .xodr maps
        if args.xodr_path:
            if os.path.exists(args.xodr_path):
                with open(args.xodr_path, encoding='utf-8') as od_file:
                    data = od_file.read()
                logging.info(f"Loading map from OpenDRIVE file: {os.path.basename(args.xodr_path)}")
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
            logging.info(f"Loading standard map: {args.map}")
            carla_world = client.load_world(args.map)
        
        original_settings = carla_world.get_settings()
        # --- End of Map Loading Logic ---

        if args.sync:
            settings = carla_world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            carla_world.apply_settings(settings)
            logging.info("SYNCHRONOUS MODE applied successfully")

        # Multi-monitor setup for panoramic view
        single_monitor_width, single_monitor_height = args.width, args.height
        total_width = single_monitor_width * 4
        total_height = single_monitor_height

        logging.info(
            f"Creating 4 monitor wide window for panoramic view: {total_width}x{total_height}"
        )

        pygame.display.set_icon(pygame.image.load("./images/icon.png"))

        display_flags = pygame.HWSURFACE | pygame.DOUBLEBUF | NOFRAME
        display = pygame.display.set_mode(
            (total_width,
             total_height),
            display_flags,
            args.display,
            )

        pygame.display.set_caption("CARLA MVD Demo")
        logging.info(
            f"Pygame Display Mode Set: Borderless Window at {total_width}x{total_height}"
        )

        #persistent_keys = show_title_screen(display, args)
        title= TitleScreen(display,args)
        persistent_keys, chosen_vehicle_id, carla_blueprint = title.show_title_screen()
 
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
            logging.info(f"Applied custom max_rpm of {args.max_rpm} to vehicle.")
        
        # Starting main gameloop. Custom configurations loaded.
        controller = DualControl(world_obj, args, joystick_mappings, persistent_keys)
        joystick_mappings = controller.mapped_controls
        world_obj.finalize_initialization(controller)
        controller.finalize_setup()
        mvd_feature_extractor = MVDFeatureExtractor(args.mvd_config)
        mvd_feature_extractor.reset_scores()

        if args.sync:
            logging.info("Performing stabilization ticks...")
            for _ in range(15):
                carla_world.tick()
            logging.info("Stabilization complete.")

        data_ingestor = DataIngestion.DataIngestion()
        clock = pygame.time.Clock()

        # --- Main Tick Loop for this Session ---
        while True:
            clock.tick(50)
            if world_obj.is_reset:
                if world_obj.should_reset_scores:
                    mvd_feature_extractor.reset_scores()
                    hud.reset()
                    world_obj.should_reset_scores = False
                world_obj.restart()
                controller = DualControl(world_obj, args, joystick_mappings)
                continue

            world_obj.player.show_debug_telemetry(True)
            world_snapshot = carla_world.get_snapshot() if args.sync else None
            if args.sync:
                carla_world.tick()

            if not world_snapshot:
                logging.error("Failed to get world snapshot.")
                continue

            if controller.parse_events(world_obj, clock):
                logging.info("User requested exit.")
                break

            if mvd_feature_extractor._catastrophic_failure_occurred:
                logging.warning("Catastrophic failure detected. Ending session.")
                break
            display_fps = clock.get_fps()
            if world_obj.player and world_obj.player.is_alive:
                controller.process_commands(world_obj.player, args)
                world_obj.tick(clock, controller.updated_hud_information(), controller, display_fps)

                if world_obj.lane_invasion_sensor_instance:
                    world_obj.lane_invasion_sensor_instance.tick()


                # --- Data Gathering for Logging and Scoring ---
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
                hud.update_mvd_scores_for_display(
                    overall_dp_score,
                    standardized_indices["mbi_0_1"],
                    standardized_indices["lmi_0_1"],
                )

                # --- RESTORED: Per-frame data logging ---
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
                #session_tick_data.append(metrics)
                # --- End of logging block ---

            world_obj.render(display)
            pygame.display.flip()

        # --- End Screen Logic ---
        logging.info("Session ended. Presenting end screen.")
        final_overall_scores = mvd_feature_extractor.get_mvd_datalog_metrics()
        end_screen = EndScreen(display, final_overall_scores, hud.panel_fonts)
        action = end_screen.run(persistent_keys)
        return action, joystick_mappings

    except Exception as e:
        logging.critical(f"Critical error in game loop: {e}", exc_info=True)
        return "exit", joystick_mappings
    finally:
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
        logging.info("GAME LOOP: Session cleanup complete.")


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

    args = argparser.parse_args()

    # Get the original state of all monitors before doing anything else
    simulation_resolution = args.res
    monitors = DynamicMonitor(simulation_resolution)
    original_layout = monitors.get_monitor_layout()
    if not original_layout:
        print("Could not detect monitor setup. Exiting.")
        return

    try:
        # ----------------------------------------------------------------------
        # Set the desired layout for the simulation
        # ----------------------------------------------------------------------
        print(f"\nConfiguring monitors for simulation with resolution: {simulation_resolution}")
        # This single function call replaces your old logic
        monitors.arrange_monitors_horizontally(simulation_resolution, original_layout)
        
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

        try:
            args.width, args.height = [int(x) for x in args.res.split("x")]
        except ValueError:
            args.width, args.height = 1920, 1080

        client = None
        global carla_server_process

        if not args.no_launch_carla:
            if not args.carla_root:
                logging.error("CARLA_ROOT path not provided.")
                sys.exit(1)
            carla_exe_path = os.path.join(args.carla_root, "CarlaUE4.sh")

            # --- CORRECTED LAUNCH COMMAND LOGIC ---
            command = [
                carla_exe_path,
                f"-carla-rpc-port={args.port}",
                f"-quality-level={args.quality.capitalize()}",
            ]
            
            # Conditionally add flags only if they are set
            if args.windowed:
                command.append('-windowed') # Add the flag without a value
            else:
                command.append('-RenderOffScreen')
            
            if args.ResX is not None:
                command.append(f"-ResX={args.ResX}")

            if args.ResY is not None:
                command.append(f"-ResY={args.ResY}")
            logging.info(f"Launching CARLA server: {' '.join(command)}")
            carla_server_process = subprocess.Popen(command)
            time.sleep(10)

        max_retries, retry_delay = 10, 5
        for i in range(max_retries):
            try:
                client = carla.Client(args.host, args.port)
                client.set_timeout(3000.0)
                logging.info(f"Successfully connected to CARLA Server {client.get_server_version()}")
                break
            except RuntimeError as e:
                logging.warning(f"Connection failed: {e}. Retrying...")
                if i == max_retries - 1:
                    raise
                time.sleep(retry_delay)

        client.start_recorder(f"session_{time.strftime('%Y%m%d_%H%M%S')}.log", True)

        joystick_mappings = None
        while True:
            logging.info("Simulation running")
            action, new_mappings = game_loop(args, client, joystick_mappings)
            joystick_mappings = new_mappings
            if action == "exit":
                break
            elif action == "restart":
                continue

    except Exception as e:
        logging.critical(f"Unhandled exception in main: {e}", exc_info=True)
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
        logging.info("Main script execution finished.")
if __name__ == "__main__":
    main()
