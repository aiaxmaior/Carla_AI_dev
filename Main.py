"""
CARLA Manual Control Client with Modular Design.
Integrates custom controls, HUD, sensors, and MVD scoring.
Supports dynamic joystick mapping and CARLA server launch.

Subject: CARLA, Fanatec-Csl~series Integration
Author: CARLA-dev, Arjun Joshi (HBSS)
Recent Date: 06.25.2025
Versioning: v0.1.62 : Corrected synchronous ticking logic for CARLA 0.9.12+ and fixed Pygame init order.
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

from World import World
from HUD import HUD, EndScreen
from controls_queue import DualControl
from MVD import MVDFeatureExtractor
import Sensors

try:
    import pygame
    from pygame.locals import K_ESCAPE, K_RETURN, K_KP_ENTER
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")

# Global variable to hold the CARLA server process if we launch it
carla_server_process = None
# ==============================================================================
# -- Title Screen Function -----------------------------------------------------
# ==============================================================================


def show_title_screen(display, args):
    """
    Displays a title/loading screen before the main simulation starts.
    """
    # 1. Load Logo and Scale it
    logo_img = None
    try:
        logo_surface = pygame.image.load("./images/logo-qryde.png").convert_alpha()
        original_size = logo_surface.get_size()
        # Scale logo to be 30% of its original size for the title screen
        scaled_size = (int(original_size[0] * 0.3), int(original_size[1] * 0.3))
        logo_img = pygame.transform.smoothscale(logo_surface, scaled_size)
    except pygame.error as e:
        logging.warning(f"Could not load logo image for title screen: {e}")

    # 2. Font Setup
    font_path = os.path.join(
        args.carla_root,
        "CarlaUE4",
        "Content",
        "Carla",
        "Fonts",
        "tt-supermolot-neue-trl.bd-it.ttf",
    )
    try:
        font_title = pygame.font.Font(font_path, 64)
        font_subtitle = pygame.font.Font(font_path, 32)
        font_credits = pygame.font.Font(font_path, 22)
        font_prompt = pygame.font.Font(font_path, 28)
    except pygame.error:
        logging.warning("Custom title font not found, falling back to default.")
        font_title = pygame.font.Font(None, 80)
        font_subtitle = pygame.font.Font(None, 40)
        font_credits = pygame.font.Font(None, 28)
        font_prompt = pygame.font.Font(None, 36)

    # 3. Define New Colors
    top_color = (44, 62, 80)  # Dark Slate Blue
    bottom_color = (27, 38, 49)  # Very Dark Blue/Charcoal
    title_color = (169, 204, 227)  # Light Steel Blue
    subtitle_color = (189, 195, 199)  # Cool Gray
    prompt_color = (169, 204, 227)  # Light Steel Blue

    # 4. Render Text Surfaces with new colors
    title_surf = font_title.render(
        "Q-Ryde Driving Behavior Simulator", True, title_color
    )
    subtitle_surf = font_subtitle.render(
        "Powered by HBSS Technologies AI", True, subtitle_color
    )
    author_surf = font_credits.render(
        "Author: Arjun Joshi", True, (150, 150, 150)
    )  # Kept original gray for contrast
    prompt_surf = font_prompt.render("Press ENTER to Begin", True, prompt_color)

    # 5. Main Title Screen Loop
    wait_for_key = True
    while wait_for_key:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == K_ESCAPE
            ):
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key in [K_RETURN, K_KP_ENTER]:
                wait_for_key = False

        # Draw Gradient
        screen_height = display.get_height()
        for y in range(screen_height):
            r = top_color[0] + (bottom_color[0] - top_color[0]) * y // screen_height
            g = top_color[1] + (bottom_color[1] - top_color[1]) * y // screen_height
            b = top_color[2] + (bottom_color[2] - top_color[2]) * y // screen_height
            pygame.draw.line(display, (r, g, b), (0, y), (display.get_width(), y))

        # Blit logo
        if logo_img:
            logo_rect = logo_img.get_rect(
                center=(display.get_width() / 2, display.get_height() * 0.3)
            )
            display.blit(logo_img, logo_rect)

        # Blit text
        title_rect = title_surf.get_rect(
            center=(display.get_width() / 2, display.get_height() * 0.55)
        )
        subtitle_rect = subtitle_surf.get_rect(
            center=(display.get_width() / 2, title_rect.bottom + 20)
        )
        author_rect = author_surf.get_rect(
            center=(display.get_width() / 2, subtitle_rect.bottom + 15)
        )

        # Blinking prompt text
        if (pygame.time.get_ticks() // 500) % 2 == 0:
            prompt_rect = prompt_surf.get_rect(
                center=(display.get_width() / 2, display.get_height() * 0.85)
            )
            display.blit(prompt_surf, prompt_rect)

        display.blit(title_surf, title_rect)
        display.blit(subtitle_surf, subtitle_rect)
        display.blit(author_surf, author_rect)

        pygame.display.flip()
        pygame.time.wait(30)


def game_loop(args, client, joystick_mappings=None):
    """
    Main simulation loop. Handles a single session from start to end (until a failure or user ends it).
    Returns a tuple: (chosen_action, updated_mappings) where chosen_action is 'restart' or 'exit'.
    """
    logging.info("GAME LOOP: Initializing new session.")
    os.environ["SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS"] = "0"

    # **FIX:** Initialize Pygame modules here, right before they are needed.
    # This ensures display detection works correctly without creating a premature window.
    pygame.font.init() 

    world_obj = None
    session_tick_data = []

    # Keep track of original world settings to restore them on exit
    original_settings = None
    carla_world = None

    try:
        logging.info(f"Loading map: {args.map}")
        carla_world = client.load_world(args.map)
        original_settings = carla_world.get_settings()

        # Apply synchronous mode settings if specified
        if args.sync:
            settings = carla_world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05  # Important for stable physics
            carla_world.apply_settings(settings)
            logging.info("SYNCHORNOUS MODE applied succesfully")

        # --- Multi-monitor selection logic ---
        num_displays = pygame.display.get_num_displays()
        logging.info(f"Found {num_displays} display(s).")

        target_display_index = 0
        largest_area = 0
        if args.width & args.height:
            target_resolution = (args.width, args.height)

        if num_displays >= 1:
            desktop_sizes = pygame.display.get_desktop_sizes()
            for i, size in enumerate(desktop_sizes):
                logging.info(f"  Display {i}: {size[0]}x{size[1]}")
                area = size[0] * size[1]
                if area > largest_area:
                    largest_area = area
                    target_display_index = i
                    target_resolution = size

        logging.info(
            f"Targeting display {target_display_index} with resolution {target_resolution[0]}x{target_resolution[1]}."
        )
        native_width, native_height = target_resolution

        # --- Pygame Display Initialization ---
        pygame.display.set_icon(pygame.image.load("./images/icon.png"))
        display_flags = pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.FULLSCREEN
        display = pygame.display.set_mode(
            (native_width, native_height), display_flags, display=target_display_index
        )
        pygame.display.set_caption("CARLA MVD Demo")
        logging.info(
            f"Pygame Display Mode Set on Display {target_display_index}: Native Fullscreen: {native_width}x{native_height}"
        )
        show_title_screen(display, args)
        # --- Object Initialization ---
        hud = HUD(native_width, native_height, args)
        hud_fonts = {
            '_font_primary_hud': hud._font_primary_hud,
            '_font_secondary_hud': hud._font_secondary_hud,
            '_font_score_hud': hud._font_score_hud
        }
        world_obj = World(carla_world, hud, args.filter, args.fov, not args.novehicleparams, args)

        if world_obj.player and args.max_rpm > 0:
            physics_control = world_obj.player.get_physics_control()
            physics_control.max_rpm = args.max_rpm
            world_obj.player.apply_physics_control(physics_control)
            logging.info(f"Applied custom max_rpm of {args.max_rpm} to vehicle.")

        # Pass existing mappings on restart to skip the config UI
        controller = DualControl(world_obj, args, joystick_mappings)
        joystick_mappings = controller.mapped_controls # Get mappings back, whether new or existing
        world_obj.finalize_initialization(controller)
        controller.finalize_setup()
        mvd_feature_extractor = MVDFeatureExtractor(args.mvd_config)
        mvd_feature_extractor.reset_scores()

        if args.sync:
            logging.info("Performing stabilization ticks...")
            for _ in range(15):
                carla_world.tick()
            logging.info("Stabilization complete.")

        clock = pygame.time.Clock()

        # --- Main Tick Loop for this Session ---
        while True:
            # Check if world needs to be reset (e.g., user pressed Backspace)
            clock.tick(20)
            if world_obj.is_reset:
                if world_obj.should_reset_scores:
                    logging.info("Full reset requested: Respawning and resetting MVD scores.")
                    mvd_feature_extractor.reset_scores()
                    hud.reset()
                    world_obj.should_reset_scores = False
                else:
                    logging.info("Respawn Requested: Respawning in different location.")

                world_obj.restart()
                # Re-initialize controller for the new player actor
                controller = DualControl(world_obj, args, joystick_mappings)
                continue

            # **FIX:** This block correctly handles synchronous and asynchronous modes.
            world_snapshot = None
            if args.sync:
                # In sync mode, world.tick() returns the frame ID (int).
                frame_id = carla_world.tick()
                # We must then call get_snapshot() to get the full snapshot object.
                world_snapshot = carla_world.get_snapshot()
                # Ensure the snapshot corresponds to the frame we just ticked
                if frame_id != world_snapshot.frame:
                    logging.warning(f"Frame ID mismatch! Tick returned {frame_id}, snapshot is for {world_snapshot.frame}")
            else:
                logging.error("sync failed")
                pass
            #    clock.tick_busy_loop(60) # Limit client FPS in async mode
            #    world_snapshot = carla_world.get_snapshot()

            if not world_snapshot:
                logging.error("Failed to get world snapshot. Server may be lagging.")
                continue
            # **FIX:** Change immediate 'return' to 'break' to allow EndScreen to show.
            if controller.parse_events(world_obj, clock):
                logging.info("User requested exit. Ending session to show score screen.")
                break # User requested exit (e.g., pressed ESC)

            # Check for a simulation-ending event from the scoring module
            if mvd_feature_extractor._catastrophic_failure_occurred:
                logging.warning("Catastrophic failure detected. Ending session.")
                break # Exit the tick loop to show the end screen

            if world_obj.player and world_obj.player.is_alive:
                controller.process_commands(world_obj.player, args)
                idling = controller.updated_hud_information()
                world_obj.tick(clock, idling, controller)

                # Update scores
                velocity = world_obj.player.get_velocity()
                speed_kmh = 3.6 * velocity.length()
                collision_data = world_obj.collision_sensor_instance.get_collision_data_and_reset() if world_obj.collision_sensor_instance else {}
                # lane_data = world_obj.lane_invasion_sensor_instance.get_invasion_data_and_reset() if world_obj.lane_invasion_sensor_instance else {}
                lane_violation_state = world_obj.lane_invasion_sensor_instance.get_violation_state() if world_obj.lane_invasion_sensor_instance else None
                lane_change_state = world_obj.lane_manager.get_lane_change_state() if world_obj.lane_manager else None
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
                hud.update_mvd_scores_for_display(overall_dp_score, standardized_indices["mbi_0_1"], standardized_indices["lmi_0_1"])
                control_datalog = (
                    controller.get_datalog()
                    if hasattr(controller, "get_datalog")
                    else {}
                )
                mvd_datalog = (
                    mvd_feature_extractor.get_datalog_metrics()
                    if hasattr(mvd_feature_extractor, "get_datalog_metrics")
                    else {}
                )

                # Determine derived log values from new states
                is_off_road = (
                    (lane_violation_state == Sensors.LaneViolationState.CRITICAL)
                    if lane_violation_state
                    else False
                )
                crossed_line = (
                    lane_violation_state is not None
                    and lane_violation_state != Sensors.LaneViolationState.NORMAL
                )

                # Log data for this tick
                loop_state_log = {
                    "frame": world_snapshot.frame,
                    "timestamp": world_snapshot.timestamp.elapsed_seconds,
                    "location": str(world_obj.player.get_location()),
                    "speed_kmh": speed_kmh,
                    "overall_dp_score": overall_dp_score,
                    "collided": collision_data.get("collided"),
                    "went_off_road": is_off_road,
                    "crossed_line": crossed_line,
                    "lane_violation_state": (
                        lane_violation_state.name if lane_violation_state else "N/A"
                    ),
                    "lane_change_state": (
                        lane_change_state.name if lane_change_state else "N/A"
                    ),
                    "controls": control_datalog,
                    "mvd_metrics": mvd_datalog,
                }
                loop_state_log.update(collision_data)
                loop_state_log.update(controller.get_datalog())
                loop_state_log.update(mvd_feature_extractor.get_mvd_datalog_metrics())

                session_tick_data.append(loop_state_log)

            # Render the scene
            world_obj.render(display)
            pygame.display.flip()

        # --- End Screen Logic ---
        # This code is reached when the inner 'while True' loop is broken (e.g., catastrophic failure)
        logging.info("Session ended. Presenting end screen.")
        final_overall_score = mvd_feature_extractor.get_overall_mvd_score()
        final_scores_data = {'overall': final_overall_score}

        end_screen = EndScreen(display, final_scores_data, hud_fonts)
        action = end_screen.run() # This will return 'restart' or 'exit'
        return action, joystick_mappings

    except Exception as e:
        logging.critical(f"Critical error in game loop: {e}", exc_info=True)
        # On any catastrophic error, signal to exit the entire application
        return 'exit', joystick_mappings
    finally:
        # This cleanup runs every time a session ends, before a potential restart or full exit
        if session_tick_data:
            try:
                log_dir = "./Session_logs/"
                if not os.path.exists(log_dir): os.makedirs(log_dir)
                log_filename = f"mvd_session_log_{time.strftime('%Y%m%d_%H%M%S')}.json"
                with open(os.path.join(log_dir, log_filename), "w") as f:
                    json.dump(session_tick_data, f, indent=2)
                logging.info(f"MVD session data written to {os.path.join(log_dir, log_filename)}")
            except Exception as e:
                logging.error(f"Error writing MVD session log: {e}")

        # Destroy actors and restore world settings
        if world_obj is not None:
            world_obj.destroy_all_actors()
        if carla_world is not None and original_settings is not None:
            logging.info("Restoring original world settings.")
            carla_world.apply_settings(original_settings)
        logging.info("GAME LOOP: Session cleanup complete.")


def main():
    # --- Argument Parsing ---
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("-v", "--verbose", action="store_true", dest="debug", help="print debug information")
    argparser.add_argument("--host", metavar="H", default="localhost", help="IP of the host server")
    argparser.add_argument("-p", "--port", metavar="P", default=2000, type=int, help="TCP port to listen to")
    argparser.add_argument("-a", "--autopilot", action="store_true", help="enable autopilot")
    argparser.add_argument("--res", metavar="WIDTHxHEIGHT", default="1920x1080", help="rendering resolution")
    argparser.add_argument("--filter", metavar="PATTERN", default="vehicle.mercedes.sprinter", help="actor filter")
    argparser.add_argument("--steer-deadzone", default=0.01, type=float, help="steering deadzone")
    argparser.add_argument("--steer-linearity", default=0.75, type=float, help="steering linearity")
    argparser.add_argument("--pedal-deadzone", default=0.02, type=float, help="pedal deadzone")
    argparser.add_argument("--carla-root",metavar="PATH",default=os.environ.get("CARLA_ROOT", ""),help="Path to CARLA installation",)
    argparser.add_argument("--no-launch-carla", action="store_true", help="Do not launch CARLA server.")
    argparser.add_argument("--fov", default=120.0, type=float, help="Horizontal field of view")
    argparser.add_argument("--sync", action="store_true", help="Enable synchronous mode")
    argparser.add_argument("--num-vehicles", default=20, type=int, help="Number of traffic vehicles")
    argparser.add_argument("--num-pedestrians", default=20, type=int, help="Number of pedestrians")
    argparser.add_argument("--max-rpm", default=4000.0, type=float, help="Maximum engine RPM")
    argparser.add_argument("--map", metavar="M", default="Town10HD", help="Map")
    argparser.add_argument("--noackermann", action="store_true", help="Disable Ackermann Steering Physics")
    argparser.add_argument("--novehicleparams", action="store_true", help="Disable custom vehicle physics")
    argparser.add_argument("--mvd-config",metavar="FILE", default="./penalty_config/default_penalties.json", help="Path to the JSON file with penalty configurations.",
)

    args = argparser.parse_args()

    # --- Logging Setup ---
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format="%(levelname)s: %(message)s", level=log_level)
    logging.info(f"CARLA Root Path: {args.carla_root}")

    # **FIX:** Initialize Pygame's main module once at the beginning.
    # The font and display modules will be initialized later, inside the loop.
    pygame.init()

    try:
        args.width, args.height = [int(x) for x in args.res.split("x")]
    except ValueError:
        logging.warning(f"Invalid resolution format: {args.res}. Defaulting to 1920x1080.")
        args.width, args.height = 1920, 1080

    # --- Main Application Execution ---
    client = None
    global carla_server_process

    try:
        if not args.no_launch_carla:
            if not args.carla_root:
                logging.error("CARLA_ROOT path not provided.")
                sys.exit(1)
            carla_exe_path = os.path.join(args.carla_root, "CarlaUE4.sh")
            command = [
                carla_exe_path,
                f"-carla-rpc-port={args.port}",
                "-quality-level=Epic",
                "-RenderOffScreen",
            ]
            logging.info(f"Launching CARLA server: {' '.join(command)}")
            carla_server_process = subprocess.Popen(command)
            logging.info("Waiting for CARLA server to initialize...")
            time.sleep(10)

        # --- Connect to CARLA Client ---
        max_retries = 10
        retry_delay = 5
        for i in range(max_retries):
            try:
                logging.info(f"Attempting to connect to CARLA... (Attempt {i + 1}/{max_retries})")
                client = carla.Client(args.host, args.port)
                client.set_timeout(20.0)
                server_version = client.get_server_version()
                logging.info(f"Successfully connected to CARLA Server {server_version}")
                break
            except RuntimeError as e:
                logging.warning(f"Connection failed: {e}. Retrying in {retry_delay} seconds...")
                if i < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    logging.critical("Could not connect to CARLA server after multiple retries. Exiting.")
                    raise

        # --- Start CARLA Recorder ---
        log_dir = "./Carla_recorder_logs/"
        if not os.path.exists(log_dir): os.makedirs(log_dir)
        recorder_filename = os.path.join(log_dir, f"session_{time.strftime('%Y%m%d_%H%M%S')}.log")
        logging.info(f"Starting CARLA recorder, saving to: {recorder_filename}")
        client.start_recorder(recorder_filename, True)

        # --- Session Management Loop ---
        joystick_mappings = None # Start with no mappings
        while True:
            action, new_mappings = game_loop(args, client, joystick_mappings)
            joystick_mappings = new_mappings # Persist mappings for next loop

            if action == 'exit':
                logging.info("Exit action received. Shutting down application.")
                break
            elif action == 'restart':
                logging.info("Restart action received. Reloading simulation.")
                continue

    except KeyboardInterrupt:
        print("\nCancelled by user. Exiting...")
    except Exception as e:
        logging.critical(f"Unhandled exception in main: {e}", exc_info=True)
    finally:
        # --- GUARANTEED SHUTDOWN SEQUENCE ---
        if client:
            logging.info("Stopping CARLA recorder...")
            client.stop_recorder()

        # Gracefully terminate the CARLA server process we started
        if carla_server_process:
            logging.info("Terminating managed CARLA server process...")
            sys_task.terminate_popen_process_gracefully(carla_server_process, "CARLA Server")

        # Final failsafe to kill any stray CARLA processes
        logging.info("Running final process cleanup to kill any stray engine instances...")
        procs = sys_task.get_running_processes()
        sys_task.sig_kill_engine(procs)

        # Quit Pygame properly to release display resources
        pygame.quit()
        logging.info("Main script execution finished.")
        print("Script finished.")


if __name__ == "__main__":
    main()
