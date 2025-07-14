"""
CARLA Manual Control Client with Modular Design.
Integrates custom controls, HUD, sensors, and MVD scoring.
Supports dynamic joystick mapping and CARLA server launch.

Subject: CARLA, Fanatec-Csl~series Integration
Author: CARLA-dev, Arjun Joshi (HBSS)
Recent Date: 07.03.2025
Versioning: v0.2.4 : Restored comprehensive per-frame session logging.
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
    from pygame.locals import K_ESCAPE, K_RETURN, K_KP_ENTER, NOFRAME
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
    # This function remains unchanged.
    logo_img = None
    try:
        logo_surface = pygame.image.load("./images/logo-qryde.png").convert_alpha()
        original_size = logo_surface.get_size()
        scaled_size = (int(original_size[0] * 0.3), int(original_size[1] * 0.3))
        logo_img = pygame.transform.smoothscale(logo_surface, scaled_size)
    except pygame.error as e:
        logging.warning(f"Could not load logo image for title screen: {e}")

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
        font_prompt = pygame.font.Font(font_path, 60)
    except pygame.error:
        logging.warning("Custom title font not found, falling back to default.")
        font_title = pygame.font.Font(None, 80)
        font_subtitle = pygame.font.Font(None, 40)
        font_credits = pygame.font.Font(None, 28)
        font_prompt = pygame.font.Font(None, 60)

    top_color, bottom_color = (44, 62, 80), (27, 38, 49)
    title_color, subtitle_color, prompt_color = (
        (169, 204, 227),
        (189, 195, 199),
        (169, 204, 227),
    )
    main_screen_offset_x = display.get_width() // 2
    single_screen_width = display.get_width() // 2
    center_x = main_screen_offset_x + (single_screen_width / 2)

    title_surf = font_title.render(
        "Q-Ryde Driving Behavior Simulator", True, title_color
    )
    subtitle_surf = font_subtitle.render(
        "Powered by HBSS Technologies AI", True, subtitle_color
    )
    author_surf = font_credits.render("Author: Arjun Joshi", True, (150, 150, 150))
    prompt_surf = font_prompt.render(
        "Press ENTER to Begin or ESC to Exit", True, prompt_color
    )

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

        screen_height = display.get_height()
        for y in range(screen_height):
            r = top_color[0] + (bottom_color[0] - top_color[0]) * y // screen_height
            g = top_color[1] + (bottom_color[1] - top_color[1]) * y // screen_height
            b = top_color[2] + (bottom_color[2] - top_color[2]) * y // screen_height
            pygame.draw.line(display, (r, g, b), (0, y), (display.get_width(), y))

        if logo_img:
            logo_rect = logo_img.get_rect(center=(center_x, display.get_height() * 0.3))
            display.blit(logo_img, logo_rect)

        # Blit text, centered on the main screen
        title_rect = title_surf.get_rect(center=(center_x, display.get_height() * 0.55))
        subtitle_rect = subtitle_surf.get_rect(center=(center_x, title_rect.bottom + 20))
        author_rect = author_surf.get_rect(center=(center_x, subtitle_rect.bottom + 15))
        
        if (pygame.time.get_ticks() // 1000) % 2 == 0:
            prompt_rect = prompt_surf.get_rect(center=(center_x, display.get_height() * 0.85))
            display.blit(prompt_surf, prompt_rect)


        display.blit(title_surf, title_rect)
        display.blit(subtitle_surf, subtitle_rect)
        display.blit(author_surf, author_rect)

        pygame.display.flip()
        pygame.time.wait(30)


def game_loop(args, client, joystick_mappings=None):
    """
    Main simulation loop. Handles a single session from start to end.
    """
    logging.info("GAME LOOP: Initializing new session.")
    os.environ["SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS"] = "0"

    pygame.font.init()

    world_obj = None
    session_tick_data = []
    original_settings = None
    carla_world = None

    try:
        logging.info(f"Loading map: {args.map}")
        carla_world = client.load_world(args.map)
        original_settings = carla_world.get_settings()

        if args.sync:
            settings = carla_world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            carla_world.apply_settings(settings)
            logging.info("SYNCHRONOUS MODE applied successfully")

        # Multi-monitor setup for panoramic view
        single_monitor_width, single_monitor_height = args.width, args.height
        total_width = single_monitor_width * 2
        total_height = single_monitor_height

        logging.info(
            f"Creating double-wide window for panoramic view: {total_width}x{total_height}"
        )

        pygame.display.set_icon(pygame.image.load("./images/icon.png"))

        display_flags = pygame.HWSURFACE | pygame.DOUBLEBUF | NOFRAME
        display = pygame.display.set_mode((total_width, total_height), display_flags)

        pygame.display.set_caption("CARLA MVD Demo")
        logging.info(
            f"Pygame Display Mode Set: Borderless Window at {total_width}x{total_height}"
        )

        show_title_screen(display, args)

        # Object Initialization
        hud = HUD(total_width, total_height, args)
        world_obj = World(
            carla_world, hud, args.filter, args.fov, not args.novehicleparams, args
        )

        if world_obj.player and args.max_rpm > 0:
            physics_control = world_obj.player.get_physics_control()
            physics_control.max_rpm = args.max_rpm
            world_obj.player.apply_physics_control(physics_control)
            logging.info(f"Applied custom max_rpm of {args.max_rpm} to vehicle.")

        controller = DualControl(world_obj, args, joystick_mappings)
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

        clock = pygame.time.Clock()

        # --- Main Tick Loop for this Session ---
        while True:
            clock.tick(20)
            if world_obj.is_reset:
                if world_obj.should_reset_scores:
                    mvd_feature_extractor.reset_scores()
                    hud.reset()
                    world_obj.should_reset_scores = False
                world_obj.restart()
                controller = DualControl(world_obj, args, joystick_mappings)
                continue

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

            if world_obj.player and world_obj.player.is_alive:
                controller.process_commands(world_obj.player, args)
                world_obj.tick(clock, controller.updated_hud_information(), controller)

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
                is_off_road = (
                    (lane_violation_state == Sensors.LaneViolationState.CRITICAL)
                    if lane_violation_state
                    else False
                )
                crossed_line = (
                    lane_violation_state
                    and lane_violation_state != Sensors.LaneViolationState.NORMAL
                )

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
                }
                loop_state_log.update(control_datalog)
                loop_state_log.update(mvd_datalog)
                session_tick_data.append(loop_state_log)
                # --- End of logging block ---

            world_obj.render(display)
            pygame.display.flip()

        # --- End Screen Logic ---
        logging.info("Session ended. Presenting end screen.")
        final_overall_scores = mvd_feature_extractor.get_mvd_datalog_metrics()
        end_screen = EndScreen(display, final_overall_scores, hud.panel_fonts)
        action = end_screen.run()
        return action, joystick_mappings

    except Exception as e:
        logging.critical(f"Critical error in game loop: {e}", exc_info=True)
        return "exit", joystick_mappings
    finally:
        # --- RESTORED: Write session log to file ---
        if session_tick_data:
            try:
                log_dir = "./Session_logs/"
                os.makedirs(log_dir, exist_ok=True)
                log_filename = f"mvd_session_log_{time.strftime('%Y%m%d_%H%M%S')}.json"
                with open(os.path.join(log_dir, log_filename), "w") as f:
                    json.dump(session_tick_data, f, indent=2)
                logging.info(
                    f"MVD session data written to {os.path.join(log_dir, log_filename)}"
                )
            except Exception as e:
                logging.error(f"Error writing MVD session log: {e}")
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
    argparser.add_argument("--map", metavar="M", default="Town10HD", help="Map")
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
        "--display",
        metavar="INDEX",
        default=1,
        type=int,
        help="Index of the display to use for the main window (e.g., 0, 1).",
    )

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format="%(levelname)s: %(message)s", level=log_level)

    os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"
    pygame.init()

    try:
        args.width, args.height = [int(x) for x in args.res.split("x")]
    except ValueError:
        args.width, args.height = 1920, 1080

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
            time.sleep(10)

        max_retries, retry_delay = 10, 5
        for i in range(max_retries):
            try:
                client = carla.Client(args.host, args.port)
                client.set_timeout(20.0)
                logging.info(
                    f"Successfully connected to CARLA Server {client.get_server_version()}"
                )
                break
            except RuntimeError as e:
                logging.warning(f"Connection failed: {e}. Retrying...")
                if i == max_retries - 1:
                    raise
                time.sleep(retry_delay)

        log_dir = "./Carla_recorder_logs/"
        os.makedirs(log_dir, exist_ok=True)
        recorder_filename = os.path.join(
            log_dir, f"session_{time.strftime('%Y%m%d_%H%M%S')}.log"
        )
        client.start_recorder(recorder_filename, True)

        joystick_mappings = None
        while True:
            action, new_mappings = game_loop(args, client, joystick_mappings)
            joystick_mappings = new_mappings
            if action == "exit":
                break
            elif action == "restart":
                continue

    except KeyboardInterrupt:
        print("\nCancelled by user. Exiting...")
    except Exception as e:
        logging.critical(f"Unhandled exception in main: {e}", exc_info=True)
    finally:
        if client:
            client.stop_recorder()
        if carla_server_process:
            sys_task.terminate_popen_process_gracefully(
                carla_server_process, "CARLA Server"
            )
        sys_task.sig_kill_engine(sys_task.get_running_processes())
        pygame.quit()
        logging.info("Main script execution finished.")


if __name__ == "__main__":
    main()
