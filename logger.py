import argparse
import logging
import os
import sys
import subprocess
import time
import datetime
import json
import carla
import sys_task
import math

from World import World
from HUD import HUD, EndScreen
from controls_queue import DualControl
from Sensors import CollisionSensor, LaneInvasionSensor, GnssSensor
from MVD import MVDFeatureExtractor
from lane_mgmt_integration import LaneManagement, LaneChangeState

try:
    import pygame
    from pygame.locals import K_ESCAPE, K_BACKSPACE
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")

# Ensure SDL minimizes on focus loss (Linux fix)
os.environ["SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS"] = "0"

# Global CARLA server process handle
carla_server_process = None


def configure_logging(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    logfile = os.path.join(log_dir, f"carla_mvd_{timestamp}.log")
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(logfile)]
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=handlers
    )
    logging.info(f"Logging to {logfile}")
    return logfile


def start_carla_server(carla_path, launch_args=None):
    """
    Launches CARLA server if path provided, returns Popen.
    """
    cmd = [carla_path]
    if launch_args:
        cmd += launch_args
    try:
        proc = subprocess.Popen(cmd)
        logging.info(f"Launched CARLA server: {' '.join(cmd)} (PID {proc.pid})")
        return proc
    except Exception as e:
        logging.error(f"Failed to launch CARLA server: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="CARLA MVD Demo with Lane & Radar Management")
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--map', default='Town03')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--fov', type=float, default=90.0)
    parser.add_argument('--filter', default='vehicle.*')
    parser.add_argument('--sync', action='store_true', help='Enable synchronous mode')
    parser.add_argument('--max-rpm', type=int, default=0)
    parser.add_argument('--noackermann', action='store_true')
    parser.add_argument('--novehicleparams', action='store_true')
    parser.add_argument('--autopilot', action='store_true')
    parser.add_argument('--carla-server', default=None, help='Path to CarlaUE4 executable')
    args = parser.parse_args()

    # Setup logging
    logfile = configure_logging()

    # Optionally launch CARLA server
    global carla_server_process
    if args.carla_server:
        carla_server_process = start_carla_server(args.carla_server)
        time.sleep(5)

    # Connect to CARLA
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    # Start CARLA recorder
    recorder_file = f"recording_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    try:
        client.start_recorder(recorder_file)
        logging.info(f"Started CARLA recorder: {recorder_file}")
    except Exception as e:
        logging.warning(f"Failed to start recorder: {e}")

    joystick_mappings = None
    action = 'restart'
    while action == 'restart':
        action, joystick_mappings = game_loop(args, client, joystick_mappings)

    # Stop recorder
    try:
        client.stop_recorder()
        logging.info("Stopped CARLA recorder.")
    except Exception as e:
        logging.warning(f"Failed to stop recorder: {e}")

    # Shutdown CARLA server
    if carla_server_process:
        sys_task.terminate_popen_process_gracefully(carla_server_process, "CARLA Server")
        procs = sys_task.get_running_processes()
        sys_task.sig_kill_engine(procs)

    logging.info("Exiting.")
    sys.exit(0)


def game_loop(args, client, joystick_mappings=None):
    logging.info("GAME LOOP: Starting session.")

    # Initialize Pygame
    pygame.init()
    pygame.font.init()

    # Load and configure world
    carla_world = client.load_world(args.map)
    if args.sync:
        settings = carla_world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        carla_world.apply_settings(settings)
        logging.info("Synchronous mode enabled.")

    # Multi-display logic
    num_displays = pygame.display.get_num_displays()
    target_idx, largest_area = 0, 0
    target_res = (args.width, args.height)
    if num_displays > 1:
        for i, size in enumerate(pygame.display.get_desktop_sizes()):
            area = size[0]*size[1]
            if area > largest_area:
                largest_area = area
                target_idx = i
                target_res = size
    native_w, native_h = target_res

    pygame.display.set_icon(pygame.image.load("./images/icon.png"))
    display = pygame.display.set_mode((native_w, native_h), pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.FULLSCREEN, target_idx)
    pygame.display.set_caption("CARLA MVD Demo")

    # HUD & world instantiation
    hud = HUD(native_w, native_h, args)
    hud_fonts = {'_font_primary_hud': hud._font_primary_hud, '_font_secondary_hud': hud._font_secondary_hud, '_font_score_hud': hud._font_score_hud}
    world_obj = World(carla_world, hud, args.filter, args.fov, not args.novehicleparams, args)

    # Apply custom physics
    if world_obj.player and args.max_rpm > 0:
        pc = world_obj.player.get_physics_control()
        pc.max_rpm = args.max_rpm
        world_obj.player.apply_physics_control(pc)
        logging.info(f"Max RPM set to {args.max_rpm}")

    # Initialize sensors
    collision_sensor = CollisionSensor(world_obj.player, hud)
    lane_sensor = LaneInvasionSensor(world_obj.player, hud)
    gnss_sensor = GnssSensor(world_obj.player, hud)

    # Controls
    controller = DualControl(world_obj, args, joystick_mappings)
    joystick_mappings = controller.mapped_controls

    # Lane & radar manager
    lane_manager = LaneManagement(world_obj.player, hud, controller)

    # MVD extractor
    mvd = MVDFeatureExtractor()
    mvd.reset_scores()
    if args.sync:
        logging.info("Running stabilization ticks...")
        for _ in range(15): carla_world.tick()
        logging.info("Stabilization done.")

    clock = pygame.time.Clock()
    running = True
    while running:
        clock.tick(20)
        if args.sync:
            carla_world.tick()
        else:
            carla_world.wait_for_tick()

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == K_ESCAPE):
                running = False
            controller.parse_event(event)

        # Sensor readings
        col_data = collision_sensor.get_collision_data_and_reset()
        lane_data = lane_sensor.get_invasion_data_and_reset()
        gps_data = gnss_sensor.get_data()

        # Manage lane changes
        if lane_data:
            lane_manager.evaluate_lane_change(lane_data)

        # Update MVD scores
        mvd.update_scores(col_data, {'markings': lane_data}, 0.0, world_obj.player, world_obj.world, time.time())
        mvd.update_lane_management(lane_manager.state)

        # HUD update
        overall = mvd.calculate_overall_score()
        mbi = mvd._harsh_driving_score
        lmi = mvd._lane_management_score
        hud.update_mvd_scores_for_display(overall, mbi, lmi)
        world_obj.world.on_tick(hud.on_world_tick)  # Sync HUD
        world_obj.world.tick()
        hud.tick(world_obj, clock, controller.idling, controller.blinker_state)
        hud.render(display)
        pygame.display.flip()

        # Check catastrophic
        if mvd._catastrophic_failure_occurred:
            logging.warning("Catastrophic failure. Ending session.")
            break

        # Reset session
        if world_obj.is_reset:
            logging.info("Reset requested.")
            mvd.reset_scores()
            hud.reset()
            world_obj.restart()
            controller = DualControl(world_obj, args, joystick_mappings)

    # End screen
    end_scores = {'overall': mvd.calculate_overall_score()}
    end_screen = EndScreen(display, end_scores, hud_fonts)
    action = end_screen.run()

    # Cleanup
    collision_sensor.destroy()
    lane_sensor.destroy()
    gnss_sensor.destroy()
    lane_manager.destroy()
    world_obj.destroy()
    pygame.quit()
    logging.info("Session ended.")

    return action, joystick_mappings
