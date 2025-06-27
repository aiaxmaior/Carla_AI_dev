import carla
import random
import time
import logging
import Sensors
import TrafficManager
import HUD
import re


def get_actor_display_name(actor, truncate=250):
    """Helper function to get display name of a CARLA actor."""
    if not actor:
        return "N/A"
    name = " ".join(actor.type_id.replace("_", ".").title().split(".")[1:])
    return (name[: truncate - 1] + "\u2026") if len(name) > truncate else name


def find_weather_presets():
    """Helper function to find CARLA weather presets."""
    rgx = re.compile(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)")
    name = lambda x: " ".join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match("[A-Z].+", x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


class World(object):
    """
    Manages the CARLA simulation world, player vehicle,
    and associated sensors.
    """

    def __init__(
        self,
        carla_world,
        hud_instance,
        actor_filter,
        fov,
        vehicleparams,
        args_for_control,
    ):
        self.world = carla_world
        self._map = carla_world.get_map()
        self.hud = hud_instance
        self._actor_filter = actor_filter
        self.fov = fov
        self.advanced_vehicle_params = vehicleparams
        self.args_for_control = args_for_control
        self.is_reset = False
        self.should_reset_scores = False

        # --- MODIFIED: Initialize controller and lane_manager to None ---
        # They will be set in finalize_initialization() after the controller is created in Main.py
        self.controller = None
        self.lane_manager = None

        self.player = None
        self.collision_sensor_instance = None
        self.lane_invasion_sensor_instance = None
        self.gnss_sensor_instance = None
        self.camera_manager = None

        self._weather_presets = find_weather_presets()
        self._weather_index = 0

        self.actors_to_destroy = []

        self.client = carla.Client(
            self.args_for_control.host, self.args_for_control.port
        )
        self.client.set_timeout(300.0)

        # --- MODIFIED: restart() is no longer called here to break the dependency loop.
        # It will be called from finalize_initialization().

    def finalize_initialization(self, controller):
        """
        --- NEW METHOD ---
        Completes the setup of the world once the controller object is available.
        This breaks the circular dependency between World and DualControl.
        """
        self.controller = controller

        # Now that we have the controller, we can safely create all player-related systems.
        self.restart()

        # Spawn traffic immediately after the player is set up.
        num_vehicles = getattr(self.args_for_control, "num_vehicles", 50)
        num_pedestrians = getattr(self.args_for_control, "num_pedestrians", 30)

        spawned_actors = TrafficManager.spawn_traffic(
            self.client,
            self.world,
            num_vehicles=num_vehicles,
            num_pedestrians=num_pedestrians,
        )

        self.actors_to_destroy.extend(spawned_actors["vehicles"])
        self.actors_to_destroy.extend(spawned_actors["pedestrians"])

        self.world.on_tick(self.hud.on_world_tick)

    def apply_advanced_vehicle_parameters(self):
        logging.info("Applying custom physics for a 15-Passenger Sprinter van...")
        physics_control = self.player.get_physics_control()
        physics_control.mass = 3000.0
        physics_control.center_of_mass.z = -1
        physics_control.torque_curve = [
            carla.Vector2D(x=0, y=150),
            carla.Vector2D(x=1000, y=170),
            carla.Vector2D(x=1500, y=225),
            carla.Vector2D(x=2000, y=250),
            carla.Vector2D(x=2500, y=258),
            carla.Vector2D(x=3500, y=258),
            carla.Vector2D(x=4000, y=235),
            carla.Vector2D(x=4500, y=210),
        ]
        physics_control.max_rpm = 4500.0
        for wheel in physics_control.wheels:
            wheel.lateral_stiffness = 15.0
            wheel.friction = 2.0
        physics_control.wheels[0].max_steer_angle = 18
        physics_control.wheels[1].max_steer_angle = 18.0
        physics_control.drag_coefficient = 0.60
        self.player.apply_physics_control(physics_control)
        logging.info("Custom physics for 15-Passenger Sprinter have been applied.")

    def restart(self):
        """
        Restarts the simulation, destroying the old player and spawning a new one
        with attached sensors and camera.
        """
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = (
            self.camera_manager.transform_index
            if self.camera_manager is not None
            else 0
        )

        blueprint = self.world.get_blueprint_library().find("vehicle.mercedes.sprinter")
        blueprint.set_attribute("role_name", "hero")

        if blueprint.has_attribute("color"):
            color = random.choice(blueprint.get_attribute("color").recommended_values)
            blueprint.set_attribute("color", color)

        if self.player is not None:
            self.destroy_player_and_sensors()
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = (
                random.choice(spawn_points) if spawn_points else carla.Transform()
            )
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            if self.player:
                spawn_point = self.player.get_transform()
                spawn_point.location.z += 1.5
                spawn_point.rotation.roll = 0.0
                spawn_point.rotation.pitch = 0.0
                self.player.set_transform(spawn_point)
            logging.info("Simulation Restarted")

        while self.player is None:
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = (
                random.choice(spawn_points) if spawn_points else carla.Transform()
            )
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            if self.player is None:
                logging.warning("Failed to spawn player, retrying...")
                time.sleep(0.5)
        logging.info("player spawned. Ticking world to ensure actor is alive")

        for _ in range(3):
            self.world.tick()

        if isinstance(self.player, carla.Vehicle):
            self.player.set_autopilot(False)

        self.player.set_simulate_physics(True)
        if self.advanced_vehicle_params:
            self.apply_advanced_vehicle_parameters()

        self.hud.reset()

        # --- SENSOR INITIALIZATION (Corrected Order) ---
        self.collision_sensor_instance = Sensors.CollisionSensor(self.player, self.hud)

        # Create the LaneManagement system first, as it's needed by the LaneInvasionSensor
        self.lane_manager = Sensors.LaneManagement(
            self.player, self.hud, self.controller
        )

        # Now create the LaneInvasionSensor and pass the lane_manager to it
        self.lane_invasion_sensor_instance = Sensors.LaneInvasionSensor(
            self.player, self.hud, self.lane_manager
        )

        self.gnss_sensor_instance = Sensors.GnssSensor(self.player, self.hud)
        logging.info("Core sensors initialized for new player.")
        # --- END SENSOR INITIALIZATION ---

        if self.camera_manager is None:
            self.camera_manager = HUD.CameraManager(self.player, self.hud, self.fov)
        else:
            self.camera_manager._parent = self.player
            self.camera_manager.set_sensor(self.camera_manager.index, notify=False)
            logging.info("restarting")
            self.is_reset = True

        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)

        actor_type = get_actor_display_name(self.player)

        if self.is_reset:
            self.hud.notification(
                f"Simulation Restarted: {actor_type} RESPAWNED", 3.0, (0, 255, 0)
            )
        else:
            self.hud.notification(f"{actor_type} Ready!")

        self.is_reset = False

    def next_weather(self, reverse=False):
        """Changes the weather conditions in the simulation."""
        if not self.player:
            return
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification("Weather: %s" % preset[1])
        self.player.get_world().set_weather(preset[0])

    def tick(self, clock, idling, controller):
        """Updates the world state, including HUD and spectator camera."""
        self.hud.tick(self, clock, idling, controller)
        if (
            self.player is not None
            and isinstance(self.player, carla.Vehicle)
            and self.player.is_alive
        ):
            spectator = self.world.get_spectator()
            vehicle_transform = self.player.get_transform()
            driver_seat_offset_location = carla.Location(x=0.8, y=-0.4, z=1.3)
            rotated_offset = vehicle_transform.transform_vector(
                driver_seat_offset_location
            )
            spectator_location = vehicle_transform.location + rotated_offset
            spectator_transform = carla.Transform(
                spectator_location, vehicle_transform.rotation
            )
            spectator.set_transform(spectator_transform)

    def render(self, display):
        """Renders the camera view and HUD to the Pygame display."""
        if self.camera_manager:
            self.camera_manager.render(display)
        if self.hud:
            self.hud.render(display)

    def destroy_player_and_sensors(self):
        """Destroys the player vehicle and all its attached sensors."""
        sensors_to_destroy = [
            self.collision_sensor_instance,
            self.lane_invasion_sensor_instance,
            self.gnss_sensor_instance,
        ]

        # --- ADDED: Destroy the lane manager's radar sensor ---
        if self.lane_manager:
            self.lane_manager.destroy()
            self.lane_manager = None

        for sensor in sensors_to_destroy:
            if sensor:
                sensor.destroy()

        self.collision_sensor_instance = None
        self.lane_invasion_sensor_instance = None
        self.gnss_sensor_instance = None

        if self.camera_manager and self.camera_manager.sensor:
            self.camera_manager.sensor.destroy()
            self.camera_manager.sensor = None

        if self.player and self.player.is_alive:
            self.player.destroy()
        self.player = None
        logging.info("Player and primary sensors destroyed.")

    def destroy_all_actors(self):
        """Destroys all actors spawned by this World instance."""
        self.destroy_player_and_sensors()
        logging.info(
            f"Destroying {len(self.actors_to_destroy)} additional traffic actors..."
        )
        if self.client:
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.actors_to_destroy]
            )
            self.actors_to_destroy.clear()
            logging.info("All traffic actors destroyed.")
        else:
            logging.warning("CARLA client not available to destroy traffic actors.")
