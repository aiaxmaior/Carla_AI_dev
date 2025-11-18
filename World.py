
import carla
import random
import time
import logging
import Sensors
import TrafficManager
import HUD
import re
import weakref
import math
import numpy as np
import Steering
import pygame
from VehicleLibrary import VehicleLibrary

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

class _SingleScreenCameras:
    def __init__(self, world, vehicle, image_wh, fps=20.0):
        self.world   = world
        self.vehicle = vehicle
        self.fps     = fps
        self.W, self.H = int(image_wh[0]), int(image_wh[1])
        self.front   = None
        self.rear    = None
        self._front_surf = None
        self._rear_surf  = None
        self.rear_enabled = True
        self._font = pygame.font.SysFont(None, 18)

    def _spawn(self, bp_id, attrs, tf):
        bp = self.world.get_blueprint_library().find(bp_id)
        for k, v in attrs.items(): bp.set_attribute(k, str(v))
        return self.world.spawn_actor(bp, tf, attach_to=self.vehicle)

    def _to_surface(self, img):
        arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
        arr = arr[:, :, :3][:, :, ::-1]  # BGRA->RGB
        return pygame.image.frombuffer(arr.tobytes(), (img.width, img.height), "RGB")

    def spawn(self):
        # Front (driver view): fill window
        front_attrs = {"image_size_x": self.W, "image_size_y": self.H, "fov": 90, "sensor_tick": 1.0/self.fps}
        tf_front = carla.Transform(carla.Location(x=0.6, z=1.5), carla.Rotation(pitch=-3))
        self.front = self._spawn("sensor.camera.rgb", front_attrs, tf_front)
        self.front.listen(lambda im: setattr(self, "_front_surf", self._to_surface(im)))

        # Rear (PIP): smaller, we scale on blit
        rear_attrs  = {"image_size_x": max(320, self.W//3), "image_size_y": max(180, self.H//5), "fov": 100, "sensor_tick": 1.0/self.fps}
        tf_rear = carla.Transform(carla.Location(x=-1.5, z=1.5), carla.Rotation(yaw=180))
        self.rear  = self._spawn("sensor.camera.rgb", rear_attrs, tf_rear)
        self.rear.listen(lambda im: setattr(self, "_rear_surf", self._to_surface(im)))

    def draw(self, display):
        # Fullscreen front
        if self._front_surf:
            if self._front_surf.get_size() != display.get_size():
                display.blit(pygame.transform.smoothscale(self._front_surf, display.get_size()), (0, 0))
            else:
                display.blit(self._front_surf, (0, 0))

        # Top-center rear PIP
        if self.rear_enabled and self._rear_surf:
            pip_w = int(display.get_width() * 0.28)
            aspect = self._rear_surf.get_width() / max(1, self._rear_surf.get_height())
            pip_h = int(pip_w / max(0.01, aspect))
            pip = pygame.transform.smoothscale(self._rear_surf, (pip_w, pip_h))
            x = (display.get_width() - pip_w)//2
            y = 16
            pygame.draw.rect(display, (10,10,10), (x-3, y-3, pip_w+6, pip_h+6), border_radius=8)
            display.blit(pip, (x, y))
            lbl = self._font.render("REAR VIEW", True, (240,240,240))
            display.blit(lbl, (x+8, y+8))

    def toggle_rear(self):
        self.rear_enabled = not self.rear_enabled

    def destroy(self):
        for s in (self.front, self.rear):
            if s is not None:
                try: s.stop(); s.destroy()
                except: pass
        self.front = self.rear = None
        self._front_surf = self._rear_surf = None

class World(object):
    """
    Manages the CARLA simulation world, player vehicle,
    and associated sensors.
    """

    def __init__(
        self,
        carla_world,
        hud_instance,
        actor_id,
        carla_blueprint,   
        fov,
        vehicleparams,
        args_for_control,
        
    ):
        self.world = carla_world
        self._map = carla_world.get_map()
        self.hud = hud_instance
        self.config_vehicle = actor_id
        self._actor_filter = carla_blueprint
        self.vehicle_config = None
        self.vehicle_config_id = None
        self.steering_model = None
        self._axle_loads = None        
        
        self.fov = fov
        self.advanced_vehicle_params = vehicleparams
        self.args_for_control = args_for_control
        self.is_reset = False
        self.should_reset_scores = False

        # --- MODIFIED: Initialize controller and lane_manager to None ---
        # They will be set in finalize_initialization() after the controller is created in Main.py
        self.controller = None
        self.lane_manager = None

        # --- Collision modification for FusoRosa
        self.collision_probes = [] # List to hold our new sensors
        self.collision_data = {"collided": False, "actor_type": None, "intensity": 0.0}
        # -----------------------

        self.player = None
        self.collision_sensor_instance = None
        self.lane_invasion_sensor_instance = None
        self.gnss_sensor_instance = None
        self.camera_manager = None

        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._last_collision_ts = 0.0
        self._collision_cooldown = 0.5

        self.single_cams=None
        self._display_size = None # (w,h) for respawns
        self.actors_to_destroy = []

        self.client = carla.Client(
            self.args_for_control.host, self.args_for_control.port
        )
        self.client.set_timeout(300.0)

        # --- MODIFIED: restart() is no longer called here to break the dependency loop.
        # It will be called from finalize_initialization().

    def enable_single_screen_cameras(self, window_size):
        """Spawn/respawn driver-view rig if in single layout."""
        self._display_size = window_size
        try:
            if self.single_cams:
                self.single_cams.destroy()
            self.single_cams = _SingleScreenCameras(self.world, self.player, window_size, fps=20.0)
            self.single_cams.spawn()
            logging.info("[SingleScreen] Cameras active (front + rear PIP).")
        except Exception as e:
            logging.error(f"[SingleScreen] Failed to enable cameras: {e}")
            self.single_cams = None

    def disable_single_screen_cameras(self):
        if self.single_cams:
            self.single_cams.destroy()
            self.single_cams = None

    def load_vehicle_config(self, id_or_path: str):
        try:
            lib = VehicleLibrary()
            cfg = lib.load(id_or_path)
            self.vehicle_config = cfg
            self.vehicle_config_id = cfg.id

            # NEW: override actor filter if JSON provides a blueprint
            bp = cfg.get("blueprint", None)
            if isinstance(bp, str) and bp.startswith("vehicle."):
                self._actor_filter = bp
                logging.info(f"[VehicleConfig] '{cfg.id}' → blueprint set: {bp}")
            else:
                logging.info(f"[VehicleConfig] '{cfg.id}' loaded (no blueprint override)")
        except Exception as e:
            logging.error(f"Vehicle config load failed: {e}")
            self.vehicle_config = None
            self.vehicle_config_id = None



    def init_steering_model(self):
        if not self.player:
            logging.warning("init_steering_model: player not ready.")
            return
        try:
            from Steering import SteeringModel
        except Exception as e:
            logging.warning(f"SteeringModel unavailable: {e}")
            self.steering_model = None
            return

        pc = self.player.get_physics_control()
        cfg = getattr(self, "vehicle_config", None)

        # geometry (prefer JSON; fallback to bbox estimate)
        bb = self.player.bounding_box.extent
        wb = float(cfg.get("wheelbase", bb.x * 2.0 if cfg else bb.x * 2.0)) if cfg else bb.x * 2.0
        tf = float(cfg.get("track_front", bb.y * 2.0 if cfg else bb.y * 2.0)) if cfg else bb.y * 2.0
        tr = float(cfg.get("track_rear", tf)) if cfg else tf

        # physical max from CARLA wheels, then cap by JSON steer_cap_deg if present
        try:
            front_wheels = [w for w in pc.wheels if getattr(w, "max_steer_angle", 0.0) > 0.0]
            max_deg = max((w.max_steer_angle for w in front_wheels), default=0.0)
        except Exception:
            max_deg = 0.0
        try:
            if cfg and cfg.get("steer_cap_deg", None) is not None:
                max_deg = min(max_deg or 1e9, float(cfg.get("steer_cap_deg")))
        except Exception:
            pass

        self.steering_model = SteeringModel.from_vehicle_and_config(
            wheelbase=wb,
            track_front=tf,
            track_rear=tr,
            mass=float(cfg.get("mass", getattr(pc, "mass", 3000.0))) if cfg else getattr(pc, "mass", 3000.0),
            front_static_ratio=float(cfg.get("front_static_ratio", 0.5)) if cfg else 0.5,
            vehicle_max_steer_deg=max_deg or 20.0,
            mu=float(cfg.get("friction_mu", 0.9)) if cfg else 0.9,
            Cf=cfg.get("Cf_axle", None) if cfg else None,
            Cr=cfg.get("Cr_axle", None) if cfg else None,
            base_tau_s=float(cfg.get("steer_tau_s", 0.25)) if cfg else 0.25,
            max_steer_rate_dps=float(cfg.get("steer_rate_dps", 360.0)) if cfg else 360.0,
            steer_curve=cfg.get("steer_curve", None) if cfg else None,
        )
        logging.info(f"[SteeringModel] wb={wb:.3f} tf={tf:.3f} tr={tr:.3f} max_deg={max_deg:.2f}")



    def finalize_initialization(self, controller):
        """
        --- NEW METHOD ---
        Completes the setup of the world once the controller object is available.
        This breaks the circular dependency between World and DualControl.
        """
        self.controller = controller

        # Now that we have the controller, we can safely create all player-related systems.
        self.restart()

        # Introduce custom steering model
        try:
            self.init_steering_model()
        except Exception as e:
            logging.error(f"init_steering_model failed: {e}")

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

    def _on_collision_event(self, event):
        """
        Callback for all collision probes. Updates a shared collision state
        and FORCES a HUD notification.
        """
        logging.info("collision event triggered")
        now = time.time()
        if (now - self._last_collision_ts) < self._collision_cooldown:
            return
        self._last_collision_ts = now

        self.collision_data["collided"] = True
        self.collision_data["actor_type"] = event.other_actor.type_id if event.other_actor else "unknown"
        impulse = event.normal_impulse
        self.collision_data["intensity"] = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        
        # This bypasses the EventManager's cooldown for the initial impact,
        # ensuring the critical alert always shows up immediately.
        self.hud.notification("COLLISION!", seconds=3.0, text_color=(255,0,0), is_critical_center=True)
        self.hud.play_sound_for_event('collision', force_play=True)


    def get_collision_data_and_reset(self):
        """NEW: Public method for Main.py to get collision data."""
        data = self.collision_data.copy()
        self.collision_data = {"collided": False, "actor_type": None, "intensity": 0.0}
        return data

    def apply_advanced_vehicle_parameters(self):
        """
        Apply per-vehicle physics from self.vehicle_config (JSON) with safe fallbacks.
        - Respects JSON keys when present; otherwise keeps current CARLA values.
        - Detects steerable wheels from physics control (max_steer_angle > 0).
        - Optional JSON keys: mass, com_x, com_y, com_z, moi, drag_coefficient,
        max_rpm, torque_curve (list of [rpm, torque] or {x:,y:}),
        wheel_friction, wheel_lateral_stiffness,
        front_wheel_friction, rear_wheel_friction,
        front_lat_stiffness, rear_lat_stiffness,
        steer_cap_deg, use_sweep_wheel_collision.
        """
        import math
        import carla

        if not getattr(self, "player", None):
            logging.warning("apply_advanced_vehicle_parameters: no player; skipping.")
            return

        pc = self.player.get_physics_control()
        cfg = getattr(self, "vehicle_config", None)

        def cget(k, default=None):
            """Prefer dict-like .get; fall back to attribute; else default."""
            if cfg is None:
                return default
            try:
                return cfg.get(k, default)  # VehicleConfig implements get()
            except Exception:
                return getattr(cfg, k, default)

        def set_if(obj, attr, value):
            """Set obj.attr = value if attr exists and value is not None."""
            try:
                if value is not None and hasattr(obj, attr):
                    setattr(obj, attr, value)
            except Exception as e:
                logging.debug(f"set_if({attr}) failed: {e}")

        logging.info("Applying custom physics from JSON (if provided)")

        # ---------- Core body parameters ----------
        # Mass
        try:
            mass_val = cget("mass", None)
            if mass_val is not None:
                pc.mass = float(mass_val)
        except Exception as e:
            logging.warning(f"[VehicleConfig] mass parse failed: {e}")

        # Center of Mass (optional: x/y/z)
        try:
            cx = cget("com_x", None)
            cy = cget("com_y", None)
            cz = cget("com_z", None)
            if cx is not None:
                pc.center_of_mass.x = float(cx)
            if cy is not None and hasattr(pc.center_of_mass, "y"):
                pc.center_of_mass.y = float(cy)
            if cz is not None:
                pc.center_of_mass.z = float(cz)
            if cx is not None or cy is not None or cz is not None:
                logging.info(f"[VehicleConfig] COM -> x:{pc.center_of_mass.x:.3f} "
                            f"y:{getattr(pc.center_of_mass, 'y', 0.0):.3f} "
                            f"z:{pc.center_of_mass.z:.3f}")
        except Exception as e:
            logging.warning(f"[VehicleConfig] COM set failed: {e}")

        # MOI (if supported)
        try:
            moi_val = cget("moi", None)
            if moi_val is not None and hasattr(pc, "moi"):
                pc.moi = float(moi_val)
        except Exception as e:
            logging.warning(f"[VehicleConfig] MOI set failed: {e}")

        # Drag
        try:
            drag = cget("drag_coefficient", None)
            if drag is not None:
                pc.drag_coefficient = float(drag)
        except Exception as e:
            logging.warning(f"[VehicleConfig] drag set failed: {e}")

        # ---------- Powertrain ----------
        # Torque curve
        tc = cget("torque_curve", None)
        try:
            if isinstance(tc, (list, tuple)) and len(tc) > 0:
                curve = []
                for pt in tc:
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        rpm, tq = pt[0], pt[1]
                    else:
                        rpm, tq = pt.get("x"), pt.get("y")
                    if rpm is None or tq is None:
                        continue
                    curve.append(carla.Vector2D(x=float(rpm), y=float(tq)))
                if curve:
                    pc.torque_curve = curve
            elif not getattr(pc, "torque_curve", None):
                # fallback default curve if none exists
                pc.torque_curve = [
                    carla.Vector2D(x=0.0,    y=300.0),
                    carla.Vector2D(x=1400.0, y=420.0),
                    carla.Vector2D(x=2800.0, y=350.0),
                    carla.Vector2D(x=4500.0, y=250.0),
                ]
        except Exception as e:
            logging.warning(f"[VehicleConfig] torque_curve set failed: {e}")

        # Max RPM
        try:
            max_rpm_val = cget("max_rpm", None)
            if max_rpm_val is not None:
                pc.max_rpm = float(max_rpm_val)
            elif not hasattr(pc, "max_rpm") or pc.max_rpm in (None, 0.0):
                pc.max_rpm = 5000.0
        except Exception as e:
            logging.warning(f"[VehicleConfig] max_rpm set failed: {e}")

        # Damping (if exposed)
        try:
            set_if(pc, "damping_rate_full_throttle", cget("damping_rate_full_throttle", None))
            set_if(pc, "damping_rate_zero_throttle_clutch_engaged", cget("damping_rate_zero_throttle_clutch_engaged", None))
            set_if(pc, "damping_rate_zero_throttle_clutch_disengaged", cget("damping_rate_zero_throttle_clutch_disengaged", None))
        except Exception as e:
            logging.debug(f"[VehicleConfig] damping set failed: {e}")

        # Sweep wheel collision (optional)
        try:
            swc = cget("use_sweep_wheel_collision", None)
            if swc is not None and hasattr(pc, "use_sweep_wheel_collision"):
                pc.use_sweep_wheel_collision = bool(swc)
        except Exception as e:
            logging.debug(f"[VehicleConfig] sweep wheel set failed: {e}")

        # ---------- Wheels ----------
        wheels = list(pc.wheels)

        # Determine steerable wheels from current physics (robust across blueprints)
        steerable_idx = [i for i, w in enumerate(wheels) if getattr(w, "max_steer_angle", 0.0) > 0.0]
        if not steerable_idx and len(wheels) >= 2:
            # last-resort fallback: assume first axle is steerable
            steerable_idx = [0, 1]

        # Per-axle helpers: treat "steerable" as front axle for tuning
        front_mask = set(steerable_idx)
        rear_mask  = set(range(len(wheels))) - front_mask

        # Friction / stiffness
        fric_common = cget("wheel_friction", None)
        lat_common  = cget("wheel_lateral_stiffness", None)
        fric_front  = cget("front_wheel_friction", fric_common)
        fric_rear   = cget("rear_wheel_friction",  fric_common)
        lat_front   = cget("front_lat_stiffness",   lat_common)
        lat_rear    = cget("rear_lat_stiffness",    lat_common)

        # Steer cap (deg)
        steer_cap = cget("steer_cap_deg", None)
        try:
            steer_cap = float(steer_cap) if steer_cap is not None else None
        except Exception:
            steer_cap = None

        # Apply per-wheel
        for i, w in enumerate(wheels):
            try:
                # friction (attribute name differs across versions; set both if present)
                if i in front_mask:
                    if fric_front is not None:
                        if hasattr(w, "friction"):
                            w.friction = float(fric_front)
                        if hasattr(w, "tire_friction"):
                            w.tire_friction = float(fric_front)
                    elif fric_common is not None:
                        if hasattr(w, "friction"):
                            w.friction = float(fric_common)
                        if hasattr(w, "tire_friction"):
                            w.tire_friction = float(fric_common)
                else:
                    if fric_rear is not None:
                        if hasattr(w, "friction"):
                            w.friction = float(fric_rear)
                        if hasattr(w, "tire_friction"):
                            w.tire_friction = float(fric_rear)
                    elif fric_common is not None:
                        if hasattr(w, "friction"):
                            w.friction = float(fric_common)
                        if hasattr(w, "tire_friction"):
                            w.tire_friction = float(fric_common)

                # lateral stiffness (some builds expose 'lateral_stiffness')
                if i in front_mask:
                    if lat_front is not None and hasattr(w, "lateral_stiffness"):
                        w.lateral_stiffness = float(lat_front)
                    elif lat_common is not None and hasattr(w, "lateral_stiffness"):
                        w.lateral_stiffness = float(lat_common)
                else:
                    if lat_rear is not None and hasattr(w, "lateral_stiffness"):
                        w.lateral_stiffness = float(lat_rear)
                    elif lat_common is not None and hasattr(w, "lateral_stiffness"):
                        w.lateral_stiffness = float(lat_common)

                # steer cap in degrees (only for steerable wheels)
                if steer_cap is not None and i in front_mask and hasattr(w, "max_steer_angle"):
                    w.max_steer_angle = float(steer_cap)

            except Exception as e:
                logging.warning(f"Wheel[{i}] parameter set failed: {e}")

        pc.wheels = wheels  # IMPORTANT: reassign the list

        # ---------- Apply & verify ----------
        try:
            self.player.apply_physics_control(pc)
        except Exception as e:
            logging.error(f"apply_physics_control failed: {e}")
            return

        pc_after = self.player.get_physics_control()
        # Log a compact wheel summary
        for i, w in enumerate(pc_after.wheels):
            steer = getattr(w, "max_steer_angle", 0.0)
            fric  = getattr(w, "friction", getattr(w, "tire_friction", 0.0))
            lat   = getattr(w, "lateral_stiffness", 0.0)
            logging.info(f"Post-apply wheel[{i}] steer={steer:.2f}° fric={fric:.2f} lat={lat:.2f}")

        # Verify steer cap if we set one
        if steer_cap is not None:
            over = [i for i in steerable_idx
                    if getattr(pc_after.wheels[i], "max_steer_angle", 0.0) > steer_cap + 1e-3]
            if over:
                logging.warning(f"Steer cap exceeded on wheels {over} (> {steer_cap:.2f} deg); blueprint may clamp later.")
            else:
                logging.info(f"Steer cap verified at {steer_cap:.2f} deg on steerable wheels.")

    def destroy_player_and_sensors(self):
        """Destroys the player vehicle and all its attached sensors/callbacks properly."""
        # Stop HUD pipelines (threads + camera actors inside HUD)
        try:
            if self.hud and hasattr(self.hud, 'destroy'):
                self.hud.destroy()
        except Exception as e:
            logging.warning(f"[cleanup] HUD destroy warning: {e}")

        # Collision cage probes
        for probe in list(self.collision_probes):
            try: probe.stop()
            except Exception: pass
            try: probe.destroy()
            except Exception: pass
        self.collision_probes.clear()

        # Lane manager (has a radar)
        try:
            if self.lane_manager:
                self.lane_manager.destroy()
        except Exception as e:
            logging.warning(f"[cleanup] LaneManagement destroy warning: {e}")
        self.lane_manager = None

        # Standalone sensors (GNSS, lane invasion, old collision if any)
        for s in (self.gnss_sensor_instance,
                self.lane_invasion_sensor_instance,
                getattr(self, "collision_sensor_instance", None)):
            if not s:
                continue
            # Some of these wrapper classes have .sensor; some are actors directly
            actor = getattr(s, "sensor", s)
            try: actor.stop()
            except Exception: pass
            try: actor.listen(None)
            except Exception: pass
            try: actor.destroy()
            except Exception: pass
        self.gnss_sensor_instance = None
        self.lane_invasion_sensor_instance = None
        self.collision_sensor_instance = None

        # CameraManager sensors (if you still keep one outside HUD)
        if getattr(self, "camera_manager", None):
            # Handle multiple sensors (plural)
            if hasattr(self.camera_manager, 'sensors') and self.camera_manager.sensors:
                for sensor in self.camera_manager.sensors.values():
                    try: sensor.stop()
                    except Exception: pass
                    try: sensor.listen(None)
                    except Exception: pass
                    try: sensor.destroy()
                    except Exception: pass
                self.camera_manager.sensors = {}
            # Handle single sensor (legacy)
            elif hasattr(self.camera_manager, 'sensor') and self.camera_manager.sensor:
                try: self.camera_manager.sensor.stop()
                except Exception: pass
                try: self.camera_manager.sensor.listen(None)
                except Exception: pass
                try: self.camera_manager.sensor.destroy()
                except Exception: pass
                self.camera_manager.sensor = None

        # Single-screen helper cams (if enabled)
        if getattr(self, "single_cams", None):
            try: self.single_cams.destroy()
            except Exception: pass
            self.single_cams = None

        # Finally the player
        if self.player:
            try: self.player.destroy()
            except Exception: pass
            self.player = None

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

        #blueprint = self.world.get_blueprint_library().find("vehicle.mitsubishi.fusorosa")
        
        if self._actor_filter != "vehicle.mercedes.sprinter":
            blueprint = self.world.get_blueprint_library().find(self._actor_filter)
        else:
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
                spawn_point.location.z += 2.0
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
        #self.collision_sensor_instance = Sensors.CollisionSensor(self.player, self.hud)
        # --- REVISED: Create a high-density 16-point Physical Collision Cage ---
        # --- REVISED: Create a high-density 16-point Physical Collision Cage ---
        bounding_box = self.player.bounding_box
        extent = bounding_box.extent
        player_transform = self.player.get_transform()
        
        buffer_multiplier = 1.3
        # Using your validated height multiplier for the fusorosa
        if self._actor_filter == 'vehicle.mitsubishi.fusorosa':
            height_multiplier = -0.8
        else:
            height_multiplier = 0.25

        x = extent.x * buffer_multiplier
        y = extent.y * buffer_multiplier
        z = extent.z * height_multiplier
        logging.info(f' EXTENT:: X:{extent.x}, Y:"{extent.y},Z:{extent.z}')
        logging.info(f' BOUNDING BOX:: X:{bounding_box.location.x}, Y: {bounding_box.location.y}, Z: {bounding_box.location.z}')
        logging.info(f' SENSOR PLACEMENT:: X:{x}, Y:{y}, Z:{z}')
        # --- ADDED: Calculate the absolute min/max range for logging ---
        # Get the world location of the bounding box center
        box_center_world = player_transform.location + bounding_box.location
        
        # Define the expected range based on the box center and its extent
        x_min, x_max = box_center_world.x - extent.x, box_center_world.x + extent.x
        y_min, y_max = box_center_world.y - extent.y, box_center_world.y + extent.y
        z_min, z_max = box_center_world.z - extent.z, box_center_world.z + extent.z

        logging.info(f"VEHICLE WORLD BOUNDS: X({x_min:.2f} to {x_max:.2f}), Y({y_min:.2f} to {y_max:.2f}), Z({z_min:.2f} to {z_max:.2f})")
        
        # Create points for the cage shape
        front_edge = [carla.Location(x=x, y=val) for val in np.linspace(-y, y, 5)]
        back_edge = [carla.Location(x=-x, y=val) for val in np.linspace(-y, y, 5)]
        left_edge = [carla.Location(x=val, y=-y) for val in np.linspace(-x, x, 5)]
        right_edge = [carla.Location(x=val, y=y) for val in np.linspace(-x, x, 5)]

        all_points = front_edge + back_edge + left_edge + right_edge
        unique_points_tuples = set((p.x, p.y) for p in all_points)
        perimeter_locations = [carla.Location(px, py, z) for px, py in unique_points_tuples]

        # --- CORRECTED SPAWNING LOOP WITH ACCURATE LOGGING ---
        collision_bp = self.world.get_blueprint_library().find("sensor.other.collision")
        player_transform = self.player.get_transform() # Get the player's transform once before the loop
        probe_count = 0

        for location in perimeter_locations:
            # Create the final relative transform for the probe
            relative_transform = carla.Transform(location + bounding_box.location)
            
            # Spawn ONE sensor using the safe try_spawn_actor method
            probe = self.world.try_spawn_actor(collision_bp, relative_transform, attach_to=self.player)
            
            if probe is not None:
                probe_count += 1
                
                # --- FIX: Calculate only the final location, which is all we need for the log ---
                # The 'transform()' method correctly calculates the probe's final world position.
                spawn_location = player_transform.transform(relative_transform.location)
                
                # Check if the calculated probe location is within the vehicle bounds
                in_x = x_min <= spawn_location.x <= x_max
                in_y = y_min <= spawn_location.y <= y_max
                in_z = z_min <= spawn_location.z <= z_max
                is_within_range = "YES" if (in_x and in_y and in_z) else "NO"

                logging.info(f"  [Probe {probe_count}] Commanded to spawn at X:{spawn_location.x:.2f} Y:{spawn_location.y:.2f} Z:{spawn_location.z:.2f}. Within Vehicle Bounds? -> {is_within_range}")
                debug = self.world.debug
                debug.draw_point(spawn_location, size=0.15, life_time=120.0, persistent_lines=True, color=carla.Color(255,0,0))
        
                weak_self = weakref.ref(self)
                probe.listen(lambda event: weak_self()._on_collision_event(event))
                self.collision_probes.append(probe)
            else:
                logging.warning(f"Failed to spawn collision probe at {relative_transform.location}, possibly occupied.")

        logging.info(f"Spawned collision cage with {len(self.collision_probes)} probes.")
        
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

        if getattr(self.args_for_control,"layout_mode","quad") !="single":
            if self.camera_manager is None:
                self.camera_manager = HUD.CameraManager(self.player,self.hud,self.fov)
            else:
                #Single layout
                if self._display_size:
                    self.enable_single_screen_cameras(self._display_size)
       # if self.camera_manager is None:
       #     self.camera_manager = HUD.CameraManager(self.player, self.hud, self.fov)

#        self.camera_manager.set_sensor(cam_index, notify=False)

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

    def tick(self, clock, idling, controller, display_fps):
        """Updates the world state, including HUD and spectator camera."""
        self.hud.tick(self, clock, idling, controller, display_fps)
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
    """
    def render(self, display):
        # Renders the camera view and HUD to the Pygame display.
        if self.camera_manager:
            self.camera_manager.render(display)
        if self.hud:
            self.hud.render(display)
    """
    def render(self, display):
        """Renders the camera view and HUD to the Pygame display."""
        if getattr(self.args_for_control, "layout_mode", "quad") == "single" and self.single_cams:
            self.single_cams.draw(display)
            if self.hud:
                self.hud.render(display)
            return

        # Default (quad) path
        if self.camera_manager:
            self.camera_manager.render(display)
        if self.hud:
            self.hud.render(display)

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
