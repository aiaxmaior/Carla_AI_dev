# ScenarioManager.py
# Lightweight, interactive scenarios for CARLA 0.9.16 (UE4).
# Player stays in control; scenarios spawn/drive context and evaluate via PredictiveIndices.

# ============================================================================
# PERF CHECK (file-level):
# ============================================================================
# [X] | Role: Scenario orchestration (spawns actors, manages test scenarios)
# [X] | Hot-path functions: tick() if scenario active (20 Hz)
# [X] |- Heavy allocs in hot path? Moderate - CSV writes, predictor updates
# [X] |- pandas/pyarrow/json/disk/net in hot path? CSV writes (file I/O)
# [ ] | Graphics here? No
# [X] | Data produced (tick schema?): Scenario state + predictor results
# [X] | Storage (Parquet/Arrow/CSV/none): CSV export per scenario
# [ ] | Queue/buffer used?: No - direct writes
# [X] | Session-aware? Per-scenario tracking
# [ ] | Debug-only heavy features?: None
# Top 3 perf risks:
# 1. [PERF_HOT] CSV write every tick (file I/O in hot path!)
# 2. [PERF_SPLIT] Should buffer writes or use async I/O
# 3. [PERF_OK] Only active during scenario runs (not normal driving)
# ============================================================================

import math, random, csv, time
from dataclasses import dataclass
from typing import Optional, Dict

import carla

try:
    from PredictiveIndices import PredictiveIndices, PredictiveConfig
except Exception:
    # Soft fallback if not present at import time
    PredictiveIndices = None
    PredictiveConfig = None

HZ = 20.0
DT = 1.0 / HZ

# ------------------ Base ------------------

@dataclass
class ScenarioState:
    name: str
    started: bool = False
    finished: bool = False
    ok: bool = True
    t: float = 0.0
    phase: str = "init"
    notes: str = ""

class InteractiveScenarioBase:
    name = "Base"
    description = "Base scenario"
    duration_s = 15.0

    def __init__(self, world: carla.World, client: carla.Client, tm_port=8000, hud=None):
        self.world = world
        self.client = client
        self.tm_port = tm_port
        self.hud = hud
        self.state = ScenarioState(name=self.name)
        self.orig_settings = None
        self.ego: Optional[carla.Vehicle] = None
        self.lead: Optional[carla.Vehicle] = None
        self.actors = []
        self.predictor = PredictiveIndices(PredictiveConfig()) if PredictiveIndices else None
        self._csv = None
        self._writer = None

    # ---- lifecycle ----
    def setup(self, ego: carla.Vehicle):
        self.ego = ego
        self.state.started = True
        self.state.t = 0.0

        # sync @20Hz (keep existing if already sync)
        self.orig_settings = self.world.get_settings()
        s = carla.WorldSettings(self.orig_settings)
        s.synchronous_mode = True
        s.fixed_delta_seconds = DT
        self.world.apply_settings(s)

        # open CSV (optional; safe if disk not writable)
        try:
            self._csv = open(f"{self.name}.csv", "w", newline="")
            self._writer = csv.writer(self._csv)
            self._writer.writerow(["t","phase","v","ax","yaw_rate","e_y","lane_w",
                                   "d_lead","v_rel","p_lane","p_col","p_oper","notes"])
        except Exception:
            self._csv = None
            self._writer = None

        self._on_setup()

    def tick(self) -> Dict:
        """Advance one frame. Returns predictor dict + metadata."""
        self.world.tick()
        self.state.t += DT
        obs = self._observe()
        preds = self.predictor.update(self.state.t, obs) if self.predictor else {}

        # scenario step
        self._on_tick(obs, preds)

        # log row
        if self._writer and obs:
            self._writer.writerow([
                f"{self.state.t:.2f}", self.state.phase,
                round(obs.get("v",0.0),3), round(obs.get("ax",0.0),3), round(obs.get("yaw_rate",0.0),3),
                round(obs.get("lateral_offset",0.0),3), round(obs.get("lane_width",3.6),2),
                "" if obs.get("lead_distance") is None else round(obs.get("lead_distance"),2),
                "" if obs.get("lead_rel_speed") is None else round(obs.get("lead_rel_speed"),2),
                round(preds.get("p_lane_violation",0.0),3),
                round(preds.get("p_collision",0.0),3),
                round(preds.get("p_harsh_operation",0.0),3),
                self.state.notes
            ])

        # HUD hint (optional)
        if self.hud and hasattr(self.hud, "set_status_hint"):
            self.hud.set_status_hint(f"{self.name}: {self.state.phase}")

        # finish?
        if self.state.t >= self.duration_s and not self.state.finished:
            self.state.finished = True
        return {"preds": preds, "obs": obs, "phase": self.state.phase, "finished": self.state.finished}

    def teardown(self):
        for a in self.actors:
            try: a.destroy()
            except: pass
        self.actors.clear()
        if self._csv:
            try: self._csv.close()
            except: pass
        if self.orig_settings:
            try: self.world.apply_settings(self.orig_settings)
            except: pass

    # ---- to override ----
    def _on_setup(self): ...
    def _on_tick(self, obs: Dict, preds: Dict): ...

    # ---- helpers ----
    def _speed(self, v: carla.Vector3D) -> float:
        return (v.x*v.x + v.y*v.y + v.z*v.z)**0.5

    def _lane_frame(self):
        wp = self.world.get_map().get_waypoint(self.ego.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
        w = getattr(wp, "lane_width", 3.6)
        yaw = wp.transform.rotation.yaw * math.pi/180.0
        loc = self.ego.get_location()
        dx = loc.x - wp.transform.location.x
        dy = loc.y - wp.transform.location.y
        e_left = -math.sin(yaw)*dx + math.cos(yaw)*dy
        e_y = -e_left
        return e_y, float(w)

    def _observe(self) -> Dict:
        v = self._speed(self.ego.get_velocity())
        ax = self.ego.get_acceleration().x
        yaw_rate = self.ego.get_angular_velocity().z
        e_y, lane_w = self._lane_frame()
        d_lead, v_rel, blinker = None, None, None
        if self.lead and self.lead.is_alive:
            d_lead = self.lead.get_location().distance(self.ego.get_location())
            v_lead = self._speed(self.lead.get_velocity())
            v_rel = v_lead - v
        # (Blinker: if your code drives CARLA lights, you can pipe state here. Keep None for now.)
        return dict(v=v, ax=ax, yaw_rate=yaw_rate, lateral_offset=e_y, lane_width=lane_w,
                    blinker=blinker, lead_distance=d_lead, lead_rel_speed=v_rel)

    def _spawn_vehicle_ahead(self, dist_m=30.0, bp_name="vehicle.audi.tt") -> Optional[carla.Vehicle]:
        ego_tf = self.ego.get_transform()
        yaw = math.radians(ego_tf.rotation.yaw)
        offset = carla.Transform(
            carla.Location(
                x=ego_tf.location.x + dist_m*math.cos(yaw),
                y=ego_tf.location.y + dist_m*math.sin(yaw),
                z=ego_tf.location.z),
            ego_tf.rotation
        )
        bp = self.world.get_blueprint_library().find(bp_name)
        v = self.world.try_spawn_actor(bp, offset)
        if v:
            v.set_autopilot(True, self.tm_port)
            self.actors.append(v)
        return v

# ------------------ Scenarios ------------------

class ScenarioCollisionS1(InteractiveScenarioBase):
    """Follow a slower lead; lead performs surprise emergency brake. Avoid collision."""
    name = "Collisions Sc. 1"
    description = "Lead slows then brakes hard; maintain safe headway or brake in time."
    duration_s = 16.0

    def _on_setup(self):
        self.lead = self._spawn_vehicle_ahead(dist_m=35.0)
        self.state.phase = "closing_gap"
        # Make lead ~25% slower to build closure, then plan an emergency brake window
        if self.lead:
            tm = self.client.get_trafficmanager(self.tm_port)
            tm.vehicle_percentage_speed_difference(self.lead, +25)
        self._brake_at = random.uniform(6.0, 9.0)  # seconds into scenario

    def _on_tick(self, obs, preds):
        t = self.state.t
        tm = self.client.get_trafficmanager(self.tm_port)
        if self.lead:
            if t < self._brake_at:
                self.state.phase = "closing_gap"
            elif t < self._brake_at + 2.5:
                self.state.phase = "lead_emergency_brake"
                tm.vehicle_percentage_speed_difference(self.lead, +95)  # crawl
            else:
                self.state.phase = "recover"
                tm.vehicle_percentage_speed_difference(self.lead, +10)

        # simple success/violation signals in notes
        if preds.get("p_collision", 0.0) > 0.66:
            self.state.notes = "HIGH collision risk"
        else:
            self.state.notes = ""

class ScenarioLaneMgmtS1(InteractiveScenarioBase):
    """Keep lane center; perform exactly one signaled lane change; avoid unsignaled boundary crossing."""
    name = "Lane Management SC 1"
    description = "Hold center; one clean signaled lane change; no unsignaled boundary cross."
    duration_s = 18.0

    def _on_setup(self):
        self.state.phase = "center_hold"
        self._target_change_t = random.uniform(6.0, 9.0)  # when player should change lanes (we'll just prompt via HUD)
        if self.hud and hasattr(self.hud, "set_status_hint"):
            self.hud.set_status_hint("LaneMgmt: Hold center. At ~7s, make a signaled lane change, then recenter.")

    def _on_tick(self, obs, preds):
        t = self.state.t
        if t < self._target_change_t:
            self.state.phase = "center_hold"
        elif t < self._target_change_t + 4.0:
            self.state.phase = "lane_change_window"
        else:
            self.state.phase = "recenter"

        # note violations if lane risk spikes (unsignaled drift likely)
        self.state.notes = "Lane risk" if preds.get("p_lane_violation",0.0) > 0.5 else ""

class ScenarioDrivingBehaviorS1(InteractiveScenarioBase):
    """Harshness coaching: accelerate to ~12 m/s smoothly; brake to ~4 m/s; make a gentle left turn."""
    name = "Driving behavior sc 1"
    description = "Smooth accel to 12 m/s, smooth brake to 4 m/s, then a gentle left arc."
    duration_s = 16.0

    def _on_setup(self):
        self.state.phase = "smooth_accel"
        if self.hud and hasattr(self.hud, "set_status_hint"):
            self.hud.set_status_hint("Behavior: accelerate smoothly to ~12 m/s, brake to ~4 m/s, then gentle left.")

    def _on_tick(self, obs, preds):
        t = self.state.t
        v = obs.get("v",0.0)

        if t < 5.0:
            self.state.phase = "smooth_accel"
        elif t < 10.0:
            self.state.phase = "smooth_brake"
        elif t < 13.0:
            self.state.phase = "gentle_left"
        else:
            self.state.phase = "cruise"

        # coaching via notes
        if preds.get("p_harsh_operation",0.0) > 0.6:
            self.state.notes = f"Too harsh ({preds.get('dominant_axis','')})"
        else:
            self.state.notes = ""

class ScenarioOpenWorld(InteractiveScenarioBase):
    """Spawn general traffic/walkers and let the player roam."""
    name = "Open World Assessment"
    description = "Traffic + walkers; explore freely."
    duration_s = 9999.0

    def _on_setup(self):
        self.state.phase = "free_roam"
        tm = self.client.get_trafficmanager(self.tm_port)
        tm.set_global_distance_to_leading_vehicle(2.5)
        # If you have your own TrafficManager.py population util, call it here.
        # Otherwise, spawn a handful of random vehicles as a quick baseline:
        spawns = self.world.get_map().get_spawn_points()
        random.shuffle(spawns)
        for i in range(min(20, len(spawns) - 1)):
            bp = random.choice(self.world.get_blueprint_library().filter("vehicle.*"))
            v = self.world.try_spawn_actor(bp, spawns[i])
            if v: 
                v.set_autopilot(True, self.tm_port)
                self.actors.append(v)

    def _on_tick(self, obs, preds):
        self.state.notes = ""

# ------------------ Registry & facade ------------------

SCENARIO_REGISTRY = {
    "collisions_s1": ScenarioCollisionS1,
    "lane_mgmt_s1": ScenarioLaneMgmtS1,
    "driving_behavior_s1": ScenarioDrivingBehaviorS1,
    "open_world": ScenarioOpenWorld,
}

def scenario_display_name(sid: str) -> str:
    return SCENARIO_REGISTRY[sid].name

def create_scenario(sid: str, world, client, tm_port=8000, hud=None) -> InteractiveScenarioBase:
    if sid not in SCENARIO_REGISTRY:
        raise KeyError(f"Unknown scenario id '{sid}'. Available: {list(SCENARIO_REGISTRY)}")
    return SCENARIO_REGISTRY[sid](world, client, tm_port=tm_port, hud=hud)
