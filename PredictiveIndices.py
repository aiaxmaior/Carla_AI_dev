# PredictiveIndices.py
# ============================================================================
# PERF CHECK (file-level):
# ============================================================================
# [X] | Role: Predictive safety calculations (TLC, TTC, harsh operation risk)
# [X] | Hot-path functions: update() - called via PredictiveManager (every 10 frames)
# [X] |- Heavy allocs in hot path? Minimal - uses deque rolling buffers
# [ ] |- pandas/pyarrow/json/disk/net in hot path? No (pure math)
# [ ] | Graphics here? No
# [X] | Data produced (tick schema?): Predictive indices dict per update
# [ ] | Storage (Parquet/Arrow/CSV/none): None (consumed by HUD)
# [X] | Queue/buffer used?: YES - RollingSignal uses deque (efficient)
# [X] | Session-aware? No - rolling window state only
# [ ] | Debug-only heavy features?: None
# Top 3 perf risks:
# 1. [PERF_OK] update() throttled to every 10 frames by PredictiveManager - acceptable
# 2. [PERF_OK] deque operations are O(1) for append/popleft - efficient
# 3. [PERF_OK] Pure math (no CARLA queries, no I/O) - very fast
# ============================================================================

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Deque, Dict, Optional, Tuple
import DataIngestion

# ---------- Helpers ----------

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def safe_div(n: float, d: float, default: float = float("inf")) -> float:
    return n / d if abs(d) > 1e-8 else default

def logistic(x: float, x0: float, k: float) -> float:
    return 1.0 / (1.0 + math.exp(-k * (x - x0)))

def piecewise_linear(x: float, knots: Tuple[Tuple[float, float], ...]) -> float:
    if x <= knots[0][0]:
        return knots[0][1]
    if x >= knots[-1][0]:
        return knots[-1][1]
    for (x0, y0), (x1, y1) in zip(knots, knots[1:]):
        if x0 <= x <= x1:
            t = safe_div(x - x0, x1 - x0, 0.0)
            return y0 + t * (y1 - y0)
    return knots[-1][1]

# ---------- Config ----------

@dataclass
class PredictiveConfig:
    horizon_s: float = 2.0
    reaction_time_s: float = 0.8
    tlc_p1: float = 1.2
    tlc_p99: float = 0.4
    allow_signal_reduce: float = 0.35
    a_comf: float = 2.0
    a_max: float = 6.0
    min_headway_buffer_m: float = 2.0
    a_x_warn: float = 2.5
    a_x_harsh: float = 4.0
    jerk_warn: float = 3.0
    jerk_harsh: float = 6.0
    a_y_warn_g: float = 0.25
    a_y_harsh_g: float = 0.40
    v_lp_tau_s: float = 0.25
    default_lane_width_m: float = 3.6

# ---------- Rolling signals & filters ----------

@dataclass
class RollingSignal:
    window: int
    values: Deque[Tuple[float, float]] = field(default_factory=deque)
    def push(self, t: float, x: float) -> None:
        self.values.append((t, x))
        while self.values and (self.values[-1][0] - self.values[0][0]) > (self.window / 20.0):
            self.values.popleft()
    def last(self) -> Optional[Tuple[float, float]]:
        return self.values[-1] if self.values else None
    def derivative(self) -> float:
        if len(self.values) < 2:
            return 0.0
        (t0, x0), (t1, x1) = self.values[0], self.values[-1]
        dt = max(1e-6, t1 - t0)
        return (x1 - x0) / dt

@dataclass
class LowPass:
    tau_s: float
    y: float = 0.0
    init: bool = False
    def update(self, x: float, dt: float) -> float:
        if not self.init or dt <= 0.0:
            self.y = x
            self.init = True
            return self.y
        alpha = clamp(dt / max(1e-6, (self.tau_s + dt)), 0.0, 1.0)
        self.y = (1.0 - alpha) * self.y + alpha * x
        return self.y

# ---------- Main predictor ----------

class PredictiveIndices:
    """
    Outputs per-tick probabilities:
      - p_lane_violation: chance of unsignaled lane boundary crossing within horizon
      - p_collision: chance of frontal collision / emergency braking within horizon
      - p_harsh_operation: isolated harsh accel/brake/turn behavior
    """
    def __init__(self, cfg: PredictiveConfig = PredictiveConfig()):
        self.cfg = cfg
        self.ax_hist = RollingSignal(window=40)      # ~2s @ 20 Hz
        self.ay_hist = RollingSignal(window=40)
        self.v_hist  = RollingSignal(window=40)
        self.lat_off_hist = RollingSignal(window=40)
        self.yaw_rate_hist = RollingSignal(window=40)
        self.v_lp = LowPass(cfg.v_lp_tau_s)

    def update(self, t: float, obs: Dict) -> Dict[str, float]:
        if not isinstance(obs, dict):
            # If the input isn't a dict, assume it's a pandas Series 
            # (like the last row of a DataFrame) and convert it.
            obs = obs.to_dict()
        cfg = self.cfg
        v = float(obs.get("velocity", 0.0))

        ax = float(obs.get("acceleration_x", 0.0))

        r  = float(obs.get("yaw_rate", 0.0))

        e_y = float(obs.get("lateral_offset", 0.0))

        w_lane = float(obs.get("lane_width", cfg.default_lane_width_m) or cfg.default_lane_width_m)

        blinker = obs.get("blinker", None)

        d_lead = obs.get("lead_distance", None)

        v_rel = obs.get("lead_rel_speed", None)  # lead - ego

        # histories
        self.ax_hist.push(t, ax)

        self.ay_hist.push(t, v * r)       # a_y ≈ v * r

        self.v_hist.push(t, v)

        self.lat_off_hist.push(t, e_y)

        self.yaw_rate_hist.push(t, r)

        # filtered speed (for stability)
        v_f = self.v_lp.update(v, dt=1/20.0)

        # lateral velocity estimate
        v_lat = self.lat_off_hist.derivative()

        if abs(v_lat) < 1e-3 and abs(r) > 1e-3:
            v_lat = v_f * r * 0.5  # small-angle proxy

        # lateral accel from curvature or yaw rate
        if "curvature" in obs and obs["curvature"] is not None:
            a_y = (v_f ** 2) * float(obs["curvature"])
        else:
            a_y = v_f * r

        # --- Lane: Time-to-Lane-Crossing (TLC) ---
        half = 0.5 * w_lane

        dist_to_boundary = max(0.0, half - abs(e_y))

        if abs(v_lat) > 1e-3:
            tlc = dist_to_boundary / abs(v_lat)
        else:
            tlc = math.sqrt(max(0.0, 2.0 * dist_to_boundary / max(1e-6, abs(a_y)))) if abs(a_y) > 1e-4 else float("inf")

        if math.isfinite(tlc):
            tlc_knots = (
                (0.0, 1.0),
                (cfg.tlc_p99, 0.99),
                (cfg.tlc_p1,  0.5),
                (cfg.horizon_s, 0.05),
                (cfg.horizon_s * 3.0, 0.0),
            )
            p_lane = piecewise_linear(tlc, tlc_knots)
        else:
            p_lane = 0.0

        if blinker in ("left", "right"):
            p_lane *= (1.0 - cfg.allow_signal_reduce)

        # --- Collision: TTC + required decel ---
        p_col = 0.0
        ttc = float("inf")
        a_req = 0.0
        if d_lead is not None and v_rel is not None:
            closing = -float(v_rel)  # >0 means closing
            if closing > 1e-3:
                d_eff = max(0.1, float(d_lead) - max(0.0, v_f) * cfg.reaction_time_s - cfg.min_headway_buffer_m)
                ttc = safe_div(d_eff, closing, float("inf"))
                a_req = (closing ** 2) / max(0.1, 2.0 * d_eff)

                # logistic on required decel vs comfort/max
                if cfg.a_max > cfg.a_comf:
                    k = 3.0 / max(1e-6, (cfg.a_max - cfg.a_comf))
                else:
                    k = 2.5
                p_col_base = logistic(a_req, x0=cfg.a_comf, k=k)

                # TTC shaping inside horizon
                if math.isfinite(ttc):
                    ttc_factor = piecewise_linear(ttc, (
                        (0.0, 1.0),
                        (0.5, 0.9),
                        (1.0, 0.8),
                        (cfg.horizon_s, 0.5),
                        (cfg.horizon_s*2.0, 0.2),
                        (cfg.horizon_s*4.0, 0.05),
                    ))
                else:
                    ttc_factor = 0.0

                p_col = clamp(0.5 * p_col_base + 0.5 * ttc_factor, 0.0, 1.0)
            else:
                p_col = 0.0
                ttc = float("inf")
                a_req = 0.0

        # --- Operation: isolated harshness (accel/brake/turn) ---
        jerk_x = self.ax_hist.derivative()
        ax_mag = abs(ax)
        ax_risk = piecewise_linear(ax_mag, (
            (0.0, 0.0),
            (cfg.a_x_warn, 0.5),
            (cfg.a_x_harsh, 0.9),
            (cfg.a_x_harsh*1.5, 0.98),
        ))
        jerk_mag = abs(jerk_x)
        jerk_risk = piecewise_linear(jerk_mag, (
            (0.0, 0.0),
            (cfg.jerk_warn, 0.5),
            (cfg.jerk_harsh, 0.9),
            (cfg.jerk_harsh*1.5, 0.98),
        ))
        a_y_g = abs(a_y) / 9.80665
        lat_risk = piecewise_linear(a_y_g, (
            (0.0, 0.0),
            (cfg.a_y_warn_g, 0.5),
            (cfg.a_y_harsh_g, 0.9),
            (cfg.a_y_harsh_g*1.5, 0.98),
        ))
        # Noisy-OR combine
        r_list = [ax_risk, jerk_risk, lat_risk]
        p_harsh = 1.0
        for r_i in r_list:
            p_harsh *= (1.0 - clamp(r_i, 0.0, 1.0))
        p_harsh = 1.0 - p_harsh

        dominant_axis = max(
            [("accel/brake", ax_risk), ("jerk", jerk_risk), ("turning", lat_risk)],
            key=lambda kv: kv[1]
        )[0]

        return {
            "p_lane_violation": clamp(p_lane, 0.0, 1.0),
            "tlc_s": float(tlc if math.isfinite(tlc) else 99.0),
            "dist_to_lane_boundary_m": dist_to_boundary,
            "p_collision": clamp(p_col, 0.0, 1.0),
            "ttc_s": float(ttc if math.isfinite(ttc) else 99.0),
            "a_required_brake_mps2": a_req,
            "p_harsh_operation": clamp(p_harsh, 0.0, 1.0),
            "dominant_axis": dominant_axis,
            "speed_mps": v_f,
            "a_lat_mps2": a_y,
            "a_long_mps2": ax,
        }
