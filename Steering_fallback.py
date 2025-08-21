# Steering.py — Custom steering dynamics helpers for CARLA 0.9.16
# Focus: vehicle geometry, axle loads/CG, stiffness & understeer, dynamic state,
# curvature↔steer, Ackermann angles, friction-limited caps, shaping, and smoothing.

from dataclasses import dataclass
from typing import Iterable, Tuple, Optional
import math

G = 9.81

# ---------------------------- Data models ----------------------------

@dataclass
class WheelSpec:
    x: float                  # vehicle-frame x (m); + forward
    y: float                  # vehicle-frame y (m); + right
    is_front: bool
    lateral_stiffness: Optional[float] = None  # N/rad (per tire)

@dataclass
class VehicleBasics:
    mass: float              # kg
    L: float                 # wheelbase (m)
    T_front: float           # front track (m)
    T_rear: float            # rear track (m)
    h_cg: float              # CG height (m)
    a: float                 # CG→front axle (m)
    b: float                 # CG→rear axle (m)
    Cf: float                # front axle cornering stiffness (N/rad)
    Cr: float                # rear axle cornering stiffness (N/rad)
    Iz: Optional[float] = None

@dataclass
class DynamicState:
    V: float = 0.0           # speed magnitude (m/s)
    v_long: float = 0.0      # body x (m/s)
    u_lat: float = 0.0       # body y (m/s, + to the right)
    omega: float = 0.0       # yaw rate (rad/s, + CCW)
    beta: float = 0.0        # slip angle (rad) = atan(u/v)
    ax: float = 0.0          # longitudinal accel (m/s^2)
    ay: float = 0.0          # lateral accel (m/s^2)
    jerk: float = 0.0        # d(ax)/dt (m/s^3)

# ---------------------- Geometry & mass properties -------------------

def derive_geometry_from_wheels(wheels: Iterable[WheelSpec]) -> Tuple[float, float, float]:
    f = [w for w in wheels if w.is_front]
    r = [w for w in wheels if not w.is_front]
    if len(f) < 2 or len(r) < 2:
        raise ValueError("Need at least two front and two rear wheels.")
    xf = sum(w.x for w in f) / len(f); xr = sum(w.x for w in r) / len(r)
    L = abs(xf - xr)

    def track(ws):
        ys = sorted(w.y for w in ws)
        return abs(ys[-1] - ys[0])

    return L, track(f), track(r)

def static_axle_loads(mass: float, front_static_ratio: float, g: float = G) -> Tuple[float, float]:
    W = mass * g
    Fzf0 = W * front_static_ratio
    Fzr0 = W - Fzf0
    return Fzf0, Fzr0

def cg_from_axle_loads(L: float, Fzf0: float, Fzr0: float) -> Tuple[float, float]:
    W = Fzf0 + Fzr0
    if W <= 0: raise ValueError("Total weight must be positive.")
    b = (Fzf0 / W) * L
    a = L - b
    return a, b

def estimate_Iz_solid(mass: float, L: float, T_front: float, T_rear: float, k_y: float = 0.28) -> float:
    T_avg = 0.5 * (T_front + T_rear)
    Ry = k_y * math.hypot(L, T_avg)
    return mass * (Ry ** 2)

# ------------------- Cornering stiffness & understeer ----------------

def aggregate_cornering_stiffness(
    wheels: Iterable[WheelSpec],
    default_front_per_tire: float = 70000.0,
    default_rear_per_tire: float = 80000.0
) -> Tuple[float, float]:
    Cf = 0.0; Cr = 0.0
    for w in wheels:
        c = w.lateral_stiffness
        if c is None:
            c = default_front_per_tire if w.is_front else default_rear_per_tire
        if w.is_front: Cf += c
        else:          Cr += c
    return Cf, Cr

def understeer_gradient(a: float, b: float, Cf: float, Cr: float,
                        L: Optional[float] = None,
                        Fzf0: Optional[float] = None,
                        Fzr0: Optional[float] = None,
                        method: str = "a_over_Cf_minus_b_over_Cr") -> float:
    if method == "a_over_Cf_minus_b_over_Cr":
        return (a / max(Cf, 1e-6)) - (b / max(Cr, 1e-6))
    if method == "weight_based":
        if L is None or Fzf0 is None or Fzr0 is None:
            raise ValueError("Weight-based Ku requires L, Fzf0, Fzr0")
        Wf = Fzf0; Wr = Fzr0
        return ((Wf / max(Cf, 1e-6)) - (Wr / max(Cr, 1e-6))) / max(L, 1e-6)
    raise ValueError("Unknown method for Ku")

# -------------------------- Load transfer ----------------------------

def longitudinal_transfer(mass: float, ax: float, h_cg: float, L: float) -> Tuple[float, float]:
    dF = mass * ax * h_cg / max(L, 1e-6)
    return dF, -dF  # front, rear

def lateral_transfer_per_axle(
    mass: float, ay: float, h_cg: float, T_front: float, T_rear: float, roll_stiff_split: float = 0.5
) -> Tuple[float, float, float, float]:
    total = mass * ay * h_cg
    front_total = total * roll_stiff_split
    rear_total  = total - front_total
    dF_front_each = front_total / max(T_front, 1e-6)
    dF_rear_each  = rear_total  / max(T_rear,  1e-6)
    # Return magnitudes; caller decides outside/inside sign.
    return dF_front_each, dF_front_each, dF_rear_each, dF_rear_each

# --------------------- Curvature & steering angles -------------------

def friction_limited_curvature(V: float, mu: float, g: float = G) -> float:
    if V <= 0.1: return float('inf')
    return (mu * g) / (V * V)

def curvature_from_radius(R: float) -> float:
    return 1.0 / max(R, 1e-6)

def radius_from_curvature(kappa: float) -> float:
    if abs(kappa) < 1e-9: return float('inf')
    return 1.0 / kappa

def delta_from_curvature(V: float, kappa: float, L: float, Ku: float = 0.0) -> float:
    return math.atan(L * kappa) + Ku * (V * V) * kappa

def curvature_from_avg_steer(delta_avg: float, L: float, Ku: float = 0.0, V: float = 0.0) -> float:
    k = math.tan(delta_avg) / max(L, 1e-6)
    for _ in range(6):
        f  = math.atan(L * k) + Ku * (V * V) * k - delta_avg
        df = (L / (1 + (L * k) ** 2)) + Ku * (V * V)
        k -= f / max(df, 1e-9)
    return k

def ackermann_wheel_angles_from_curvature(L: float, T_front: float, kappa: float) -> Tuple[float, float]:
    R = radius_from_curvature(kappa)
    inner = math.atan(L / max(R - T_front / 2.0, 1e-6))
    outer = math.atan(L / max(R + T_front / 2.0, 1e-6))
    return inner, outer

def ackermann_from_delta_avg(L: float, T_front: float, delta_avg: float) -> Tuple[float, float]:
    # Convert δ_avg → κ → inner/outer
    kappa = curvature_from_avg_steer(delta_avg, L)
    return ackermann_wheel_angles_from_curvature(L, T_front, kappa)

def max_delta_from_Rmin(L: float, R_min: float) -> float:
    return math.atan(L / max(R_min, 1e-6))

# ----------------------- Command shaping utils ----------------------

def normalize_to_carla_steer(delta_avg: float, delta_avg_max: float) -> float:
    if delta_avg_max <= 1e-6: return 0.0
    s = delta_avg / delta_avg_max
    return max(-1.0, min(1.0, s))

def speed_gain_schedule(V: float, ratio_lo: float = 10.0, ratio_hi: float = 18.0,
                        V_mid: float = 13.4, sharp: float = 2.2) -> float:
    t = 1.0 / (1.0 + math.exp(-sharp * (V - V_mid)))
    ratio = ratio_lo + (ratio_hi - ratio_lo) * t
    return ratio_lo / ratio  # multiplicative gain

def shape_center_zone(x: float, deadzone: float = 0.02, linear_window: float = 0.15, expo: float = 1.8) -> float:
    if abs(x) <= deadzone: return 0.0
    x = math.copysign(max(0.0, abs(x) - deadzone) / max(1e-6, 1.0 - deadzone), x)
    if abs(x) <= linear_window: return x
    return math.copysign((abs(x) ** expo), x)

def rate_limit(prev: float, target: float, dt: float, max_rate_per_s: float) -> float:
    max_step = max_rate_per_s * max(dt, 1e-3)
    return prev + max(-max_step, min(max_step, target - prev))

def lowpass(prev: float, target: float, alpha: float) -> float:
    alpha = max(0.0, min(1.0, alpha))
    return prev + alpha * (target - prev)

# -------------------------- Dynamic state ---------------------------

def world_to_body_vel(vx: float, vy: float, yaw_rad: float) -> Tuple[float, float]:
    """
    Rotate world-plane velocity into vehicle body frame.
    CARLA: x forward, y right, yaw about +z (degrees in API).
    """
    cy = math.cos(yaw_rad); sy = math.sin(yaw_rad)
    v_long =  cy * vx + sy * vy
    u_lat  = -sy * vx + cy * vy
    return v_long, u_lat

def update_dynamic_state(
    vx: float, vy: float, yaw_rad: float, omega_z: float, dt: float, prev: Optional[DynamicState] = None
) -> DynamicState:
    v_long, u_lat = world_to_body_vel(vx, vy, yaw_rad)
    V = math.hypot(vx, vy)
    beta = math.atan2(u_lat, max(1e-6, v_long))
    # Approximate lateral accel from kinematics; longitudinal from speed diff
    ay = V * omega_z  # small-slip approx
    if prev is None:
        ax = 0.0; jerk = 0.0
    else:
        ax = (V - prev.V) / max(dt, 1e-3)
        jerk = (ax - prev.ax) / max(dt, 1e-3)
    return DynamicState(V=V, v_long=v_long, u_lat=u_lat, omega=omega_z, beta=beta, ax=ax, ay=ay, jerk=jerk)

def slip_ratio_total(mass: float, ay: float, Cf: float, Cr: float, Fzf0: float, Fzr0: float) -> float:
    """
    Very simple aggregate 'slip ratio' proxy: Fy_total / (Cα_total * Fz_norm)
    Not a true tire model; useful as a sanity index for limiting/safety.
    """
    Fy = mass * ay
    Ctot = max(Cf + Cr, 1e-6)
    Fz  = max(Fzf0 + Fzr0, 1e-6)
    return (Fy / Ctot) / Fz

# -------------------- One-stop steering synthesis -------------------

def synthesize_normalized_steer(
    hw_steer: float,                 # raw hardware ∈ [-1,1]
    V: float,                        # m/s
    basics: VehicleBasics,
    mu: float,                       # tire-road friction
    delta_avg_max_control: float,    # rad (software/physics cap)
    dt: float,
    prev_cmd: float,
    deadzone: float = 0.02,
    linear_window: float = 0.12,
    expo: float = 1.6,
    max_rate_per_s: float = 2.5,
    smooth_alpha: float = 0.25,
    Ku: Optional[float] = None
) -> float:
    shaped = shape_center_zone(hw_steer, deadzone, linear_window, expo)
    shaped *= speed_gain_schedule(V)

    # Map shaped control to a curvature command within the friction envelope
    kappa_limit = friction_limited_curvature(V, mu)
    kappa_cmd = shaped * kappa_limit

    Ku_val = (Ku if Ku is not None else understeer_gradient(basics.a, basics.b, basics.Cf, basics.Cr))
    delta_avg = delta_from_curvature(V, kappa_cmd, basics.L, Ku_val)

    # Respect control/physics cap (rad)
    delta_avg = max(-delta_avg_max_control, min(delta_avg_max_control, delta_avg))

    # Convert road-wheel angle to CARLA control.steer ∈ [-1,1]
    s_norm = normalize_to_carla_steer(delta_avg, delta_avg_max_control)

    # Rate limiting and smoothing for stability
    s_norm = rate_limit(prev_cmd, s_norm, dt, max_rate_per_s)
    s_norm = lowpass(prev_cmd, s_norm, smooth_alpha)
    return s_norm
