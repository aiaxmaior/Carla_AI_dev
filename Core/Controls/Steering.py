# Steering.py  — physics-informed steer curve, CARLA-agnostic (0.9.16)
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
import math

_G = 9.81

@dataclass
class VehicleParams:
    # Geometry
    wheelbase: float
    track_front: float
    track_rear: float

    # Limits
    vehicle_max_steer_deg: float

    # Mass / load split
    mass: float = 3000.0
    front_static_ratio: float = 0.5

    # Tire/handling
    mu: float = 0.9
    Cf: float = 160000.0         # [N/rad] front axle cornering stiffness
    Cr: float = 170000.0         # [N/rad] rear axle cornering stiffness

    # Dynamics shaping
    base_tau_s: float = 0.25     # first-order lag time constant (low speed)
    max_steer_rate_dps: float = 360.0  # road-wheel slew rate

    # User steer curve table: list of (v_ms, deg) sorted by v
    steer_table: Optional[List[Tuple[float, float]]] = None

@dataclass
class SteeringModel:
    p: VehicleParams
    _delta_deg_prev: float = field(default=0.0, init=False)

    # ---------------- Construction ----------------
    @classmethod
    def from_vehicle_and_config(
        cls,
        *,
        wheelbase: float,
        track_front: float,
        track_rear: float,
        mass: float,
        front_static_ratio: float,
        vehicle_max_steer_deg: float,
        mu: float = 0.9,
        Cf: Optional[float] = None,
        Cr: Optional[float] = None,
        base_tau_s: float = 0.25,
        max_steer_rate_dps: float = 360.0,
        steer_curve: Optional[dict] = None,
    ) -> "SteeringModel":
        # normalize optional table to (m/s, deg)
        table_ms = None
        if steer_curve:
            try:
                units = (steer_curve.get("units") or "kph").lower()
                pts = steer_curve.get("points") or steer_curve.get("table") or []
                raw = []
                for pt in pts:
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        v, deg = float(pt[0]), float(pt[1])
                    else:
                        v, deg = float(pt.get("v")), float(pt.get("deg"))
                    v_ms = v / 3.6 if units in ("kph", "kmh") else v
                    raw.append((max(0.0, v_ms), max(0.0, deg)))
                raw.sort(key=lambda x: x[0])
                # dedup
                table_ms = []
                last_v = None
                for v_ms, deg in raw:
                    if last_v is None or v_ms > last_v + 1e-6:
                        table_ms.append((v_ms, deg))
                        last_v = v_ms
            except Exception:
                table_ms = None

        p = VehicleParams(
            wheelbase=wheelbase,
            track_front=track_front,
            track_rear=track_rear,
            vehicle_max_steer_deg=vehicle_max_steer_deg,
            mass=mass,
            front_static_ratio=front_static_ratio,
            mu=mu,
            Cf=(Cf if Cf is not None else 160000.0),
            Cr=(Cr if Cr is not None else 170000.0),
            base_tau_s=base_tau_s,
            max_steer_rate_dps=max_steer_rate_dps,
            steer_table=table_ms,
        )
        return cls(p=p)

    # ---------------- Public API ----------------
    def compute(self, driver_input: float, speed_ms: float, yaw_rate: float, dt: float) -> float:
        """
        Input: normalized driver input in [-1,1], vehicle speed [m/s], yaw_rate [rad/s], dt [s]
        Output: target front road-wheel angle [deg], signed.
        Pipeline:
          - allowed angle from min(physics, user table, physical cap)
          - driver scaling
          - understeer gradient shaping
          - first-order lag + slew limit
        """
        u = max(-1.0, min(1.0, float(driver_input)))
        allowed_deg = self._allowed_wheel_angle_deg(speed_ms)
        delta_cmd = u * allowed_deg
        delta_cmd = self._apply_understeer(delta_cmd, speed_ms)
        delta_resp = self._first_order_lag(self._delta_deg_prev, delta_cmd, speed_ms, dt)
        delta_rate = self._rate_limit(self._delta_deg_prev, delta_resp, dt)
        out = self._clip_deg(delta_rate, self.p.vehicle_max_steer_deg)
        self._delta_deg_prev = out
        return out

    def to_normalized(self, wheel_angle_deg: float) -> float:
        m = max(1e-6, self.p.vehicle_max_steer_deg)
        return max(-1.0, min(1.0, float(wheel_angle_deg) / m))

    def ackermann_wheel_angles(self, delta_deg: float, front: bool = True) -> Tuple[float, float]:
        T = self.p.track_front if front else self.p.track_rear
        L = self.p.wheelbase
        sgn = 1.0 if delta_deg >= 0.0 else -1.0
        kappa = math.tan(math.radians(abs(delta_deg))) / max(L, 1e-6)
        if kappa < 1e-9:
            return (0.0, 0.0)
        R = 1.0 / kappa
        Ri = max(1e-6, R - T * 0.5)
        Ro = max(1e-6, R + T * 0.5)
        return (math.degrees(math.atan2(L, Ri)) * sgn,
                math.degrees(math.atan2(L, Ro)) * sgn)

    # ---------------- Internals ----------------
    def _allowed_wheel_angle_deg(self, v: float) -> float:
        L = max(1e-6, self.p.wheelbase)
        max_deg = max(1e-6, self.p.vehicle_max_steer_deg)

        # (1) Geometry: kappa ≤ tan(max_deg)/L
        kappa_geom = math.tan(math.radians(max_deg)) / L

        # (2) Friction: a_y = v^2 kappa ≤ mu g  →  kappa ≤ mu g / v^2
        if v <= 0.1:
            kappa_fric = 1e6
        else:
            kappa_fric = (self.p.mu * _G) / (v * v)

        phys_kappa = min(kappa_geom, kappa_fric)
        phys_deg = math.degrees(math.atan(L * phys_kappa))

        # (3) User table (if provided)
        tbl_deg = self._table_allowed_deg(v)

        # Final allowed angle
        return max(0.0, min(max_deg, phys_deg, tbl_deg))

    def _table_allowed_deg(self, v: float) -> float:
        tbl = self.p.steer_table
        if not tbl:
            return float("inf")
        if v <= tbl[0][0]:
            return tbl[0][1]
        if v >= tbl[-1][0]:
            return tbl[-1][1]
        for i in range(1, len(tbl)):
            v0, d0 = tbl[i-1]; v1, d1 = tbl[i]
            if v0 <= v <= v1:
                t = (v - v0) / max(1e-9, (v1 - v0))
                return d0 + t * (d1 - d0)
        return tbl[-1][1]

    def _apply_understeer(self, delta_deg: float, v: float) -> float:
        L = max(1e-6, self.p.wheelbase)
        b = self.p.front_static_ratio * L
        a = L - b
        Cf = max(1e-3, self.p.Cf)
        Cr = max(1e-3, self.p.Cr)
        K = max(0.0, (a / Cf) - (b / Cr))     # simple understeer gradient proxy
        k_cmd = math.tan(math.radians(delta_deg)) / L
        k_out = k_cmd / (1.0 + K * (v * v))
        return math.degrees(math.atan(L * k_out))

    def _first_order_lag(self, prev_deg: float, cmd_deg: float, v: float, dt: float) -> float:
        if dt <= 0.0:
            return cmd_deg
        tau = self.p.base_tau_s / (1.0 + max(0.0, v) / 5.0)
        alpha = max(0.0, min(1.0, dt / (tau + dt)))
        return prev_deg + alpha * (cmd_deg - prev_deg)

    def _rate_limit(self, prev_deg: float, cmd_deg: float, dt: float) -> float:
        if dt <= 0.0:
            return cmd_deg
        max_step = self.p.max_steer_rate_dps * dt
        if cmd_deg > prev_deg + max_step:
            return prev_deg + max_step
        if cmd_deg < prev_deg - max_step:
            return prev_deg - max_step
        return cmd_deg

    @staticmethod
    def _clip_deg(x: float, max_abs: float) -> float:
        m = float(max_abs)
        return max(-m, min(m, float(x)))