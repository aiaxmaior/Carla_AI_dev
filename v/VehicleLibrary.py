# VehicleLibrary.py
import os
import json
import glob
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

DEFAULT_SEARCH_PATHS = ["./configs/vehicles", "./config/vehicles"]
ENV_VAR = "VEHICLE_LIBRARY_PATH"

# ---------------- VehicleConfig dataclass (moved here) ----------------
@dataclass
class VehicleConfig:
    # Required core statics
    id: str
    carla_blueprint: str
    mass: float
    wheelbase: float
    track_front: float
    track_rear: float
    cg_height: float
    front_static_ratio: float
    # Common optionals
    friction_mu: float = 0.9
    steer_cap_deg: Optional[float] = None
    default_front_stiffness_per_tire: float = 70000.0
    default_rear_stiffness_per_tire: float = 80000.0

    # NEW: let JSON carry a friendly name without breaking ctor
    display_name: Optional[str] = None

    # Bag for anything else in the JSON
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VehicleConfig":
        known_keys = {
            "id", "display_name","carla_blueprint","mass", "wheelbase", "track_front", "track_rear",
            "cg_height", "front_static_ratio", "friction_mu",
            "steer_cap_deg", "default_front_stiffness_per_tire",
            "default_rear_stiffness_per_tire"
        }
        extras = {k: v for k, v in d.items() if k not in known_keys}
        return cls(
            id=str(d["id"]),
            carla_blueprint=str(d["carla_blueprint"]),
            display_name=d.get("display_name"),   # <- was d["display_name"]
            mass=float(d["mass"]),
            wheelbase=float(d["wheelbase"]),
            track_front=float(d["track_front"]),
            track_rear=float(d["track_rear"]),
            cg_height=float(d["cg_height"]),
            front_static_ratio=float(d["front_static_ratio"]),
            friction_mu=float(d.get("friction_mu", 0.9)),
            steer_cap_deg=(None if d.get("steer_cap_deg") in (None, "")
                        else float(d.get("steer_cap_deg"))),
            default_front_stiffness_per_tire=float(d.get("default_front_stiffness_per_tire", 70000.0)),
            default_rear_stiffness_per_tire=float(d.get("default_rear_stiffness_per_tire", 80000.0)),
            extras=extras
        )

    # Small helper so you can read optional fields uniformly
    def get(self, key: str, default=None):
        return getattr(self, key, self.extras.get(key, default))

# -------------------------- Library loader ---------------------------
def _deep_merge(base: Dict[str, Any], child: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in child.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

class VehicleLibrary:
    def __init__(self, search_paths: Optional[List[str]] = None):
        paths: List[str] = []
        env = os.environ.get(ENV_VAR, "")
        if env:
            paths.extend([p for p in env.split(os.pathsep) if p])
        if search_paths:
            paths.extend(search_paths)
        if not paths:
            paths = DEFAULT_SEARCH_PATHS
        self.search_paths = [os.path.abspath(p) for p in paths]
        self._index = self._build_index()
    
    def list_display_items(self) -> List[Dict[str, str]]:
        """
        Returns a de-duplicated list of {'id': <primary id>, 'name': <display name or id>}
        sorted by display name (case-insensitive). Each config file appears once,
        even if it has many aliases.
        """
        items: List[Dict[str, str]] = []
        seen_paths = set()

        # unique JSON paths (avoid duplicates from aliases)
        for path in sorted(set(self._index.values())):
            if path in seen_paths:
                continue
            seen_paths.add(path)
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                vid = str(data["id"])
                name = data.get("display_name") or vid
                c_bp = str(data["carla_blueprint"])
                items.append({"id": vid, "name": name, "carla_blueprint":c_bp})
            except Exception:
                continue

        items.sort(key=lambda x: x["name"].lower())
        return items

    def resolve_to_id(self, id_or_alias: str) -> str:
        """
        Resolve an id or any alias to the primary id in the JSON.
        """
        path = self._index.get(id_or_alias)
        if not path:
            return id_or_alias
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return str(data.get("id", id_or_alias))
        except Exception:
            return id_or_alias
        
    def _build_index(self) -> Dict[str, str]:
        idx: Dict[str, str] = {}
        for root in self.search_paths:
            for path in glob.glob(os.path.join(root, "*.json")):
                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                    vid = data.get("id")
                    if vid:
                        idx[vid] = path
                    for alias in data.get("aliases", []) or []:
                        idx[alias] = path
                except Exception:
                    continue
        return idx

    def list_ids(self) -> List[str]:
        return sorted(set(self._index.keys()))

    def _load_raw(self, id_or_path: str, _seen: Optional[set] = None) -> Dict[str, Any]:
        if os.path.isfile(id_or_path):
            path = id_or_path
        else:
            path = self._index.get(id_or_path)
            if not path:
                raise FileNotFoundError(f"Vehicle blueprint not found: {id_or_path}")
        with open(path, "r") as f:
            data = json.load(f)

        parent = data.get("extends")
        if parent:
            _seen = _seen or set()
            if parent in _seen:
                raise ValueError("Cyclic extends in vehicle configs")
            _seen.add(parent)
            base = self._load_raw(parent, _seen=_seen)
            data = _deep_merge(base, data)
        return data

    def load(self, id_or_path: str) -> VehicleConfig:
        raw = self._load_raw(id_or_path)
        return VehicleConfig.from_dict(raw)

class MapLibrary:

    def __init__(self, search_paths: Optional[List[str]] = None):
        if search_paths:
            self._paths=search_paths
        else:
            self._paths = "./CarlaUE4/Content/Carla/Maps"
        
    def list_display_items(self) -> List[Dict[str, str]]:
        """
        Returns a de-duplicated list of {'id': <primary id>, 'name': <display name or id>}
        sorted by display name (case-insensitive). Each config file appears once,
        even if it has many aliases.
        """
        items: List[Dict[str, str]] = []
        seen_paths = set()

        # unique JSON paths (avoid duplicates from aliases)
        for path in sorted(set(self._index.values())):
            if path in seen_paths:
                continue
            seen_paths.add(path)
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                vid = str(data["id"])
                name = data.get("display_name") or vid
                c_bp = str(data["carla_blueprint"])
                items.append({"id": vid, "name": name, "carla_blueprint":c_bp})
            except Exception:
                continue

    def _build_index(self):
        return 