# vehicle_library.py
import os, json, glob
from typing import Dict, Any, Optional, List
from vehicle_config import VehicleConfig

DEFAULT_SEARCH_PATHS = ["./configs/vehicles", "./config/vehicles"]
ENV_VAR = "VEHICLE_LIBRARY_PATH"

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
        paths = []
        env = os.environ.get(ENV_VAR, "")
        if env: paths.extend([p for p in env.split(os.pathsep) if p])
        if search_paths: paths.extend(search_paths)
        if not paths: paths = DEFAULT_SEARCH_PATHS
        self.search_paths = [os.path.abspath(p) for p in paths]
        self._index = self._build_index()

    def _build_index(self) -> Dict[str, str]:
        idx: Dict[str, str] = {}
        for root in self.search_paths:
            for path in glob.glob(os.path.join(root, "*.json")):
                try:
                    with open(path, "r") as f: data = json.load(f)
                    vid = data.get("id")
                    if vid: idx[vid] = path
                    for alias in data.get("aliases", []):
                        idx[alias] = path
                except Exception:
                    continue
        return idx

    def list_ids(self) -> List[str]:
        return sorted(set(self._index.keys()))

    def _load_raw(self, id_or_path: str, _seen=None) -> Dict[str, Any]:
        if os.path.isfile(id_or_path): path = id_or_path
        else:
            path = self._index.get(id_or_path)
            if not path: raise FileNotFoundError(f"Vehicle blueprint not found: {id_or_path}")
        with open(path, "r") as f: data = json.load(f)
        parent = data.get("extends")
        if parent:
            _seen = _seen or set()
            if parent in _seen: raise ValueError("Cyclic extends in vehicle configs")
            _seen.add(parent)
            base = self._load_raw(parent, _seen=_seen)
            data = _deep_merge(base, data)
        return data

    def load(self, id_or_path: str) -> VehicleConfig:
        raw = self._load_raw(id_or_path)
        return VehicleConfig.from_dict(raw)
