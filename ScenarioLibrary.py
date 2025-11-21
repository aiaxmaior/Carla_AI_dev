# ScenarioLibrary.py
# ============================================================================
# PERF CHECK (file-level):
# ============================================================================
# [X] | Role: Scenario config loader (NOT in hot path - init only)
# [ ] | Hot-path functions: None (called at startup/menu only)
# [ ] |- Heavy allocs in hot path? N/A
# [ ] |- pandas/pyarrow/json/disk/net in hot path? Disk reads at init only
# [ ] | Graphics here? No
# [ ] | Data produced (tick schema?): Config dicts
# [ ] | Storage (Parquet/Arrow/CSV/none): None (reads configs)
# [ ] | Queue/buffer used?: No
# [ ] | Session-aware? No
# [ ] | Debug-only heavy features?: None
# Top 3 perf risks:
# 1. [PERF_OK] NOT in hot path - menu/init only
# 2. [PERF_OK] Lightweight config loading
# ============================================================================

# import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class ScenarioLibrary:
    """
    Repository + helpers for scenario selection.

    - Scans ./configs/scenarios/ for optional per-scenario scripts (e.g., collisions_s1.py or .json)
    - Provides dropdown options in the exact (label, id) shape you were using
    - Supplies per-scenario help text identical to your previous mapping
    """

    def __init__(self, config_dir: str = "./configs/scenarios/") -> None:
        self.config_dir = Path(config_dir)

        # Keep the labels EXACT so the UI text doesn’t drift from your prior build.
        self._scenarios: List[Dict[str, str]] = [
            {"id": "collisions_s1",       "label": "Collisons Sc. 1"},
            {"id": "lane_mgmt_s1",        "label": "Lane Management SC 1"},
            {"id": "driving_behavior_s1", "label": "Driving behavior sc 1"},
            {"id": "open_world",          "label": "Open World Assessment"},
        ]

        # Same help strings you rendered in the consolidated menu.
        self._help: Dict[str, str] = {
            "collisions_s1":       "Follow → surprise lead brake. Avoid collision.",
            "lane_mgmt_s1":        "Hold center; one signaled lane change.",
            "driving_behavior_s1": "Smooth accel/brake; gentle turn.",
            "open_world":          "Free roam with traffic.",
        }

        # Files we’ll look for (first-hit wins)
        self._candidates = (".py", ".json", ".yaml", ".yml")

    # ---------- Public API ----------

    def get_dropdown_options(self, default_sid: Optional[str] = None) -> Tuple[List[Tuple[str, str]], int]:
        """
        Returns: (options, idx)
          options -> [(label, id), ...] in the fixed order shown above
          idx     -> default selection index (defaults to 'open_world' if not found)
        """
        options = [(s["label"], s["id"]) for s in self._scenarios]
        # default is "open_world" (index 3 in our fixed list), unless caller matches another id
        idx = next((i for i, (_, sid) in enumerate(options) if sid == (default_sid or "open_world")), 3)
        return options, idx

    def list_display_items(self) -> List[Dict[str, Optional[str]]]:
        """
        For parity with your VehicleLibrary style:
        Returns a list of dicts with fields useful for menus or scenario launchers.
        Each dict: {"id", "name", "help", "path"}
        - "path" points to the first existing file in ./configs/scenarios/ matching the id with a known extension.
        """
        items: List[Dict[str, Optional[str]]] = []
        for s in self._scenarios:
            sid = s["id"]
            items.append({
                "id": sid,
                "name": s["label"],
                "help": self._help.get(sid, ""),
                "path": self.get_runner_path(sid),
            })
        return items

    def get_help_text(self, scenario_id: str) -> str:
        """Returns the single-line description used under the dropdown."""
        return self._help.get(scenario_id, "")

    def resolve_to_id(self, value: Optional[str]) -> Optional[str]:
        """
        Accepts either an id or a display label (case-insensitive) and returns the canonical id.
        Returns None if no match.
        """
        if not value:
            return None
        v = value.strip().lower()
        # direct id match
        for s in self._scenarios:
            if v == s["id"].lower():
                return s["id"]
        # label match
        for s in self._scenarios:
            if v == s["label"].strip().lower():
                return s["id"]
        # loose “open world” alias
        if v in ("open world", "free roam", "open_world"):
            return "open_world"
        return None

    def get_runner_path(self, scenario_id: str) -> Optional[str]:
        """
        Finds a scenario script/config under ./configs/scenarios/.
        Looks for: <id>.py / <id>.json / <id>.yaml / <id>.yml (in that order).
        Returns absolute path string, or None if nothing exists (e.g., open_world).
        """
        if not self.config_dir.exists():
            logging.debug(f"[ScenarioLibrary] Config dir not found: {self.config_dir}")
            return None
        for ext in self._candidates:
            p = (self.config_dir / f"{scenario_id}{ext}")
            if p.exists():
                try:
                    return str(p.resolve())
                except Exception:
                    return str(p)  # relative fallback
        return None
