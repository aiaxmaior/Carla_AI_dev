# Tools/monitor_layout_fixed.py
# -------------------------------------------------------------------------
# Zero-fantasy monitor layout helpers for Linux (xrandr + optional NVIDIA).
# Keeps PHYSICAL modes intact; uses LOGICAL sizes only for your window math.
# -------------------------------------------------------------------------

import re
import shlex
import subprocess
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

__all__ = [
    "MonitorInfo",
    "probe_monitors",
    "arrange_horizontally",
    "arrange_panoramic_quadh",
    "restore_layout",
    "set_nvidia_viewport",
]

# --------------------------- Data structures -----------------------------------

@dataclass
class MonitorInfo:
    name: str
    connected: bool
    primary: bool
    physical_mode: str          # e.g., "5120x1440"
    physical_w: int
    physical_h: int
    position: Tuple[int, int]   # (x, y)
    modes: List[str] = field(default_factory=list)

    # saved for restore
    original_mode: str = ""
    original_pos: Tuple[int, int] = (0, 0)
    original_primary: bool = False

    # logical size (for app/window math only)
    logical_w: Optional[int] = None
    logical_h: Optional[int] = None

    def supports_mode(self, mode: str) -> bool:
        return mode in self.modes

# --------------------------- Shell helpers -------------------------------------

def _run(cmd: List[str], check: bool = False, text: bool = True) -> subprocess.CompletedProcess:
    # Avoid mixing capture_output with explicit pipes
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=text, check=check)

# --------------------------- Probe ---------------------------------------------

def _xrandr_query() -> str:
    res = _run(["xrandr", "--query"])
    if res.returncode != 0:
        raise RuntimeError(f"xrandr --query failed:\n{res.stderr}")
    return res.stdout

def _parse_block(name: str, block: str) -> MonitorInfo:
    # header like:
    #   DP-1 connected primary 5120x1440+0+0 (normal ...) 597mm x 336mm
    header = block.splitlines()[0]
    connected = " connected" in header
    primary = (" connected primary " in header) or header.strip().startswith(f"{name} connected primary")

    m_cur = re.search(r"(\d{3,5})x(\d{3,5})\+(-?\d+)\+(-?\d+)", header)
    if m_cur:
        w, h, x, y = map(int, m_cur.groups())
        mode_str = f"{w}x{h}"
    else:
        w = h = x = y = 0
        mode_str = "0x0"

    modes: List[str] = []
    for line in block.splitlines()[1:]:
        m = re.match(r"\s*(\d{3,5}x\d{3,5})\b", line)
        if m:
            modes.append(m.group(1))

    info = MonitorInfo(
        name=name,
        connected=connected,
        primary=primary,
        physical_mode=mode_str,
        physical_w=w,
        physical_h=h,
        position=(x, y),
        modes=modes,
    )
    info.original_mode = info.physical_mode
    info.original_pos = info.position
    info.original_primary = info.primary
    info.logical_w = w or None
    info.logical_h = h or None
    return info

def probe_monitors() -> Dict[str, MonitorInfo]:
    """Return { output_name: MonitorInfo } for connected outputs (physical state only)."""
    q = _xrandr_query()
    blocks = re.split(r"(?=^[A-Za-z0-9-]+\s+(?:connected|disconnected))", q, flags=re.M)
    monitors: Dict[str, MonitorInfo] = {}
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        m_name = re.match(r"^([A-Za-z0-9-]+)\s+(?:connected|disconnected)", b)
        if not m_name:
            continue
        name = m_name.group(1)
        info = _parse_block(name, b)
        if info.connected:
            monitors[name] = info
    return monitors

# --------------------------- xrandr command builder -----------------------------

def _xrandr_set_output(
    name: str,
    mode: Optional[str] = None,
    pos: Optional[Tuple[int, int]] = None,
    primary: Optional[bool] = None,
    off: bool = False,
) -> None:
    """
    Issue a single xrandr command for one output.
      - mode: must be a *physical* supported mode (or None to leave as-is)
      - pos: absolute desktop position
      - primary: True set, False clear (no explicit --no-primary in xrandr)
      - off: power off output
    """
    cmd = ["xrandr", "--output", name]
    if off:
        cmd += ["--off"]
    else:
        if mode:
            cmd += ["--mode", mode]
        if pos is not None:
            cmd += ["--pos", f"{pos[0]}x{pos[1]}"]
        if primary is True:
            cmd += ["--primary"]
    res = _run(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"xrandr failed ({name}): {' '.join(shlex.quote(c) for c in cmd)}\n{res.stderr}")

# --------------------------- Arrangers ------------------------------------------

def arrange_horizontally(
    monitors: Dict[str, MonitorInfo],
    left_to_right: Optional[List[str]] = None,
    set_primary: Optional[str] = None,
    apply_modes: bool = False,
    gap_px: int = 0,
) -> Dict[str, MonitorInfo]:
    """Place outputs side-by-side using physical widths; never force fantasy modes."""
    if left_to_right:
        ordered = [monitors[n] for n in left_to_right if n in monitors]
    else:
        ordered = sorted(monitors.values(), key=lambda m: (m.position[0], m.name))

    x_cursor = 0
    for m in ordered:
        mode = m.physical_mode if apply_modes else None
        _xrandr_set_output(m.name, mode=mode, pos=(x_cursor, 0),
                           primary=(m.name == set_primary if set_primary else None))
        m.position = (x_cursor, 0)
        x_cursor += m.physical_w + gap_px
    return monitors

def arrange_panoramic_quadh(
    monitors: Dict[str, MonitorInfo],
    sim_res: str = "1920x1080",
    primary_hint: Optional[str] = None,
    split_ultrawide_names: Optional[List[str]] = None,
    gap_px: int = 0,
) -> Dict[str, MonitorInfo]:
    """
    QUAD panoramic layout without touching physical modes.
    - Keeps outputs at physical modes; uses logical splits for your app math only.
    - split_ultrawide_names: list of outputs to treat as 2 logical halves.
    """
    ordered = sorted(monitors.values(), key=lambda m: (m.position[0], m.name))

    x_cursor = 0
    for m in ordered:
        _xrandr_set_output(m.name, mode=None, pos=(x_cursor, 0),
                           primary=(m.name == primary_hint if primary_hint else None))
        m.position = (x_cursor, 0)
        x_cursor += m.physical_w + gap_px

    for name in (split_ultrawide_names or []):
        if name in monitors:
            m = monitors[name]
            m.logical_w = m.physical_w // 2
            m.logical_h = m.physical_h
    return monitors

# --------------------------- Restore --------------------------------------------

def restore_layout(monitors: Dict[str, MonitorInfo]) -> None:
    """Restore each output to its original physical mode/position/primary."""
    for m in monitors.values():
        _xrandr_set_output(
            m.name,
            mode=m.original_mode,
            pos=m.original_pos,
            primary=(m.original_primary if m.original_primary else None),
        )

# --------------------------- NVIDIA viewport scaling ----------------------------

def set_nvidia_viewport(
    output: str,
    viewport_in: Tuple[int, int],
    viewport_out: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Ask the NVIDIA driver to scale a logical render size (ViewPortIn) into the physical panel (ViewPortOut).
    Leaves xrandr mode alone. Example: render 3840x1080 on a 5120x1440 panel.
    """
    monitors = probe_monitors()
    if output not in monitors:
        raise RuntimeError(f"Output {output} not found.")
    m = monitors[output]
    if not viewport_out:
        viewport_out = (m.physical_w, m.physical_h)

    vp_in = f"{viewport_in[0]}x{viewport_in[1]}"
    vp_out = f"{viewport_out[0]}x{viewport_out[1]}+0+0"
    meta = f"{output}: {m.physical_mode} {{ ViewPortIn={vp_in}, ViewPortOut={vp_out} }}"

    res = _run(["nvidia-settings", "--assign", f"CurrentMetaMode={meta}"])
    if res.returncode != 0:
        raise RuntimeError(f"nvidia-settings failed:\n{res.stderr}")
