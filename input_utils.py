#! Helper Utilities
# ============================================================================
# PERF CHECK (file-level):
# ============================================================================
# [X] | Role: Helper utilities for joystick event ID resolution
# [ ] | Hot-path functions: ui_button_pressed() called during event processing
# [ ] |- Heavy allocs in hot path? Minimal - dict/tuple lookups only
# [ ] |- pandas/pyarrow/json/disk/net in hot path? No
# [ ] | Graphics here? No
# [ ] | Data produced (tick schema?): None
# [ ] | Storage (Parquet/Arrow/CSV/none): None
# [ ] | Queue/buffer used?: No
# [ ] | Session-aware? No
# [ ] | Debug-only heavy features?: None
# Top 3 perf risks:
# 1. [PERF_OK] Helper functions - lightweight event matching
# 2. [PERF_OK] Called during UI events only (not every frame)
# 3. [PERF_OK] No heavy operations
# ============================================================================

import pygame

def _resolve_event_ids(event):
    """Return (instance_id, internal_index) for a joystick event."""
    # SDL2 events have instance_id; older Pygame also exposes .joy
    inst = getattr(event, "instance_id", getattr(event, "joy", None))
    idx = None
    if inst is not None:
        # If you cache caps as [{'id': instance_id, 'index': i}, ...]
        caps = getattr(event, "_joystick_caps", None)  # optional; see below
        if caps and isinstance(caps, list):
            for c in caps:
                if c.get("id") == inst:
                    idx = c.get("index")
                    break
    return inst, idx

def ui_button_pressed(event, mapping, joystick_caps):
    """
    Strictly for buttons.
    mapping: {"type":"button","joy_idx":<int> or None,"joy_id":<int> or None,"id":<button_id:int>}
    joystick_caps: [{'id': instance_id, 'index': internal_index}, ...]
    """
    if event.type != pygame.JOYBUTTONDOWN:
        return False
    if not isinstance(mapping, dict) or mapping.get("type") != "button":
        return False

    # Attach caps so _resolve_event_ids can see them (avoids global state).
    setattr(event, "_joystick_caps", joystick_caps or [])

    ev_inst, ev_idx = _resolve_event_ids(event)
    btn_ok = (event.button == mapping.get("id"))
    if not btn_ok:
        return False

    map_idx = mapping.get("joy_idx")
    map_id  = mapping.get("joy_id")

    # Accept any of these:
    return (
        (map_id  is not None and ev_inst == map_id)  or
        (map_idx is not None and ev_idx  == map_idx) or
        # tolerate files where 'joy_idx' actually holds instance id:
        (map_idx is not None and ev_inst == map_idx)
    )
