# ============================================================================
# PERF CHECK (file-level):
# ============================================================================
# [X] | Role: Test/utility script for evdev event reading (NOT in hot path)
# [ ] | Hot-path functions: None (standalone event monitor)
# [ ] |- Heavy allocs in hot path? N/A
# [ ] |- pandas/pyarrow/json/disk/net in hot path? No
# [ ] | Graphics here? No
# [ ] | Data produced (tick schema?): None (console output only)
# [ ] | Storage (Parquet/Arrow/CSV/none): None
# [ ] | Queue/buffer used?: No
# [ ] | Session-aware? No
# [ ] | Debug-only heavy features?: None
# Top 3 perf risks:
# 1. [PERF_OK] Test/utility script - NOT in hot path
# 2. [PERF_OK] evdev read_loop acceptable for testing
# 3. [PERF_OK] No performance concerns
# ============================================================================

from evdev import InputDevice, categorize, ecodes

dev = InputDevice('/dev/input/event11')
print(f"Listening on {dev.name} ({dev.path})")

for event in dev.read_loop():
    if event.type == ecodes.EV_KEY:
        key_event = categorize(event)
        print(f"[KEY] {key_event.keycode}: {'DOWN' if key_event.keystate == 1 else 'UP'}")
    
    elif event.type == ecodes.EV_ABS:
        print(f"[AXIS] Code {event.code} Value: {event.value}")
    
    elif event.type == ecodes.EV_REL:
        print(f"[REL] Code {event.code} Value: {event.value}")
    
    else:
        print(f"[OTHER] Type {event.type} Code {event.code} Value {event.value}")