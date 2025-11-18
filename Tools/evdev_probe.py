# ============================================================================
# PERF CHECK (file-level):
# ============================================================================
# [X] | Role: Test/utility script for evdev device probing (NOT in hot path)
# [ ] | Hot-path functions: None (standalone probe tool)
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
# 2. [PERF_OK] evdev probing acceptable for testing
# 3. [PERF_OK] No performance concerns
# ============================================================================

from evdev import InputDevice, list_devices, ecodes
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--needle", default="Arduino")
args = parser.parse_args()

for p in list_devices():
    d = InputDevice(p)
    if args.needle.lower() in (d.name or "").lower():
        print(f"Found {d.name} at {p}")
        for e in d.read_loop():
            if e.type in (ecodes.EV_KEY, ecodes.EV_SW):
                print(f"type={e.type} code={e.code} value={e.value}")