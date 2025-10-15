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