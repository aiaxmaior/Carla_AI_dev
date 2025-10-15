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