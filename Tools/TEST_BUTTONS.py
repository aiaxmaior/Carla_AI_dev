# ============================================================================
# PERF CHECK (file-level):
# ============================================================================
# [X] | Role: Test/utility script for joystick button detection (NOT in hot path)
# [ ] | Hot-path functions: None (standalone test script)
# [ ] |- Heavy allocs in hot path? N/A
# [ ] |- pandas/pyarrow/json/disk/net in hot path? No
# [ ] | Graphics here? YES - pygame (test only)
# [ ] | Data produced (tick schema?): None (console output only)
# [ ] | Storage (Parquet/Arrow/CSV/none): None
# [ ] | Queue/buffer used?: No
# [ ] | Session-aware? No
# [ ] | Debug-only heavy features?: None
# Top 3 perf risks:
# 1. [PERF_OK] Test/utility script - NOT in hot path
# 2. [PERF_OK] pygame operations acceptable for testing
# 3. [PERF_OK] No performance concerns
# ============================================================================

import pygame
import time

def find_joystick_inputs():
    pygame.init()
    pygame.joystick.init()

    joystick_count = pygame.joystick.get_count()
    if joystick_count == 0:
        print("No joysticks found. Please connect your controller and try again.")
        return

    print(f"Found {joystick_count} joystick(s).")
    joysticks = []
    for i in range(joystick_count):
        joystick = pygame.joystick.Joystick(i)
        joystick.init()
        joysticks.append(joystick)
        print(f"\nJoystick {i}: {joystick.get_name()}")
        print(f"  Number of Axes: {joystick.get_numaxes()}")
        print(f"  Number of Buttons: {joystick.get_numbuttons()}")
        print(f"  Number of Hats: {joystick.get_numhats()}")

    print("\n--- Testing Joystick Inputs ---")
    print("Move axes, press buttons, and move hats to see their events.")
    print("Press Ctrl+C or close the window to exit.")

    active_joystick_states = {} # To track last reported values for axes and hats

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt

                if event.type == pygame.JOYAXISMOTION:
                    # Only print if the value has changed significantly to reduce spam
                    current_value = round(event.value, 3)
                    key = (event.joy, 'axis', event.axis)
                    if key not in active_joystick_states or abs(active_joystick_states[key] - current_value) > 0.01:
                        print(f"Joystick {event.joy} - Axis {event.axis}: {current_value:.3f}")
                        active_joystick_states[key] = current_value

                elif event.type == pygame.JOYBUTTONDOWN:
                    print(f"Joystick {event.joy} - Button {event.button}: DOWN")

                elif event.type == pygame.JOYBUTTONUP:
                    print(f"Joystick {event.joy} - Button {event.button}: UP")

                elif event.type == pygame.JOYHATMOTION:
                    # Hat motion reports a tuple (x, y) e.g., (0, 1) for up
                    current_value = event.value
                    key = (event.joy, 'hat', event.hat)
                    if key not in active_joystick_states or active_joystick_states[key] != current_value:
                        print(f"Joystick {event.joy} - Hat {event.hat}: {current_value}")
                        active_joystick_states[key] = current_value
            
            time.sleep(0.01) # Small delay to prevent busy-waiting

    except KeyboardInterrupt:
        print("\nExiting joystick input tester.")
    finally:
        pygame.quit()

if __name__ == "__main__":
    find_joystick_inputs()
