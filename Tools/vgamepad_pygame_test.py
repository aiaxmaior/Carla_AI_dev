# ============================================================================
# PERF CHECK (file-level):
# ============================================================================
# [X] | Role: Test/utility script for virtual gamepad testing (NOT in hot path)
# [ ] | Hot-path functions: None (standalone test script)
# [ ] |- Heavy allocs in hot path? N/A
# [ ] |- pandas/pyarrow/json/disk/net in hot path? No
# [X] | Graphics here? YES - pygame (test only)
# [ ] | Data produced (tick schema?): None (console output only)
# [ ] | Storage (Parquet/Arrow/CSV/none): None
# [ ] | Queue/buffer used?: No
# [ ] | Session-aware? No
# [ ] | Debug-only heavy features?: None
# Top 3 perf risks:
# 1. [PERF_OK] Test/utility script - NOT in hot path
# 2. [PERF_OK] vgamepad + pygame operations acceptable for testing
# 3. [PERF_OK] No performance concerns
# ============================================================================

import vgamepad as vg
import pygame
import time

# Create virtual gamepad
print("Creating virtual gamepad...")
gamepad = vg.VX360Gamepad()

# Wait a moment for device to register
time.sleep(0.5)

# Initialize pygame
pygame.init()
pygame.joystick.init()

# Check if pygame sees it
print(f"\nJoysticks detected by pygame: {pygame.joystick.get_count()}")

for i in range(pygame.joystick.get_count()):
    js = pygame.joystick.Joystick(i)
    js.init()
    print(f"  [{i}] {js.get_name()} - {js.get_numbuttons()} buttons, {js.get_numaxes()} axes")
    
    # Check if it's the virtual controller
    if "360" in js.get_name().lower() or "xbox" in js.get_name().lower():
        print(f"    ^ This is likely the virtual gamepad!")

# Test button press
print("\nTesting button press...")
gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
gamepad.update()

# Check pygame events
pygame.event.pump()
for event in pygame.event.get():
    if event.type == pygame.JOYBUTTONDOWN:
        print(f"  pygame detected: Button {event.button} pressed on joystick {event.joy}")

gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
gamepad.update()