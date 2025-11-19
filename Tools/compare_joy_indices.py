#!/usr/bin/env python3
"""
Compare pygame's joystick indexing with mapping file indices
"""
import pygame
import json
import os

pygame.init()
pygame.joystick.init()

print("\n" + "="*70)
print("JOYSTICK INDEX COMPARISON")
print("="*70)

# Check what pygame sees
print("\n1. PYGAME'S VIEW:")
print("-"*40)
count = pygame.joystick.get_count()
print(f"Joystick count: {count}")

pygame_joysticks = {}
for i in range(count):
    joy = pygame.joystick.Joystick(i)
    joy.init()
    pygame_joysticks[i] = {
        'name': joy.get_name(),
        'id': joy.get_id(),
        'instance_id': joy.get_instance_id(),
        'buttons': joy.get_numbuttons()
    }
    print(f"\nJoystick at index {i}:")
    print(f"  Name: {joy.get_name()}")
    print(f"  Instance ID: {joy.get_instance_id()}")
    print(f"  Buttons: {joy.get_numbuttons()}")

# Check mapping files
print("\n2. MAPPING FILES:")
print("-"*40)

mapping_files = [
    'welcome_mapping.json',
    'controller_config.json',
    'joystick_config.json'
]

for filename in mapping_files:
    if os.path.exists(filename):
        print(f"\nFound: {filename}")
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                
            # Look for joy_idx values
            def find_joy_indices(obj, path=""):
                indices = []
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if key == 'joy_idx':
                            indices.append((path + "." + key if path else key, value))
                        elif isinstance(value, (dict, list)):
                            indices.extend(find_joy_indices(value, path + "." + key if path else key))
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        indices.extend(find_joy_indices(item, f"{path}[{i}]"))
                return indices
            
            indices = find_joy_indices(data)
            if indices:
                print("  Joy indices found:")
                for path, idx in indices:
                    print(f"    {path}: {idx}")
                    
                # Check if these indices exist in pygame
                unique_indices = set(idx for _, idx in indices)
                for idx in unique_indices:
                    if idx in pygame_joysticks:
                        print(f"    ✓ Index {idx} exists in pygame")
                    else:
                        print(f"    ✗ Index {idx} NOT found in pygame (pygame has 0-{count-1})")
            else:
                print("  No joy_idx values found in this file")
                
        except Exception as e:
            print(f"  Error reading file: {e}")
    else:
        print(f"\n{filename} not found")

# Test button events
print("\n3. LIVE BUTTON TEST:")
print("-"*40)
print("Press button 37 (your ENTER) or button 36 (your ESCAPE)...")
print("Press Ctrl+C to stop\n")

screen = pygame.display.set_mode((100, 100))

try:
    test_count = 0
    while test_count < 10:  # Just test 10 button presses
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                test_count += 1
                print(f"Event: joy={event.joy}, button={event.button}", end="")
                
                # Your specific buttons
                if event.button == 37:
                    print(" <- ENTER button")
                    print(f"  This event has joy index: {event.joy}")
                    print(f"  Your mapping expects joy_idx: 1")
                    if event.joy == 1:
                        print("  ✓ MATCH!")
                    else:
                        print(f"  ✗ MISMATCH! (pygame sees {event.joy}, mapping wants 1)")
                elif event.button == 36:
                    print(" <- ESCAPE button")
                    print(f"  This event has joy index: {event.joy}")
                    print(f"  Your mapping expects joy_idx: 1")
                    if event.joy == 1:
                        print("  ✓ MATCH!")
                    else:
                        print(f"  ✗ MISMATCH! (pygame sees {event.joy}, mapping wants 1)")
                else:
                    print()
                    
            elif event.type == pygame.QUIT:
                raise KeyboardInterrupt
                
except KeyboardInterrupt:
    print("\nStopped by user")
except Exception:
    pass

pygame.quit()

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print("""
If you're seeing index mismatches:
- Pygame uses 0-based indexing (first controller is 0)
- Your mappings might use 1-based indexing (first controller is 1)
- OR you might have multiple controllers/devices detected
""")
