#!/usr/bin/env python3
"""
Quick console check of pygame joystick indexing
"""
import pygame
import time

pygame.init()
pygame.joystick.init()

print("\n" + "="*70)
print("PYGAME JOYSTICK INDEX CHECK")
print("="*70)

# Check pygame version
print(f"\nPygame version: {pygame.version.ver}")

# Get joystick count
count = pygame.joystick.get_count()
print(f"Joysticks found: {count}")

if count == 0:
    print("\nNo joysticks detected!")
    print("Make sure your controller is connected and recognized by the system.")
else:
    print("\n" + "-"*70)
    print("JOYSTICK ENUMERATION:")
    print("-"*70)
    
    for i in range(count):
        joy = pygame.joystick.Joystick(i)
        joy.init()
        
        print(f"\n[Index {i}]")
        print(f"  Name: {joy.get_name()}")
        print(f"  ID: {joy.get_id()}")
        print(f"  Instance ID: {joy.get_instance_id()}")
        print(f"  Buttons: {joy.get_numbuttons()}")
        print(f"  Axes: {joy.get_numaxes()}")
        
    print("\n" + "-"*70)
    print("TESTING BUTTON EVENTS (Press Ctrl+C to stop):")
    print("-"*70)
    print("\nPress any button on your controller...")
    print("(Looking for button 37 specifically - your ENTER mapping)\n")
    
    # Need a display for events to work
    screen = pygame.display.set_mode((100, 100))
    
    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN:
                    print(f"Button pressed: joy={event.joy}, button={event.button}", end="")
                    
                    # Check if this matches your mapping
                    if event.button == 37:
                        print(" <- This is your ENTER button (37)!", end="")
                    if event.button == 36:
                        print(" <- This is your ESCAPE button (36)!", end="")
                    
                    # Show which joystick index this would need
                    print(f"\n  -> To map this, you'd use joy_idx={event.joy}")
                    
                    # Extra info if available
                    if hasattr(event, 'instance_id'):
                        print(f"  -> Instance ID: {event.instance_id}")
                        
                elif event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                    
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\n\nStopped by user")

pygame.quit()
print("\nDone!")
