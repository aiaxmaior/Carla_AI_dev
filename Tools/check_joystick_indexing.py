#!/usr/bin/env python3
"""
Simple diagnostic to check pygame joystick indexing
"""
import pygame
import sys

pygame.init()
pygame.joystick.init()

print("=" * 60)
print("PYGAME JOYSTICK DIAGNOSTIC")
print("=" * 60)

# Get joystick count
joystick_count = pygame.joystick.get_count()
print(f"\nNumber of joysticks detected: {joystick_count}")

if joystick_count == 0:
    print("No joysticks found!")
    sys.exit()

print("\n" + "=" * 60)
print("JOYSTICK DETAILS:")
print("=" * 60)

# Initialize all joysticks and show their details
joysticks = []
for i in range(joystick_count):
    joy = pygame.joystick.Joystick(i)
    joy.init()
    joysticks.append(joy)
    
    print(f"\nJoystick {i}:")
    print(f"  Name: {joy.get_name()}")
    print(f"  ID: {joy.get_id()}")  
    print(f"  Instance ID: {joy.get_instance_id()}")
    print(f"  GUID: {joy.get_guid()}")
    print(f"  Axes: {joy.get_numaxes()}")
    print(f"  Buttons: {joy.get_numbuttons()}")
    print(f"  Hats: {joy.get_numhats()}")
    print(f"  Balls: {joy.get_numballs()}")
    print(f"  Initialized: {joy.get_init()}")

# Set up display (required for event loop)
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Joystick Test - Press buttons or ESC to quit")
font = pygame.font.Font(None, 36)
small_font = pygame.font.Font(None, 24)

print("\n" + "=" * 60)
print("MONITORING EVENTS (Press ESC to quit):")
print("=" * 60)

clock = pygame.time.Clock()
running = True
last_event = "Waiting for input..."
event_history = []

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
                
        elif event.type == pygame.JOYBUTTONDOWN:
            msg = f"JOYBUTTONDOWN: joy={event.joy}, button={event.button}"
            if hasattr(event, 'instance_id'):
                msg += f", instance_id={event.instance_id}"
            print(msg)
            last_event = msg
            event_history.append(msg)
            
        elif event.type == pygame.JOYBUTTONUP:
            msg = f"JOYBUTTONUP: joy={event.joy}, button={event.button}"
            if hasattr(event, 'instance_id'):
                msg += f", instance_id={event.instance_id}"
            print(msg)
            last_event = msg
            event_history.append(msg)
            
        elif event.type == pygame.JOYAXISMOTION:
            if abs(event.value) > 0.1:  # Only show significant movement
                msg = f"JOYAXISMOTION: joy={event.joy}, axis={event.axis}, value={event.value:.2f}"
                if hasattr(event, 'instance_id'):
                    msg += f", instance_id={event.instance_id}"
                print(msg)
                last_event = msg
                
        elif event.type == pygame.JOYHATMOTION:
            if event.value != (0, 0):
                msg = f"JOYHATMOTION: joy={event.joy}, hat={event.hat}, value={event.value}"
                if hasattr(event, 'instance_id'):
                    msg += f", instance_id={event.instance_id}"
                print(msg)
                last_event = msg
                event_history.append(msg)
    
    # Clear screen
    screen.fill((30, 30, 40))
    
    # Draw title
    title = font.render("Joystick Diagnostic", True, (255, 255, 255))
    screen.blit(title, (250, 20))
    
    # Show joystick count
    info = small_font.render(f"Joysticks detected: {joystick_count}", True, (200, 200, 200))
    screen.blit(info, (20, 80))
    
    # Show each joystick info
    y_pos = 120
    for i, joy in enumerate(joysticks):
        if joy.get_init():
            text = small_font.render(f"Joy {i}: {joy.get_name()}", True, (150, 200, 150))
            screen.blit(text, (20, y_pos))
            
            # Show some live axis values
            axes_text = "Axes: "
            for axis in range(min(4, joy.get_numaxes())):  # Show first 4 axes
                val = joy.get_axis(axis)
                if abs(val) > 0.1:
                    axes_text += f"[{axis}]={val:.2f} "
            if axes_text != "Axes: ":
                axis_display = small_font.render(axes_text, True, (150, 150, 200))
                screen.blit(axis_display, (40, y_pos + 25))
            
            y_pos += 60
    
    # Show last event
    event_text = small_font.render(f"Last event: {last_event}", True, (255, 255, 150))
    screen.blit(event_text, (20, 400))
    
    # Show recent button history
    history_y = 450
    history_text = small_font.render("Recent buttons:", True, (200, 200, 200))
    screen.blit(history_text, (20, history_y))
    for i, evt in enumerate(event_history[-5:]):  # Show last 5 button events
        evt_text = small_font.render(evt, True, (150, 150, 150))
        screen.blit(evt_text, (40, history_y + 25 + i*20))
    
    # Instructions
    instructions = small_font.render("Press joystick buttons or ESC to quit", True, (100, 100, 100))
    screen.blit(instructions, (200, 550))
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
