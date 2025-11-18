#!/usr/bin/env python3
"""
Diagnostic script to understand pygame window positioning and centering.
Run this to see how pygame interprets --display and window dimensions.
"""

import pygame
import sys

def diagnose_layout(display_index=2, window_width=7680, window_height=1080):
    pygame.init()

    # Get desktop info
    desktop_sizes = pygame.display.get_desktop_sizes()
    print("\n" + "="*70)
    print("DESKTOP MONITOR DETECTION")
    print("="*70)
    for i, size in enumerate(desktop_sizes):
        print(f"  Monitor {i}: {size[0]}x{size[1]}")

    # Create window
    print(f"\n{'='*70}")
    print(f"CREATING PYGAME WINDOW")
    print(f"{'='*70}")
    print(f"  Window size: {window_width}x{window_height}")
    print(f"  Display index: {display_index}")

    try:
        display = pygame.display.set_mode(
            (window_width, window_height),
            pygame.HWSURFACE | pygame.DOUBLEBUF,
            display=display_index
        )

        # Get window info
        w = display.get_width()
        h = display.get_height()

        print(f"\n{'='*70}")
        print("PYGAME WINDOW COORDINATES")
        print(f"{'='*70}")
        print(f"  display.get_width()  = {w}")
        print(f"  display.get_height() = {h}")
        print(f"  Window center X      = {w // 2}")
        print(f"  Window center Y      = {h // 2}")

        # Calculate monitor positions within pygame window
        print(f"\n{'='*70}")
        print("LOGICAL MONITOR LAYOUT (assuming 4 equal panels)")
        print(f"{'='*70}")
        panel_w = w // 4
        for i in range(4):
            x_start = i * panel_w
            x_end = x_start + panel_w
            x_center = x_start + (panel_w // 2)
            print(f"  Panel {i}: x={x_start:4d} to {x_end:4d}, center at x={x_center:4d}")

        print(f"\n{'='*70}")
        print("CENTERING CALCULATIONS")
        print(f"{'='*70}")
        print(f"  Full window center (w//2):           {w // 2}")
        print(f"  Panel 1 center (1*panel_w + panel_w//2): {1*panel_w + panel_w//2}")
        print(f"  Panel 2 center (2*panel_w + panel_w//2): {2*panel_w + panel_w//2}")
        print(f"  Panels 1+2 center ((1*panel_w) + panel_w): {1*panel_w + panel_w}")

        # Fill with gradient to visualize
        font = pygame.font.Font(None, 72)

        # Color each panel differently
        colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100)]
        for i in range(4):
            x = i * panel_w
            pygame.draw.rect(display, colors[i], (x, 0, panel_w, h))

            # Label each panel
            label = font.render(f"Panel {i}", True, (255, 255, 255))
            display.blit(label, (x + 50, 50))

            # Show center line
            center_x = x + (panel_w // 2)
            pygame.draw.line(display, (255, 255, 255), (center_x, 0), (center_x, h), 3)
            center_label = font.render(f"x={center_x}", True, (255, 255, 255))
            display.blit(center_label, (center_x - 100, h // 2))

        # Mark full window center
        center_x = w // 2
        pygame.draw.line(display, (255, 0, 255), (center_x, 0), (center_x, h), 5)
        center_label = font.render(f"FULL CENTER x={center_x}", True, (255, 0, 255))
        display.blit(center_label, (center_x - 200, h - 100))

        pygame.display.flip()

        print(f"\n{'='*70}")
        print("VISUAL TEST WINDOW")
        print(f"{'='*70}")
        print("  A test window is now displayed.")
        print("  - Each colored panel represents a logical monitor")
        print("  - White lines show panel centers")
        print("  - MAGENTA line shows full window center")
        print("\n  Press any key to close...")

        # Wait for keypress
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                elif event.type == pygame.KEYDOWN:
                    waiting = False

        pygame.quit()

    except Exception as e:
        print(f"\nERROR creating window: {e}")
        pygame.quit()
        return

if __name__ == "__main__":
    # Parse arguments
    display_idx = 2
    if len(sys.argv) > 1:
        try:
            display_idx = int(sys.argv[1])
        except:
            print(f"Invalid display index: {sys.argv[1]}")
            sys.exit(1)

    print("\nUsage: python diagnose_display_layout.py [display_index]")
    print(f"Using display index: {display_idx}\n")

    diagnose_layout(display_index=display_idx)
