# ============================================================================
# PERF CHECK (file-level):
# ============================================================================
# [X] | Role: Test/demo script for Fanatec axis calibration (NOT in hot path)
# [ ] | Hot-path functions: None (standalone demo/test)
# [ ] |- Heavy allocs in hot path? N/A
# [ ] |- pandas/pyarrow/json/disk/net in hot path? No
# [X] | Graphics here? YES - pygame (demo only)
# [ ] | Data produced (tick schema?): None (console output only)
# [ ] | Storage (Parquet/Arrow/CSV/none): None
# [ ] | Queue/buffer used?: No
# [ ] | Session-aware? No
# [ ] | Debug-only heavy features?: None
# Top 3 perf risks:
# 1. [PERF_OK] Test/demo script - NOT in hot path
# 2. [PERF_OK] pygame operations acceptable for testing
# 3. [PERF_OK] No performance concerns
# ============================================================================

import pygame
import sys

# --- IMPORTANT: Replace these placeholders with your actual joystick and control IDs! ---
# Use the output from the joystick_input_finder.py script to find these values.
# These IDs are specific to your hardware setup on your computer.

# Steering
STEER_JOYSTICK_ID = 1
STEER_AXIS_ID = 0

# Throttle
THROTTLE_JOYSTICK_ID = 1
THROTTLE_AXIS_ID = 1

# Brake
BRAKE_JOYSTICK_ID = 1
BRAKE_AXIS_ID = 5

# Add IDs for other axes if needed (e.g., Clutch)
# CLUTCH_JOYSTICK_ID = 1
# CLUTCH_AXIS_ID = 2


# --- Helper function for pedal mapping (handles inversion and 0-1 range) ---
def map_pedal_input(raw_value):
    """
    Maps raw pedal axis value to CARLA's throttle/brake range (0.0 to 1.0).
    Assumes raw_value is -1 (pressed) to 1 (released) and inverts it.
    Add deadzone or non-linearity here if needed.
    """
    # Pygame output is often -1 (pressed) to 1 (released) for pedals
    # We want 0 (released) to 1 (pressed) for CARLA
    # Invert the value, then scale from -1..1 to 0..2, then divide by 2
    print('raw_value:', raw_value)
    return (raw_value * -1 + 1) / 2

# --- Helper function for steering mapping ---
def map_steering_input(raw_value):
    """
    Maps raw steering axis value to CARLA's steer range (-1.0 to 1.0).
    Assumes raw_value is already -1.0 (left) to 1.0 (right).
    Add deadzone or non-linearity here if needed.
    """
    # Assuming raw_value is already -1.0 (left) to 1.0 (right)
    # Add deadzone or non-linearity here if needed
    return raw_value

# --- Function to get current Fanatec control input (Axes Only) ---
# This function is designed to be called repeatedly (e.g., per CARLA tick)
def get_fanatec_axis_input(joysticks):
    """
    Reads and maps the current state of configured Fanatec axes (Steer, Throttle, Brake).

    Args:
        joysticks (list): A list of initialized pygame.joystick.Joystick objects.
                          This list should be created and initialized once in the main script.

    Returns:
        dict: A dictionary containing the mapped axis values:
              'steer' (float), 'throttle' (float), 'brake' (float).
              Returns 0.0 for a value if the corresponding joystick or axis ID is invalid
              or a pygame.error occurs during reading.
    """
    # Process Pygame events to keep the event queue from overflowing.
    # This is crucial even if you don't explicitly handle all event types in this function.
    # Event handling (like button presses for gear shifts) should primarily be done
    # in the main script's event loop which calls this function.
    pygame.event.pump()

    control_values = {
        'steer': 0.0,
        'throttle': 0.0,
        'brake': 0.0,
        # Add other axis keys if needed (e.g., 'clutch')
    }

    joystick_count = len(joysticks)

    # --- Read Steering Axis ---
    # Check if the configured joystick and axis IDs are valid before attempting to read
    if STEER_JOYSTICK_ID < joystick_count and joysticks[STEER_JOYSTICK_ID].get_numaxes() > STEER_AXIS_ID:
        try:
            raw_steer = joysticks[STEER_JOYSTICK_ID].get_axis(STEER_AXIS_ID)
            control_values['steer'] = map_steering_input(raw_steer)
        except pygame.error:
             # Handle potential read error - leave default value (0.0)
             pass
    # Note: No error printing here to keep the function clean.
    # The main script calling this function can decide how to handle potential errors
    # if a required input consistently fails to read (e.g., log a warning).


    # --- Read Throttle Axis ---
    # Check if the configured joystick and axis IDs are valid
    if THROTTLE_JOYSTICK_ID < joystick_count and joysticks[THROTTLE_JOYSTICK_ID].get_numaxes() > THROTTLE_AXIS_ID:
        try:
            raw_throttle = joysticks[THROTTLE_JOYSTICK_ID].get_axis(THROTTLE_AXIS_ID)
            control_values['throttle'] = map_pedal_input(raw_throttle)
        except pygame.error:
             pass

    # --- Read Brake Axis ---
    # Check if the configured joystick and axis IDs are valid
    if BRAKE_JOYSTICK_ID < joystick_count and joysticks[BRAKE_JOYSTICK_ID].get_numaxes() > BRAKE_AXIS_ID:
        try:
            raw_brake = joysticks[BRAKE_JOYSTICK_ID].get_axis(BRAKE_AXIS_ID)
            control_values['brake'] = map_pedal_input(raw_brake)
        except pygame.error:
             pass

    # --- Read Other Axes (Example for Clutch) ---
    # if 'CLUTCH_JOYSTICK_ID' in globals() and CLUTCH_JOYSTICK_ID < joystick_count and joysticks[CLUTCH_JOYSTICK_ID].get_numaxes() > CLUTCH_AXIS_ID:
    #     try:
    #         raw_clutch = joysticks[CLUTCH_JOYSTICK_ID].get_axis(CLUTCH_AXIS_ID)
    #         control_values['clutch'] = map_pedal_input(raw_clutch) # Map clutch similarly
    #     except pygame.error:
    #          pass

    # This function does NOT handle button or hat events.
    # Button/hat event handling (like gear shifts, handbrake toggle) should be done
    # in the main script's pygame event loop (for event types JOYBUTTONDOWN, JOYBUTTONUP, JOYHATMOTION).
    # The state of buttons/hats should be stored in variables in the main script
    # and combined with the axis_control_values dictionary AFTER calling this function.


    # This function does NOT print output to the console.
    # Printing or displaying the control values should be done in the main script.


    return control_values

# Note: This script defines a function to read input.
# Pygame initialization (pygame.init(), pygame.joystick.init()),
# finding and initializing joysticks (pygame.joystick.Joystick(i).init()),
# and the main simulation loop should be in your main CARLA scenario script.
# Pygame.quit() should also be called in the main script's cleanup.
