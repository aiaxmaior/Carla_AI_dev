    # MozaVirtualGamepad.py
# ============================================================================
# PERF CHECK (file-level):
# ============================================================================
# [X] | Role: Hardware bridge for MOZA stalk inputs (blinkers, hazard)
# [ ] | Hot-path functions: None (runs in separate thread)
# [ ] |- Heavy allocs in hot path? Minimal - event processing only
# [ ] |- pandas/pyarrow/json/disk/net in hot path? No
# [ ] | Graphics here? No
# [ ] | Data produced (tick schema?): None (translates hardware events)
# [ ] | Storage (Parquet/Arrow/CSV/none): None
# [ ] | Queue/buffer used?: No (direct event translation)
# [ ] | Session-aware? No
# [ ] | Debug-only heavy features?: None
# Top 3 perf risks:
# 1. [PERF_OK] Runs in SEPARATE THREAD - isolated from main loop
# 2. [PERF_OK] Threaded I/O with evdev - non-blocking
# 3. [PERF_OK] time.sleep(0.1) for button press/release - acceptable
# ============================================================================

import threading
import time
import vgamepad as vg
from evdev import InputDevice, ecodes
import logging


class MozaVirtualGamepad:
    """Translates MOZA stalk events to virtual gamepad buttons."""

    def __init__(self, moza_device_path="/dev/input/event8"):
        self.moza_path = moza_device_path
        self.moza = None
        self.gamepad = vg.VX360Gamepad()
        self._running = False
        self._thread = None

        # Map MOZA codes to gamepad buttons
        self.button_map = {
            297: vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER,  # Left on
            295: vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER,  # Right on
            291: vg.XUSB_BUTTON.XUSB_GAMEPAD_Y,  # Hazard
        }

        # Track states for toggle behavior
        self._left_active = False
        self._right_active = False

    def start(self):
        """Start the translation thread."""
        if self._running:
            return

        try:
            self.moza = InputDevice(self.moza_path)
            self.moza.grab()  # Exclusive access
        except Exception as e:
            logging.error(f"Failed to open MOZA device: {e}")
            return False

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logging.info(f"MOZA Virtual Gamepad started on {self.moza_path}")
        return True

    def stop(self):
        """Stop the translation thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self.moza:
            try:
                self.moza.ungrab()
                self.moza.close()
            except Exception:
                pass
        self.gamepad.reset()
        self.gamepad.update()

    def _loop(self):
        """Main loop reading MOZA events and sending gamepad inputs."""
        try:
            for event in self.moza.read_loop():
                if not self._running:
                    break

                if event.type != ecodes.EV_KEY:
                    continue

                code = event.code
                value = event.value  # 1=pressed, 0=released

                # Handle left blinker toggle
                if code == 297 and value == 1:  # Left on
                    if not self._left_active:
                        self._left_active = True
                        self._right_active = False
                        # Press and release left shoulder button
                        self.gamepad.press_button(
                            button=vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER
                        )
                        self.gamepad.update()
                        time.sleep(0.1)
                        self.gamepad.release_button(
                            button=vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER
                        )
                        self.gamepad.update()
                        logging.info("MOZA: Left blinker -> Virtual LB pressed")

                elif code == 296 and value == 1:  # Cancel
                    if self._left_active:
                        self._left_active = False
                        # Press left shoulder again to toggle off
                        self.gamepad.press_button(
                            button=vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER
                        )
                        self.gamepad.update()
                        time.sleep(0.1)
                        self.gamepad.release_button(
                            button=vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER
                        )
                        self.gamepad.update()
                        logging.info("MOZA: Left cancel -> Virtual LB pressed")

                # Handle right blinker toggle
                elif code == 295 and value == 1:  # Right on
                    if not self._right_active:
                        self._right_active = True
                        self._left_active = False
                        # Press and release right shoulder button
                        self.gamepad.press_button(
                            button=vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER
                        )
                        self.gamepad.update()
                        time.sleep(0.1)
                        self.gamepad.release_button(
                            button=vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER
                        )
                        self.gamepad.update()
                        logging.info("MOZA: Right blinker -> Virtual RB pressed")

                elif code == 296 and value == 1 and self._right_active:  # Right cancel
                    self._right_active = False
                    # Press right shoulder again to toggle off
                    self.gamepad.press_button(
                        button=vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER
                    )
                    self.gamepad.update()
                    time.sleep(0.1)
                    self.gamepad.release_button(
                        button=vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER
                    )
                    self.gamepad.update()
                    logging.info("MOZA: Right cancel -> Virtual RB pressed")

        except Exception as e:
            logging.error(f"MOZA virtual gamepad loop error: {e}")
        finally:
            self.gamepad.reset()
            self.gamepad.update()

            # Add this at the bottom of MozaVirtualGamepad.py


if __name__ == "__main__":
    import sys

    # Set up logging to see what's happening
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    print("=" * 60)
    print("MOZA Virtual Gamepad Test")
    print("=" * 60)
    print("\nThis will translate MOZA stalk inputs to virtual gamepad buttons.")
    print("Press Ctrl+C to exit.\n")

    # Try to find the MOZA device
    from evdev import list_devices

    print("Searching for MOZA device...")
    moza_path = None

    for device_path in list_devices():
        try:
            dev = InputDevice(device_path)
            if "moza" in dev.name.lower() or "gudsen" in dev.name.lower():
                moza_path = device_path
                print(f"✓ Found MOZA device: {dev.name} at {device_path}")
                break
        except Exception:
            continue

    if not moza_path:
        print("✗ No MOZA device found. Available devices:")
        for device_path in list_devices():
            try:
                dev = InputDevice(device_path)
                print(f"  - {dev.name} at {device_path}")
            except Exception:
                pass
        sys.exit(1)

    # Create and start the bridge
    bridge = MozaVirtualGamepad(moza_path)

    print("\nStarting MOZA Virtual Gamepad bridge...")
    if not bridge.start():
        print("✗ Failed to start bridge")
        sys.exit(1)


def test_moza_raw():
    """Test raw MOZA input without virtual gamepad."""
    print("\n" + "=" * 60)
    print("RAW MOZA INPUT TEST (no virtual gamepad)")
    print("=" * 60)

    from evdev import list_devices, InputDevice, categorize, ecodes

    # Find MOZA
    moza_path = None
    for device_path in list_devices():
        try:
            dev = InputDevice(device_path)
            if "moza" in dev.name.lower() or "gudsen" in dev.name.lower():
                moza_path = device_path
                print(f"Found: {dev.name} at {device_path}")
                break
        except Exception:
            continue

    if not moza_path:
        print("No MOZA device found!")
        return

    device = InputDevice(moza_path)
    print("\nDevice info:")
    print(f"  Name: {device.name}")
    print(f"  Path: {device.path}")
    print(f"  Phys: {device.phys}")

    print("\nListening for events... (Ctrl+C to exit)")
    print("Move the stalks to see raw event codes:\n")

    try:
        for event in device.read_loop():
            if event.type == ecodes.EV_KEY:
                key_event = categorize(event)
                state = (
                    "PRESSED"
                    if event.value == 1
                    else "RELEASED"
                    if event.value == 0
                    else "HELD"
                )
                print(
                    f"Code: {event.code:3d} | State: {state:8s} | {key_event.keycode if hasattr(key_event, 'keycode') else 'Unknown'}"
                )

                # Interpret known codes
                if event.code == 297 and event.value == 1:
                    print("  → LEFT BLINKER ON")
                elif event.code == 295 and event.value == 1:
                    print("  → RIGHT BLINKER ON")
                elif event.code == 296 and event.value == 1:
                    print("  → CANCEL/OFF")
                elif event.code == 291 and event.value == 1:
                    print("  → HAZARD")

    except KeyboardInterrupt:
        print("\nTest ended")

    print("✓ Bridge started successfully!")
    print("\n" + "=" * 60)
    print("TEST INSTRUCTIONS:")
    print("=" * 60)
    print("1. Move the LEFT stalk - should see 'Left blinker' message")
    print("2. Move the RIGHT stalk - should see 'Right blinker' message")
    print("3. The stalks should toggle (engage once, cancel once)")
    print("\nYou can also run 'jstest /dev/input/js*' in another terminal")
    print("to see the virtual gamepad buttons being pressed.")
    print("\nMonitoring MOZA events... (Ctrl+C to exit)")
    print("-" * 60)

    try:
        # Keep the program running
        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nShutting down...")
        bridge.stop()
        print("✓ Bridge stopped")
        print("Goodbye!")
