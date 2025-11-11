# MozaArduinoVirtualGamepad.py
import threading
import time
import vgamepad as vg
from evdev import InputDevice, ecodes, list_devices
import logging
import json
import os

class HardwareVirtualGamepad:
    """Translates MOZA stalk and Arduino seatbelt events to virtual gamepad."""
    
    def __init__(self, config_path="./configs/joystick_mappings/input_devices.json"):
        self.gamepad = vg.VX360Gamepad()
        self._running = False
        self._threads = []
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # MOZA state tracking
        self._left_active = False
        self._right_active = False
        
        # Seatbelt state tracking
        self._seatbelt_fastened = False
        
        # Device handles
        self.moza_device = None
        self.arduino_device = None
    
    def _load_config(self, path):
        """Load device configuration from JSON."""
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f).get('evdev', {})
            except Exception as e:
                logging.error(f"Failed to load config: {e}")
        return {}
    
    def _find_device(self, name_substring):
        """Find device by name substring."""
        for device_path in list_devices():
            try:
                dev = InputDevice(device_path)
                if name_substring.lower() in dev.name.lower():
                    return device_path
            except:
                continue
        return None
    
    def start(self):
        """Start all device threads."""
        if self._running:
            return True
            
        self._running = True
        success = False
        
        # Start MOZA thread
        moza_config = self.config.get('moza', {})
        moza_path = moza_config.get('event') or self._find_device('moza')
        
        if moza_path:
            try:
                self.moza_device = InputDevice(moza_path)
                self.moza_device.grab()
                thread = threading.Thread(target=self._moza_loop, args=(moza_config,), daemon=True)
                thread.start()
                self._threads.append(thread)
                logging.info(f"MOZA device started: {moza_path}")
                success = True
            except Exception as e:
                logging.error(f"Failed to start MOZA: {e}")
        else:
            logging.warning("MOZA device not found")
        
        # Start Arduino thread
        arduino_config = self.config.get('arduino', {})
        arduino_path = arduino_config.get('event') or self._find_device('arduino')
        
        if arduino_path:
            try:
                self.arduino_device = InputDevice(arduino_path)
                self.arduino_device.grab()
                thread = threading.Thread(target=self._arduino_loop, args=(arduino_config,), daemon=True)
                thread.start()
                self._threads.append(thread)
                logging.info(f"Arduino device started: {arduino_path}")
                success = True
            except Exception as e:
                logging.error(f"Failed to start Arduino: {e}")
        else:
            logging.warning("Arduino device not found")
        
        return success
    
    def stop(self):
        """Stop all device threads."""
        self._running = False
        
        for thread in self._threads:
            thread.join(timeout=1.0)
        
        if self.moza_device:
            try:
                self.moza_device.ungrab()
                self.moza_device.close()
            except:
                pass
                
        if self.arduino_device:
            try:
                self.arduino_device.ungrab()
                self.arduino_device.close()
            except:
                pass
        
        self.gamepad.reset()
        self.gamepad.update()
    
    def _moza_loop(self, config):
        """Handle MOZA stalk events."""
        button_map = config.get('button_map', {})
        left_on = button_map.get('left_on', 297)
        right_on = button_map.get('right_on', 295)
        cancel = button_map.get('left_off', 296)  # Shared cancel code
        hazard = button_map.get('hazard_push', 291)
        
        try:
            for event in self.moza_device.read_loop():
                if not self._running:
                    break
                    
                if event.type != ecodes.EV_KEY or event.value != 1:  # Only key press
                    continue
                    
                code = event.code
                
                # Left blinker toggle
                if code == left_on:
                    if not self._left_active:
                        self._left_active = True
                        self._right_active = False
                        self._pulse_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)
                        logging.info("MOZA: Left blinker ON -> LB pressed")
                
                # Right blinker toggle
                elif code == right_on:
                    if not self._right_active:
                        self._right_active = True
                        self._left_active = False
                        self._pulse_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
                        logging.info("MOZA: Right blinker ON -> RB pressed")
                
                # Cancel (shared for both)
                elif code == cancel:
                    if self._left_active:
                        self._left_active = False
                        self._pulse_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)
                        logging.info("MOZA: Left blinker OFF -> LB pressed")
                    elif self._right_active:
                        self._right_active = False
                        self._pulse_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
                        logging.info("MOZA: Right blinker OFF -> RB pressed")
                
                # Hazard lights
                elif code == hazard:
                    self._pulse_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_Y)
                    logging.info("MOZA: Hazard -> Y pressed")
                    
        except Exception as e:
            logging.error(f"MOZA loop error: {e}")
    
    def _arduino_loop(self, config):
        """Handle Arduino seatbelt events."""
        contact_code = config.get('contact_code', 288)
        invert = config.get('invert', False)
        
        try:
            for event in self.arduino_device.read_loop():
                if not self._running:
                    break
                    
                if event.type != ecodes.EV_KEY or event.code != contact_code:
                    continue
                    
                # Determine seatbelt state
                pressed = (event.value == 1)
                fastened = (not pressed) if invert else pressed
                
                # Only act on state changes
                if fastened != self._seatbelt_fastened:
                    self._seatbelt_fastened = fastened
                    
                    if fastened:
                        # Seatbelt fastened - Press A button
                        self._pulse_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
                        logging.info("Arduino: Seatbelt FASTENED -> A pressed")
                    else:
                        # Seatbelt unfastened - Press B button
                        self._pulse_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
                        logging.info("Arduino: Seatbelt UNFASTENED -> B pressed")
                        
        except Exception as e:
            logging.error(f"Arduino loop error: {e}")
    
    def _pulse_button(self, button):
        """Press and release a button with a short delay."""
        self.gamepad.press_button(button=button)
        self.gamepad.update()
        time.sleep(0.1)
        self.gamepad.release_button(button=button)
        self.gamepad.update()


# Test script
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 70)
    print("HARDWARE VIRTUAL GAMEPAD TEST")
    print("=" * 70)
    print("\nThis bridges MOZA stalks and Arduino seatbelt to virtual gamepad.")
    print("Press Ctrl+C to exit.\n")
    
    # Create and start the bridge
    bridge = HardwareVirtualGamepad()
    
    print("Starting hardware bridge...")
    if not bridge.start():
        print("✗ Failed to start bridge (no devices found)")
        sys.exit(1)
    
    print("✓ Bridge started successfully!")
    print("\n" + "=" * 70)
    print("BUTTON MAPPING:")
    print("=" * 70)
    print("MOZA:")
    print("  Left Stalk  -> Left Shoulder Button (LB)")
    print("  Right Stalk -> Right Shoulder Button (RB)")
    print("  Hazard      -> Y Button")
    print("\nARDUINO:")
    print("  Seatbelt Fastened   -> A Button")
    print("  Seatbelt Unfastened -> B Button")
    print("\n" + "=" * 70)
    print("TEST INSTRUCTIONS:")
    print("=" * 70)
    print("1. Move MOZA stalks - should see button messages")
    print("2. Trigger seatbelt sensor - should see A/B button messages")
    print("3. Run 'jstest /dev/input/js*' in another terminal to verify")
    print("\nMonitoring hardware events... (Ctrl+C to exit)")
    print("-" * 70)
    
    try:
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        bridge.stop()
        print("✓ Bridge stopped")
        print("Goodbye!")