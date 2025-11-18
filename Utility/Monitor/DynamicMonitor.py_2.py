import re
import subprocess
import os
import sys
import time
import logging
from Utility.Font.FontIconLibrary import IconLibrary

ilib = IconLibrary()

class DynamicMonitor(object):

    def __init__(self, simulation_resolution):
        self._sim_res = simulation_resolution
        self._original_layout = None
        self.total_logical_displays = None
        # Ultra-wide monitor configurations
        # Define known ultra-wide resolutions and how to handle them
        self._ultrawide_configs = {
            # 32:9 ultra-wides that split into two 16:9 displays
            "7680x2160": {"split_count": 2, "split_width": 3840, "split_height": 2160},  # Two 4K displays
            "3840x1080": {"split_count": 2, "split_width": 1920, "split_height": 1080},  # Two FHD displays
            # 32:9 ultra-wide at 1440p
            "5120x1440": {"split_count": 2, "split_width": 2560, "split_height": 1440},
            
            # 21:9 ultra-wides (treat as single)
            "3440x1440": {"split_count": 1, "split_width": 3440, "split_height": 1440},
            "2560x1080": {"split_count": 1, "split_width": 2560, "split_height": 1080},
            
            # Standard 16:9 displays (no split needed)
            "3840x2160": {"split_count": 1, "split_width": 3840, "split_height": 2160},  # 4K
            "2560x1440": {"split_count": 1, "split_width": 2560, "split_height": 1440},
            "1920x1080": {"split_count": 1, "split_width": 1920, "split_height": 1080}
            # Removed duplicate 3840x1080 entry
        }
        # Monitor role assignments for panoramic layout
        self._monitor_roles = []

    def get_monitor_layout(self):
        """
        Enhanced monitor layout detection with NVIDIA viewport support.
        Parses xrandr output and checks for NVIDIA viewport transformations.
        
        Returns:
            A list of dictionaries, sorted left-to-right by the 'x' coordinate.
        """
        monitors = []
        
        # First, standard xrandr detection
        try:
            xrandr_output = subprocess.check_output(['xrandr', '--query']).decode('utf-8')
            
            # Regex to split the output into blocks for each monitor interface
            monitor_blocks = re.findall(r'(\S+ connected(?:.|\n)*?(?=\n\S+ connected|\Z))', xrandr_output)

            for block in monitor_blocks:
                # Find the main connection line
                conn_match = re.search(r"(\S+) connected (primary )?(\d+)x(\d+)\+(\d+)\+(\d+)", block)
                if not conn_match:
                    continue

                # Find the active refresh rate for the current mode
                rate_match = re.search(r'^\s+\d+x\d+\s+.*?\s([\d\.]+)\*\s.*$', block, re.MULTILINE)
                
                w, h = int(conn_match.group(3)), int(conn_match.group(4))
                resolution_str = f"{w}x{h}"
                
                monitor_data = {
                    "name": conn_match.group(1),
                    "primary": bool(conn_match.group(2)),
                    "w": w,
                    "h": h,
                    "x": int(conn_match.group(5)),
                    "y": int(conn_match.group(6)),
                    "rate": float(rate_match.group(1)) if rate_match else None,
                    "resolution_str": resolution_str,
                    "original_resolution": resolution_str,
                    "viewport_scaled": False
                }
                
                monitors.append(monitor_data)
                
        except Exception as e:
            print(f"🚨 An error occurred while getting monitor layout: {e}")
            return []
        
        # Check NVIDIA viewport configurations
        try:
            # Query nvidia-settings for viewport information
            nvidia_cmd = ['nvidia-settings', '-q', 'CurrentMetaMode', '-t']
            result = subprocess.run(nvidia_cmd, capture_output=True, text=True, stderr=subprocess.DEVNULL)
            
            if result.returncode == 0:
                metamode = result.stdout
                
                # Check for viewport configurations
                if 'ViewPortIn=3840x1080' in metamode and 'ViewPortOut=1920x1080' in metamode:
                    print(f"🔧 Detected NVIDIA viewport scaling: 3840x1080 → 1920x1080")
                    
                    # Find center monitor (typically between x=1920 and x < 4000)
                    for monitor in monitors:
                        if 1900 < monitor['x'] < 4000:
                            print(f"  Updating {monitor['name']} from {monitor['w']}x{monitor['h']} to 3840x1080")
                            monitor['w'] = 3840
                            monitor['h'] = 1080
                            monitor['resolution_str'] = "3840x1080"
                            monitor['viewport_scaled'] = True
                            break
                            
                elif 'ViewPortIn=5120x1440' in metamode:
                    print(f"🔧 Detected 5120x1440 viewport configuration")
                    for monitor in monitors:
                        if 1900 < monitor['x'] < 6000:
                            monitor['w'] = 5120
                            monitor['h'] = 1440
                            monitor['resolution_str'] = "5120x1440"
                            monitor['viewport_scaled'] = True
                            break
                            
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass  # nvidia-settings not available
        except Exception as e:
            print(f"⚠️ NVIDIA viewport check failed: {e}")
        
        # Environment variable override
        viewport_override = os.environ.get('ULTRAWIDE_OVERRIDE')
        if viewport_override:
            try:
                # Support both lowercase and uppercase X
                viewport_override = viewport_override.replace('X', 'x')
                w, h = map(int, viewport_override.split('x'))
                for monitor in monitors:
                    if 1900 < monitor['x'] < 6000:  # Center monitor
                        print(f"🔧 Manual ULTRAWIDE_OVERRIDE: {w}x{h}")
                        monitor['w'] = w
                        monitor['h'] = h
                        monitor['resolution_str'] = f"{w}x{h}"
                        monitor['viewport_scaled'] = True
                        break
            except Exception as e:
                print(f"⚠️ Invalid ULTRAWIDE_OVERRIDE format: {e}")
        
        # Now continue with ultra-wide detection using the corrected resolutions
        for monitor in monitors:
            w, h = monitor['w'], monitor['h']
            resolution_str = monitor['resolution_str']
            
            # Detect ultra-wide monitors
            aspect_ratio = w / h if h > 0 else 1.0
            is_ultrawide = (aspect_ratio > 2.1) or (resolution_str in self._ultrawide_configs)
            ultrawide_config = self._ultrawide_configs.get(resolution_str, {})
            
            monitor['aspect_ratio'] = aspect_ratio
            monitor['is_ultrawide'] = is_ultrawide
            monitor['ultrawide_config'] = ultrawide_config
            
            viewport_note = " [VIEWPORT-SCALED]" if monitor.get('viewport_scaled') else ""
            print(f"📺 Detected: {monitor['name']} @ {resolution_str} (AR: {aspect_ratio:.2f}) "
                  f"{'[ULTRA-WIDE]' if is_ultrawide else ''}{viewport_note}")
        
        monitors.sort(key=lambda m: m['x'])
        return monitors

    def analyze_panoramic_layout(self, monitors):
        """
        Analyzes the monitor setup and determines the best panoramic configuration.
        
        Returns:
            Dict with layout strategy and monitor assignments
        """
        if not monitors:
            return {"strategy": "single", "assignments": []}
        
        # Count effective displays (treating ultra-wides as multiple logical displays)
        total_logical_displays = 0
        monitor_assignments = []
        current_logical_x = 0  # Track logical X position in target resolution units
        
        try:
            target_w, target_h = map(int, self._sim_res.split('x'))
        except ValueError:
            target_w, target_h = 1920, 1080  # Fallback
        
        for monitor in monitors:
            if monitor['is_ultrawide'] and monitor['ultrawide_config'].get('split_count', 1) > 1:
                split_count = monitor['ultrawide_config']['split_count']
                total_logical_displays += split_count
                
                # Create virtual assignments for split ultra-wide
                for i in range(split_count):
                    virtual_x = current_logical_x + (i * target_w)  # Use target resolution for spacing
                    assignment = {
                        "physical_monitor": monitor['name'],
                        "logical_index": len(monitor_assignments),
                        "virtual_x": virtual_x,
                        "virtual_y": 0,
                        "virtual_w": target_w,  # All displays use target width
                        "virtual_h": target_h,  # All displays use target height
                        "is_virtual_split": True,
                        "split_index": i
                    }
                    monitor_assignments.append(assignment)
                
                # Advance by the number of virtual displays created
                current_logical_x += split_count * target_w
            else:
                total_logical_displays += 1
                assignment = {
                    "physical_monitor": monitor['name'],
                    "logical_index": len(monitor_assignments),
                    "virtual_x": current_logical_x,
                    "virtual_y": 0,
                    "virtual_w": target_w,  # Use target width, not original width
                    "virtual_h": target_h,  # Use target height, not original height
                    "is_virtual_split": False,
                    "split_index": 0
                }
                monitor_assignments.append(assignment)
                
                # Advance by one target width
                current_logical_x += target_w
        
        # Determine layout strategy
        if total_logical_displays >= 4:
            strategy = "quad"
        elif total_logical_displays >= 2:
            strategy = "dual"
        else:
            strategy = "single"
            
        print(f"🎯 Layout Analysis: {total_logical_displays} logical displays → {strategy.upper()} mode")
        self.total_logical_displays = total_logical_displays
        return {
            "strategy": strategy,
            "assignments": monitor_assignments,
            "total_logical": total_logical_displays
        }

    def arrange_monitors_for_panoramic(self, target_resolution, monitors):
        """
        Enhanced monitor arrangement that handles ultra-wide monitors and mixed resolutions.
        """
        if not monitors:
            print("No connected monitors found to arrange.")
            return False
        
        layout_info = self.analyze_panoramic_layout(monitors)
        
        try:
            target_w, target_h = map(int, target_resolution.split('x'))
        except ValueError:
            print(f"⌐ Invalid resolution format: '{target_resolution}'. Use 'WIDTHxHEIGHT'.")
            return False

        print(f"\n🚀 Configuring monitors for {layout_info['strategy']} panoramic layout...")
        
        # Phase 1: Configure each physical monitor
        xrandr_commands = []
        current_x_pos = 0
        
        for i, monitor in enumerate(monitors):
            cmd = ['xrandr', '--output', monitor['name']]
            
            # CRITICAL FIX: Handle your specific ultra-wide differently
            if monitor['resolution_str'] == "7680x2160":
                # This ultra-wide stays at native res but will provide 2 logical displays
                cmd.extend(['--mode', monitor['resolution_str']])
                print(f"  📺 {monitor['name']}: @ {monitor['resolution_str']} (native for 2x split)")
                split_count = 2
            elif monitor['is_ultrawide'] and monitor['ultrawide_config'].get('split_count', 1) > 1:
                # Other ultra-wides that need splitting
                cmd.extend(['--mode', monitor['resolution_str']])
                print(f"  📺 {monitor['name']}: @ {monitor['resolution_str']} (native for virtual split)")
                split_count = monitor['ultrawide_config']['split_count']
            else:
                # Standard monitors: use target resolution
                cmd.extend(['--mode', target_resolution])
                print(f"  📺 {monitor['name']}: @ {target_resolution}")
                split_count = 1
            
            # Position the monitor
            cmd.extend(['--pos', f"{current_x_pos}x0"])
            
            # Add refresh rate if detected
            if monitor.get('rate'):
                cmd.extend(['--rate', str(monitor['rate'])])
            
            # Set primary monitor (first one)
            if i == 0:
                cmd.append('--primary')
            
            xrandr_commands.append(cmd)
            
            # Advance position based on actual resolution used
            if monitor['resolution_str'] == "7680x2160":
                current_x_pos += 3840 * 2  # Two 3840-wide segments
                
            if split_count > 1:
                # For splits, use the split_width from config
                split_width = monitor['ultrawide_config'].get('split_width', target_w)
                current_x_pos += split_width * split_count
            else:
                # Standard monitor uses target width
                current_x_pos += target_w
        # Execute all xrandr commands
        success = True
        for cmd in xrandr_commands:
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"⌐ Failed to configure monitor. Command: {' '.join(cmd)}")
                print(f"   Error: {e}")
                success = False
        
        if success:
            print("✓ Panoramic layout successfully applied!")
            self._log_final_layout(layout_info)
            self._last_layout_info = layout_info
        
        return success

    def _log_final_layout(self, layout_info):
        """Log the final panoramic layout configuration."""
        print(f"\n📋 Final Panoramic Configuration ({layout_info['strategy']} mode):")
        for i, assignment in enumerate(layout_info['assignments']):
            if assignment['is_virtual_split']:
                print(f"  Camera {i}: Virtual display on {assignment['physical_monitor']} "
                      f"(split {assignment['split_index']}) @ {assignment['virtual_x']},{assignment['virtual_y']}")
            else:
                print(f"  Camera {i}: {assignment['physical_monitor']} @ "
                      f"{assignment['virtual_x']},{assignment['virtual_y']}")

    def arrange_monitors_horizontally(self, resolution_str, monitors):
        """
        Legacy method - now routes to the enhanced panoramic arrangement.
        Maintained for backward compatibility.
        """
        print("🔄 Using enhanced panoramic monitor arrangement...")
        return self.arrange_monitors_for_panoramic(resolution_str, monitors)

    def get_layout_strategy(self):
        """
        Get the recommended layout strategy based on the last monitor configuration.
        
        Returns:
            str: 'single', 'dual', or 'quad'
        """
        if hasattr(self, '_last_layout_info') and self._last_layout_info:
            return self._last_layout_info['strategy']
        return 'single'  # Default fallback
    
    def get_total_logical_displays(self):
        """
        Get the number of logical displays available.
        
        Returns:
            int: Number of logical displays
        """
        if hasattr(self, '_last_layout_info') and self._last_layout_info:
            return self._last_layout_info['total_logical']
        return 1  # Default fallback

    def get_ultrawide_virtual_layout(self):
        """
        Returns the virtual layout information for ultra-wide monitors.
        This can be used by HUD.py to understand how to position camera views.
        """
        if hasattr(self, '_last_layout_info'):
            return self._last_layout_info
        return None

    def restore_monitor_layout(self, original_layout):
        """
        Restores the original monitor layout using a more forceful "off-then-on"
        method for each monitor to ensure picky displays reset correctly.
        """
        if not original_layout:
            print("No original layout to restore.")
            return

        print("\nRestoring original monitor layout with hard reset method...")
        primary_monitor_name = None

        # First, apply settings to each monitor individually
        for monitor in original_layout:
            # --- The New "Hard Reset" Logic ---
            try:
                # 1. Turn the monitor output OFF completely
                print(f"Resetting {monitor['name']}...")
                subprocess.run(['xrandr', '--output', monitor['name'], '--off'], check=True)
                time.sleep(0.5) # A brief pause

                # 2. Turn it back ON with all original settings
                xrandr_cmd = ['xrandr', '--output', monitor['name']]
                res_str = f"{monitor['w']}x{monitor['h']}"
                pos_str = f"{monitor['x']}x{monitor['y']}"
                xrandr_cmd.extend(['--mode', res_str, '--pos', pos_str])
                
                if monitor['rate']:
                    xrandr_cmd.extend(['--rate', str(monitor['rate'])])

                subprocess.run(xrandr_cmd, check=True)
                time.sleep(0.5) # Another pause for adjustment

            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to restore {monitor['name']}. Error: {e}")
            # --- End of New Logic ---
            
            if monitor['primary']:
                primary_monitor_name = monitor['name']

        # After all monitors are configured, set the primary display
        if primary_monitor_name:
            try:
                print(f"Setting {primary_monitor_name} as primary...")
                subprocess.run(['xrandr', '--output', primary_monitor_name, '--primary'], check=True)
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to set primary display. Error: {e}")

        print("✅ Original layout restored.")

    def print_configuration_summary(self):
        """
        Prints a detailed summary of the current monitor configuration.
        Useful for debugging and user information.
        """
        monitors = self.get_monitor_layout()
        layout_info = self.analyze_panoramic_layout(monitors)
        
        print("\n" + "="*60)
        print("📺 MONITOR CONFIGURATION SUMMARY")
        print("="*60)
        
        print(f"Target Resolution: {self._sim_res}")
        print(f"Layout Strategy: {layout_info['strategy'].upper()}")
        print(f"Total Logical Displays: {layout_info['total_logical']}")
        
        print(f"\nPhysical Monitors ({len(monitors)}):")
        for monitor in monitors:
            status = "PRIMARY" if monitor['primary'] else "SECONDARY"
            ultrawide_note = " [ULTRA-WIDE]" if monitor['is_ultrawide'] else ""
            print(f"  • {monitor['name']}: {monitor['resolution_str']} @ {monitor['x']},{monitor['y']} ({status}){ultrawide_note}")
            if monitor['rate']:
                print(f"    Refresh Rate: {monitor['rate']}Hz")
        
        print(f"\nCamera View Assignments:")
        camera_names = ['left_side_cam', 'left_dash_cam', 'right_dash_cam', 'right_side_cam']
        for i, assignment in enumerate(layout_info['assignments'][:4]):  # Only show first 4 for quad layout
            camera_name = camera_names[i] if i < len(camera_names) else f"camera_{i}"
            if assignment['is_virtual_split']:
                print(f"  • {camera_name}: Virtual split {assignment['split_index']} on {assignment['physical_monitor']}")
            else:
                print(f"  • {camera_name}: {assignment['physical_monitor']}")
        
        print("="*60)