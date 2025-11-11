import re
import subprocess
import os
import sys
import time
import logging
from FontIconLibrary import IconLibrary

ilib = IconLibrary()

class DynamicMonitor(object):

    def __init__(self, simulation_resolution):
        self._sim_res = simulation_resolution
        self._original_layout = None

    def get_monitor_layout(self):
        """
        Parses xrandr output to get a list of connected monitors, including their
        name, resolution, position, primary status, and refresh rate.

        Returns:
            A list of dictionaries, sorted left-to-right by the 'x' coordinate.
        """
        monitors = []
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
                
                monitors.append({
                    "name": conn_match.group(1),
                    "primary": bool(conn_match.group(2)),
                    "w": int(conn_match.group(3)),
                    "h": int(conn_match.group(4)),
                    "x": int(conn_match.group(5)),
                    "y": int(conn_match.group(6)),
                    "rate": float(rate_match.group(1)) if rate_match else None
                })
            
            monitors.sort(key=lambda m: m['x'])
            return monitors
        except Exception as e:
            print(f"🚨 An error occurred while getting monitor layout: {e}")
            return []


    def arrange_monitors_horizontally(self,resolution_str, monitors):
        """
        Builds and executes a single xrandr command to arrange monitors horizontally.
        """
        if not monitors:
            print("No connected monitors found to arrange.")
            return
        
        try:
            target_w, _ = map(int, resolution_str.split('x'))
        except ValueError:
            print(f"❌ Invalid resolution format: '{resolution_str}'. Use 'WIDTHxHEIGHT'.")
            return

        xrandr_cmd = ['xrandr']
        current_x_pos = 0
        
        for i, monitor in enumerate(monitors):
            xrandr_cmd.extend(['--output', monitor['name'], '--mode', resolution_str, '--pos', f"{current_x_pos}x0"])
            
            # ADDED: Check if a refresh rate was detected and apply it.
            if monitor['rate']:
                xrandr_cmd.extend(['--rate', str(monitor['rate'])])

            if i == 0:
                xrandr_cmd.append('--primary')
            current_x_pos += target_w
        
        print("🚀 Applying new horizontal layout...")
        ilib.ilog("warning", f"width = {resolution_str}",'alerts','wn',3)
        try:
            subprocess.run(xrandr_cmd, check=True)
            print("✅ Layout successfully applied!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to apply new layout. Error: {e}")

    # ==============================================================================
    # -- GLOBAL Functions| ----------------------------------------------------------
    #--------------------|
    ### Dynamic Display Configuration
    # ------------------------------------------------------------------------------
    # ==============================================================================


    def restore_monitor_layout(self,original_layout):
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