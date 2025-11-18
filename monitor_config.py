# monitor_config.py
"""
Monitor Configuration Utility for Q-DRIVE Cortex
Helps users set up complex monitor arrangements including ultra-wide displays.

Usage: python monitor_config.py [--test] [--interactive]
"""
# ============================================================================
# PERF CHECK (file-level):
# ============================================================================
# [X] | Role: Monitor configuration utility (CLI tool, NOT in hot path)
# [ ] | Hot-path functions: None (interactive setup wizard)
# [ ] |- Heavy allocs in hot path? N/A - not in hot path
# [ ] |- pandas/pyarrow/json/disk/net in hot path? JSON config read/write
# [ ] | Graphics here? No (CLI only)
# [ ] | Data produced (tick schema?): None
# [X] | Storage (Parquet/Arrow/CSV/none): JSON (monitor configs)
# [ ] | Queue/buffer used?: No
# [ ] | Session-aware? No
# [ ] | Debug-only heavy features?: None
# Top 3 perf risks:
# 1. [PERF_OK] NOT in hot path - utility/setup tool only
# 2. [PERF_OK] xrandr subprocess calls acceptable for setup
# 3. [PERF_OK] Interactive UI loops acceptable
# ============================================================================

import sys
import json
from Utility.Monitor import DynamicMonitor

class MonitorConfigUtility:
    
    def __init__(self):
        self.monitor_system = DynamicMonitor("1920x1080")  # Default resolution for testing
        
    def interactive_configuration(self):
        """
        Interactive setup wizard for monitor configuration.
        """
        print("=" * 70)
        print("🎮 Q-DRIVE CORTEX MONITOR CONFIGURATION WIZARD")
        print("=" * 70)
        
        # Step 1: Detect current monitors
        print("\n📡 STEP 1: Detecting connected monitors...")
        monitors = self.monitor_system.get_monitor_layout()
        
        if not monitors:
            print("❌ No monitors detected. Please check your connections.")
            return False
            
        # Step 2: Show current setup
        print(f"\n🖥️  STEP 2: Current setup analysis")
        layout_info = self.monitor_system.analyze_panoramic_layout(monitors)
        self.monitor_system.print_configuration_summary()
        
        # Step 3: Resolution selection
        print(f"\n🎯 STEP 3: Choose simulation resolution")
        resolution_options = [
            "1920x1080",  # Full HD
            "2560x1440",  # 1440p
            "1680x1050",  # 16:10 aspect
            "1440x900"    # Lower end option
        ]
        
        print("Available simulation resolutions:")
        for i, res in enumerate(resolution_options):
            print(f"  {i+1}) {res}")
        print(f"  {len(resolution_options)+1}) Custom resolution")
        
        while True:
            try:
                choice = input(f"\nSelect resolution (1-{len(resolution_options)+1}): ").strip()
                choice_idx = int(choice) - 1
                
                if 0 <= choice_idx < len(resolution_options):
                    selected_resolution = resolution_options[choice_idx]
                    break
                elif choice_idx == len(resolution_options):
                    custom_res = input("Enter custom resolution (WxH): ").strip()
                    if self._validate_resolution(custom_res):
                        selected_resolution = custom_res
                        break
                    else:
                        print("❌ Invalid resolution format. Use 'WIDTHxHEIGHT' format.")
                else:
                    print(f"❌ Please enter a number between 1 and {len(resolution_options)+1}")
            except ValueError:
                print("❌ Please enter a valid number")
        
        # Step 4: Ultra-wide configuration
        ultrawide_monitors = [m for m in monitors if m['is_ultrawide']]
        if ultrawide_monitors:
            print(f"\n🖼️  STEP 4: Ultra-wide monitor configuration")
            for monitor in ultrawide_monitors:
                self._configure_ultrawide_monitor(monitor)
        
        # Step 5: Test configuration
        print(f"\n🧪 STEP 5: Test configuration")
        print(f"Selected resolution: {selected_resolution}")
        
        test_choice = input("Would you like to test this configuration? (y/n): ").strip().lower()
        if test_choice in ['y', 'yes']:
            return self._test_configuration(selected_resolution, monitors)
        else:
            print("Configuration completed without testing.")
            return True
    
    def _configure_ultrawide_monitor(self, monitor):
        """
        Configure how an ultra-wide monitor should be handled.
        """
        print(f"\n📺 Configuring ultra-wide monitor: {monitor['name']} ({monitor['resolution_str']})")
        
        config = monitor['ultrawide_config']
        if config and config.get('split_count', 1) > 1:
            print(f"   Current config: Split into {config['split_count']} virtual displays")
            print(f"   Each virtual display: {config['split_width']}x{config['split_height']}")
        else:
            print("   Current config: Use as single display")
        
        choice = input("Keep this configuration? (y/n): ").strip().lower()
        if choice not in ['y', 'yes']:
            print("\nUltra-wide options:")
            print("  1) Keep as single display")
            print("  2) Split into 2 virtual displays")
            print("  3) Custom configuration")
            
            while True:
                try:
                    option = int(input("Choose option (1-3): ").strip())
                    if option == 1:
                        monitor['ultrawide_config'] = {"split_count": 1}
                        break
                    elif option == 2:
                        split_width = monitor['w'] // 2
                        monitor['ultrawide_config'] = {
                            "split_count": 2, 
                            "split_width": split_width, 
                            "split_height": monitor['h']
                        }
                        break
                    elif option == 3:
                        self._custom_ultrawide_config(monitor)
                        break
                    else:
                        print("❌ Please enter 1, 2, or 3")
                except ValueError:
                    print("❌ Please enter a valid number")
    
    def _custom_ultrawide_config(self, monitor):
        """
        Allow custom ultra-wide configuration.
        """
        print(f"\nCustom configuration for {monitor['name']} ({monitor['w']}x{monitor['h']}):")
        
        while True:
            try:
                split_count = int(input("Number of virtual displays (1-4): ").strip())
                if 1 <= split_count <= 4:
                    break
                else:
                    print("❌ Split count must be between 1 and 4")
            except ValueError:
                print("❌ Please enter a valid number")
        
        if split_count > 1:
            split_width = monitor['w'] // split_count
            monitor['ultrawide_config'] = {
                "split_count": split_count,
                "split_width": split_width,
                "split_height": monitor['h']
            }
            print(f"✅ Configured: {split_count} virtual displays of {split_width}x{monitor['h']} each")
        else:
            monitor['ultrawide_config'] = {"split_count": 1}
            print("✅ Configured as single display")
    
    def _validate_resolution(self, resolution_str):
        """
        Validate resolution string format.
        """
        try:
            w, h = map(int, resolution_str.split('x'))
            return w > 0 and h > 0
        except:
            return False
    
    def _test_configuration(self, resolution, monitors):
        """
        Test the monitor configuration.
        """
        print(f"\n🧪 Testing configuration with resolution {resolution}...")
        
        # Store original layout
        original_layout = monitors.copy()
        
        try:
            # Apply the new configuration
            test_monitor = DynamicMonitor(resolution)
            success = test_monitor.arrange_monitors_for_panoramic(resolution, monitors)
            
            if success:
                input("\n✅ Configuration applied! Check your monitors and press ENTER to restore original layout...")
            else:
                print("\n❌ Configuration failed!")
            
            # Restore original layout
            print("🔄 Restoring original monitor layout...")
            test_monitor.restore_monitor_layout(original_layout)
            
            return success
            
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            # Attempt to restore anyway
            try:
                test_monitor = DynamicMonitor(resolution)
                test_monitor.restore_monitor_layout(original_layout)
            except:
                print("⚠️  Warning: Could not restore original layout. You may need to manually reset your display settings.")
            return False
    
    def quick_setup_for_user_config(self):
        """
        Quick setup specifically for the user's described configuration:
        1. UWQHD monitor @ 3440x1440
        2. 49" monitor @ 3840x1080  
        3. 24 inch monitor @ 1920x1080
        """
        print("🚀 Quick Setup for Your Configuration")
        print("   1. UWQHD @ 3440x1440")
        print("   2. 49\" Ultra-wide @ 3840x1080")
        print("   3. 24\" @ 1920x1080")
        
        # Update the ultrawide configs for this specific case
        self.monitor_system._ultrawide_configs.update({
            "3840x1080": {"split_count": 2, "split_width": 1920, "split_height": 1080},
            "3440x1440": {"split_count": 1, "split_width": 3440, "split_height": 1440}
        })
        
        monitors = self.monitor_system.get_monitor_layout()
        if len(monitors) >= 3:
            layout_info = self.monitor_system.analyze_panoramic_layout(monitors)
            
            if layout_info['total_logical'] >= 4:
                print("✅ Perfect! Your setup provides 4+ logical displays for full panoramic view.")
                resolution = input("Enter simulation resolution (default 1920x1080): ").strip() or "1920x1080"
                return self._test_configuration(resolution, monitors)
            else:
                print("⚠️  Your setup provides fewer than 4 logical displays. The system will use dual or single mode.")
                
        else:
            print("❌ Expected 3 monitors but found fewer. Please check connections.")
            
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Q-DRIVE Cortex Monitor Configuration Utility")
    parser.add_argument("--test", action="store_true", help="Test current monitor configuration")
    parser.add_argument("--interactive", action="store_true", help="Run interactive configuration wizard")
    parser.add_argument("--quick-setup", action="store_true", help="Quick setup for described configuration")
    parser.add_argument("--resolution", default="1920x1080", help="Target simulation resolution")
    
    args = parser.parse_args()
    
    config_util = MonitorConfigUtility()
    
    if args.quick_setup:
        config_util.quick_setup_for_user_config()
    elif args.interactive:
        config_util.interactive_configuration()
    elif args.test:
        monitor_system = DynamicMonitor(args.resolution)
        monitors = monitor_system.get_monitor_layout()
        monitor_system.print_configuration_summary()
    else:
        # Default: show current configuration
        monitor_system = DynamicMonitor(args.resolution)
        monitors = monitor_system.get_monitor_layout()
        if monitors:
            monitor_system.print_configuration_summary()
        else:
            print("No monitors detected. Use --interactive for setup wizard.")

if __name__ == "__main__":
    main()