#!/usr/bin/env python3
"""
Test script for DynamicMonitor skip-first-monitor functionality
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Utility.Monitor.DynamicMonitor import DynamicMonitor

def test_skip_logic():
    print("=" * 60)
    print("Testing DynamicMonitor skip-first-monitor logic")
    print("=" * 60)

    # Test without skip
    print("\n🔍 Test 1: Normal mode (no skip)")
    print("-" * 60)
    dm1 = DynamicMonitor("1920x1080", skip_first_monitor=False)
    monitors1 = dm1.get_monitor_layout()
    print(f"\n✅ Found {len(monitors1)} monitor(s)")
    for i, m in enumerate(monitors1):
        print(f"   {i}: {m['name']} @ {m['resolution_str']}")

    # Test with skip
    print("\n🔍 Test 2: Skip-first mode")
    print("-" * 60)
    dm2 = DynamicMonitor("1920x1080", skip_first_monitor=True)
    monitors2 = dm2.get_monitor_layout()
    print(f"\n✅ Found {len(monitors2)} monitor(s)")
    for i, m in enumerate(monitors2):
        print(f"   {i}: {m['name']} @ {m['resolution_str']}")

    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"Normal mode:     {len(monitors1)} monitors")
    print(f"Skip-first mode: {len(monitors2)} monitors")

    if len(monitors1) > 3:
        if len(monitors2) == len(monitors1) - 1:
            print("✅ Skip logic working correctly (>3 monitors, first skipped)")
        else:
            print("❌ Skip logic error")
    else:
        if len(monitors2) == len(monitors1):
            print("✅ Skip logic working correctly (≤3 monitors, none skipped)")
        else:
            print("❌ Skip logic error")

    print("=" * 60)

if __name__ == "__main__":
    test_skip_logic()
