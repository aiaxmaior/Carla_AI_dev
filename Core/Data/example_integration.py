"""
Example integration of MLDataLogger with Main.py

This shows three integration strategies:
1. Drop-in replacement (simplest)
2. Parallel logging (safest for transition)
3. Advanced usage with event callbacks

Author: Q-DRIVE Team
"""

import time
import logging
from Core.Data import MLDataLogger, EventType, EventSeverity

logging.basicConfig(level=logging.INFO)


# ============================================================================
# STRATEGY 1: Drop-in Replacement
# ============================================================================

def example_drop_in_replacement():
    """
    Simplest integration: Replace DataIngestion with MLDataLogger

    In Main.py, change:
        from DataIngestion import DataIngestion
        data_logger = DataIngestion()

    To:
        from Core.Data import MLDataLogger
        data_logger = MLDataLogger(session_id="test", export_csv=True)

    Everything else stays the same!
    """

    print("\n" + "="*70)
    print("STRATEGY 1: Drop-in Replacement")
    print("="*70)

    # Initialize (same interface as DataIngestion)
    logger = MLDataLogger(
        session_id="drop_in_test",
        export_csv=True,       # Backward compatibility
        export_parquet=True,   # ML-ready format
    )

    # Simulate main loop (same as your existing code)
    for frame_num in range(100):
        # Collect frame data (your existing code structure)
        frame_data = {
            'frame': frame_num,
            'timestamp': time.time(),
            'speed_kmh': 60.0 + frame_num * 0.1,
            'mvd_overall_score': 85.0 - frame_num * 0.05,
            'collision_occurred': False,
            'ttc_s': 5.0,
            'tlc_s': 3.0,
            'acceleration_x': -0.5,
            'acceleration_lateral': 0.1,
            'nearby_vehicles_count': 5,
            'weather_cloudiness': 20,
            'weather_precipitation': 0,
            # ... all your other fields ...
        }

        # Log frame (SAME INTERFACE as DataIngestion.log_frame)
        result = logger.log_frame(frame_data)

        # Optional: Check for immediate events
        if result and result.get('immediate_events'):
            for event in result['immediate_events']:
                print(f"  Frame {frame_num}: {event.event_type.value} ({event.severity.name})")

    # Save session (same as DataIngestion.save_to_csv)
    logger.save_session("./Session_logs/")
    logger.cleanup()

    print("✓ Drop-in replacement complete!")
    print("  - CSV saved for backward compatibility")
    print("  - Parquet saved for ML workflows")


# ============================================================================
# STRATEGY 2: Parallel Logging (Safest Transition)
# ============================================================================

def example_parallel_logging():
    """
    Safest integration: Run both loggers in parallel during transition

    This lets you:
    1. Keep your existing CSV pipeline working
    2. Start generating Parquet files for ML
    3. Verify outputs match before switching
    """

    print("\n" + "="*70)
    print("STRATEGY 2: Parallel Logging")
    print("="*70)

    # Your existing logger (keep it!)
    # from DataIngestion import DataIngestion
    # old_logger = DataIngestion()

    # New ML logger (add it!)
    ml_logger = MLDataLogger(
        session_id="parallel_test",
        export_csv=False,      # Old logger handles CSV
        export_parquet=True,   # ML logger handles Parquet
    )

    # Simulate main loop
    for frame_num in range(100):
        frame_data = {
            'frame': frame_num,
            'timestamp': time.time(),
            'speed_kmh': 70.0,
            'mvd_overall_score': 90.0,
            # ... your fields ...
        }

        # Log to both systems
        # old_logger.log_frame(world_obj, metrics)  # Your existing code
        ml_logger.log_frame(frame_data)             # New ML logger

    # Save both
    # old_logger.save_to_csv()         # Existing
    ml_logger.save_session()           # New

    ml_logger.cleanup()

    print("✓ Parallel logging complete!")
    print("  - Old CSV pipeline still works")
    print("  - New Parquet files ready for ML")


# ============================================================================
# STRATEGY 3: Advanced with Event Callbacks
# ============================================================================

def example_advanced_with_events():
    """
    Advanced integration: React to detected events in real-time

    Use cases:
    - Trigger HUD warnings
    - Send alerts to driver
    - Adjust scenario difficulty
    - Log critical events separately
    """

    print("\n" + "="*70)
    print("STRATEGY 3: Advanced with Event Callbacks")
    print("="*70)

    logger = MLDataLogger(session_id="advanced_test")

    # Event handler (your custom logic)
    def handle_critical_event(event):
        """React to critical events in real-time"""
        if event.severity.value >= EventSeverity.CRITICAL.value:
            print(f"  🚨 CRITICAL: {event.event_type.value} at frame {event.frame_start}")
            print(f"      Speed: {event.speed_kmh:.1f} km/h, MVD: {event.mvd_score:.1f}")
            print(f"      Factors: {', '.join(event.factors)}")

            # Your custom actions:
            # - Update HUD warning
            # - Send to EventManager
            # - Trigger audio alert
            # - etc.

    # Simulate scenario with events
    for frame_num in range(200):
        # Simulate harsh braking event at frame 100
        if frame_num == 100:
            frame_data = {
                'frame': frame_num,
                'timestamp': time.time(),
                'speed_kmh': 80.0,
                'mvd_overall_score': 65.0,
                'acceleration_forward': -7.0,  # Harsh brake!
                'collision_occurred': False,
                'ttc_s': 1.5,  # Near-miss!
                'tlc_s': 2.0,
                'nearby_vehicles_count': 8,
                'weather_cloudiness': 40,
            }
        # Simulate collision at frame 150
        elif frame_num == 150:
            frame_data = {
                'frame': frame_num,
                'timestamp': time.time(),
                'speed_kmh': 60.0,
                'mvd_overall_score': 45.0,
                'collision_occurred': True,
                'collision_intensity': 2500,  # Severe!
                'collision_actor_type': 'vehicle',
                'ttc_s': 0.0,
                'tlc_s': 1.5,
            }
        else:
            # Normal driving
            frame_data = {
                'frame': frame_num,
                'timestamp': time.time(),
                'speed_kmh': 60.0,
                'mvd_overall_score': 85.0,
                'acceleration_forward': 0.0,
                'collision_occurred': False,
                'ttc_s': 8.0,
                'tlc_s': 5.0,
            }

        # Log frame
        result = logger.log_frame(frame_data)

        # Handle detected events
        if result and result.get('immediate_events'):
            for event in result['immediate_events']:
                handle_critical_event(event)

        # Check window statistics (every 30 frames)
        if frame_num % 30 == 0 and frame_num > 0:
            tactical_stats = logger.store.windows.window_stats.get('tactical', {})
            if tactical_stats:
                print(f"  Frame {frame_num}: Last 30s - "
                      f"MVD: {tactical_stats.get('avg_mvd_score', 0):.1f}, "
                      f"Harsh events: {tactical_stats.get('total_harsh_events', 0)}")

    # Save session
    logger.save_session()
    logger.cleanup()

    print("✓ Advanced integration complete!")
    print("  - Events detected and handled in real-time")
    print("  - Window statistics tracked")


# ============================================================================
# STRATEGY 4: Jetson Streaming
# ============================================================================

def example_jetson_streaming():
    """
    Stream compressed telemetry to Jetson Nano for edge inference

    Use cases:
    - Real-time driver coaching
    - Edge AI inference
    - Distributed processing
    """

    print("\n" + "="*70)
    print("STRATEGY 4: Jetson Streaming")
    print("="*70)

    # Enable memory-mapped IPC
    logger = MLDataLogger(
        session_id="jetson_stream",
        mmap_enabled=True  # Creates /dev/shm/qdrive_jetson_stream.arrow
    )

    # Simulate data collection
    for frame_num in range(150):
        frame_data = {
            'frame': frame_num,
            'timestamp': time.time(),
            'speed_kmh': 65.0,
            'mvd_overall_score': 82.0,
        }

        logger.log_frame(frame_data)

        # Stream to Jetson every 10 frames
        if frame_num % 10 == 0 and frame_num > 0:
            # Get compressed batch
            batch = logger.get_streaming_batch(n=20)

            print(f"  Frame {frame_num}: Streaming {len(batch)} windows to Jetson")

            # In real implementation:
            # - Send via WebSocket: jetson_ws.send(json.dumps(batch))
            # - Or via gRPC: jetson_client.StreamTelemetry(batch)
            # - Or read from mmap on Jetson side

            # Show compressed format
            if batch:
                print(f"    Last window (compressed): {batch[-1][:80]}...")

    logger.cleanup()

    print("✓ Jetson streaming complete!")
    print("  - Compressed data ready for edge inference")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("MLDataLogger Integration Examples")
    print("="*70)

    # Run all examples
    try:
        example_drop_in_replacement()
        example_parallel_logging()
        example_advanced_with_events()
        example_jetson_streaming()

        print("\n" + "="*70)
        print("All examples completed successfully!")
        print("="*70)
        print("\nNext steps:")
        print("1. Install pyarrow: pip install pyarrow >= 14.0.0")
        print("2. Choose integration strategy (Strategy 1 recommended to start)")
        print("3. Update Main.py with chosen strategy")
        print("4. Run test session and verify outputs")
        print("5. Check Session_logs/ for .parquet files")
        print("\nFor ML training:")
        print("- Use ParquetSequenceDataset for PyTorch")
        print("- Query Parquet files with pyarrow.parquet")
        print("- Extract events from events_json column")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure PyArrow is installed: pip install pyarrow")
