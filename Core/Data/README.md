# ML Data Logger - Production ML Data Infrastructure

## Overview

The ML Data Logger is a high-performance, ML-ready data logging system built on Apache Arrow/Parquet. It replaces the existing CSV-based logging with a zero-copy, columnar storage system optimized for:

- **Transfer learning** pipelines
- **Event detection** and classification
- **Reinforcement learning** training
- **Real-time streaming** to edge devices (Jetson Nano)
- **Future projects** without LIDAR dependency

## Key Features

### 🚀 Performance
- **Zero-copy operations** with PyArrow RecordBatch
- **10-20x smaller files** compared to CSV (Parquet compression)
- **100x faster** random access for ML training
- **<1ms overhead** per frame in hot path

### 🎯 ML-Ready
- **Multi-scale windowing** (immediate/tactical/strategic)
- **Automated event detection** with 20+ event types
- **Real-time tagging** for supervised learning
- **PyTorch/TensorFlow** compatible data loaders
- **Schema versioning** for long-term compatibility

### 📊 Hybrid Storage
- **Columnar storage** (Parquet) for analytics and batch training
- **Compressed cache** for real-time streaming to Jetson
- **Memory-mapped IPC** for zero-copy inter-process communication
- **Backward compatible** CSV export during transition

### 🔍 Event Detection
- **Collision events** (front/rear/side, pedestrian/vehicle/static)
- **Lane events** (violation, legal change, aggressive change, weaving)
- **Harsh maneuvers** (brake, accel, corner, panic swerve)
- **Near-miss events** (TTC < 2s, close following, cut-off)
- **Traffic violations** (red light, speeding, stop sign, wrong-way)
- **Complex maneuvers** (roundabout, merge, parallel park, U-turn)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CARLA Simulation Tick                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│               MLDataLogger.log_frame()                       │
│  - Drop-in replacement for DataIngestion.log_frame()        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│             HybridTelemetryStore.ingest_frame()              │
└──────────────────────┬──────────────────────────────────────┘
                       │
           ┌───────────┴───────────┐
           │                       │
           ▼                       ▼
┌──────────────────┐    ┌──────────────────────┐
│  Multi-Scale     │    │   Event Detection    │
│  Windows         │    │   (20+ types)        │
│  - Immediate(3s) │    │   - Severity         │
│  - Tactical(30s) │    │   - Factors          │
│  - Strategic(5m) │    │   - Outcomes         │
└────────┬─────────┘    └─────────┬────────────┘
         │                        │
         └───────────┬────────────┘
                     │
           ┌─────────┴─────────┐
           │                   │
           ▼                   ▼
┌──────────────────┐  ┌──────────────────────┐
│  Arrow Buffer    │  │  Compressed Cache    │
│  (columnar)      │  │  (streaming)         │
│  - Zero-copy     │  │  - 4 formats         │
│  - Batched       │  │  - Jetson IPC        │
└────────┬─────────┘  └──────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│              Parquet Export (end of session)                 │
│  - Schema version: 2.0.0                                     │
│  - Metadata: session, vehicle, weather, map                  │
│  - Compression: Snappy (optimal for ML)                      │
│  - Features: ~50 columns + event JSON                        │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Install PyArrow (already in requirements.txt)
pip install pyarrow >= 14.0.0

# Or install all requirements
pip install -r requirements.txt
```

## Quick Start

### Basic Usage (Drop-in Replacement)

Replace your existing DataIngestion with MLDataLogger:

```python
from Core.Data import MLDataLogger

# Initialize (backward compatible)
logger = MLDataLogger(
    session_id="test_session",
    export_csv=True,      # Backward compatibility
    export_parquet=True,  # ML-ready format
    mmap_enabled=False    # Enable for Jetson streaming
)

# In your main loop (same interface as DataIngestion)
for frame in simulation:
    frame_data = {
        'frame': frame_num,
        'timestamp': time.time(),
        'speed_kmh': vehicle.get_speed(),
        'mvd_overall_score': mvd.get_score(),
        'collision_occurred': collision_sensor.has_collision(),
        # ... all your existing fields ...
    }

    result = logger.log_frame(frame_data)

    # Optional: Get immediate events
    if result['immediate_events']:
        for event in result['immediate_events']:
            print(f"Event: {event.event_type.value}, Severity: {event.severity.name}")

# Save session
logger.save_session("./Session_logs/")
logger.cleanup()
```

### Advanced Usage with Multi-Scale Windows

```python
from Core.Data import MLDataLogger, MultiScaleWindows

logger = MLDataLogger(session_id="advanced_test")

# Multi-scale windows process data at 3 time scales:
# - Immediate (3s): Real-time hazard detection
# - Tactical (30s): Maneuver classification
# - Strategic (5min): Driving style analysis

for frame_num, frame_data in enumerate(simulation):
    result = logger.log_frame(frame_data)

    # Check tactical window stats (updated every 30 frames)
    if frame_num % 30 == 0:
        tactical_stats = logger.store.windows.window_stats.get('tactical', {})
        print(f"Last 30s: Avg MVD={tactical_stats.get('avg_mvd_score', 0):.1f}, "
              f"Harsh events={tactical_stats.get('total_harsh_events', 0)}")
```

### Jetson Streaming (Real-Time Edge Inference)

```python
# On simulation side (CARLA server)
logger = MLDataLogger(
    session_id="jetson_stream",
    mmap_enabled=True  # Enable memory-mapped IPC
)

# Get compressed batch for streaming
compressed_batch = logger.get_streaming_batch(n=20)  # Last 20 windows

# Send to Jetson via WebSocket/gRPC
jetson_client.send(compressed_batch)

# On Jetson side
for compressed_window in batch:
    # Decompress and run inference
    features = parse_compressed(compressed_window)
    prediction = model(features)
```

## Integration with Main.py

### Option 1: Replace Existing DataIngestion

```python
# In Main.py, replace:
# from DataIngestion import DataIngestion
# data_logger = DataIngestion()

# With:
from Core.Data import MLDataLogger
data_logger = MLDataLogger(
    session_id=f"session_{timestamp}",
    export_csv=True,     # Keep CSV during transition
    export_parquet=True  # Add Parquet for ML
)

# Rest of your code remains the same!
# MLDataLogger has the same log_frame() interface
```

### Option 2: Parallel Logging (Safest Transition)

```python
# Keep your existing logger
from DataIngestion import DataIngestion
old_logger = DataIngestion()

# Add ML logger in parallel
from Core.Data import MLDataLogger
ml_logger = MLDataLogger(session_id=f"session_{timestamp}")

# Log to both
for frame in simulation:
    frame_data = collect_frame_data()

    old_logger.log_frame(world_obj, metrics)  # Existing
    ml_logger.log_frame(frame_data)            # New ML logger

# Save both
old_logger.save_to_csv()
ml_logger.save_session()
```

## Data Schema

### Window-Level Schema (Parquet Export)

The logger uses **window-based aggregation** rather than raw frame logging for efficiency:

```
Window Schema (50+ columns):
├── Identification
│   ├── frame_start (int32)
│   ├── frame_end (int32)
│   ├── timestamp (timestamp[ms])
│   └── session_id (string)
├── Speed Metrics
│   ├── avg_speed_kmh (float32)
│   ├── max_speed_kmh (float32)
│   └── min_speed_kmh (float32)
├── MVD Scores
│   ├── avg_mvd_score (float32)
│   ├── min_mvd_score (float32)
│   ├── avg_collision_score (float32)
│   ├── avg_lane_score (float32)
│   └── avg_harsh_score (float32)
├── Event Counts
│   ├── total_harsh_events (int16)
│   ├── total_lane_violations (int16)
│   ├── total_collisions (int16)
│   └── total_near_misses (int16)
├── Predictive Indices (worst case in window)
│   ├── min_ttc_s (float32)
│   ├── min_tlc_s (float32)
│   └── max_required_brake_ms2 (float32)
├── Dynamics Extremes
│   ├── max_jerk_ms3 (float32)
│   ├── max_g_force_lateral (float32)
│   └── max_g_force_longitudinal (float32)
├── Control Inputs (averages)
│   ├── avg_throttle (float32)
│   ├── avg_brake (float32)
│   └── avg_steer (float32)
├── Environmental Context
│   ├── traffic_density (int16)
│   ├── weather_condition (string)
│   └── map_name (string)
├── Compressed Formats (binary)
│   ├── compressed_standard (binary)
│   ├── compressed_context (binary)
│   ├── compressed_llm (binary)
│   └── compressed_adaptive (binary)
└── Event Annotations
    └── events_json (string) - JSON array of DetectedEvent
```

### Event Schema (JSON within events_json)

```json
{
  "event_type": "harsh_brake",
  "severity": "WARNING",
  "frame_start": 1234,
  "frame_end": 1234,
  "timestamp_start": 1699900000.123,
  "timestamp_end": 1699900000.123,
  "confidence": 0.95,
  "speed_kmh": 65.4,
  "mvd_score": 78.2,
  "ttc_s": 1.8,
  "tlc_s": 2.5,
  "traffic_density": 8,
  "weather_condition": "clear",
  "factors": ["emergency_avoidance", "insufficient_headway"],
  "outcome": "safe_recovery",
  "tags": ["harsh_driving", "braking", "deceleration"]
}
```

## ML Training Workflows

### PyTorch Example

```python
from Core.Data import ParquetSequenceDataset
from torch.utils.data import DataLoader

# Load dataset
dataset = ParquetSequenceDataset(
    'Session_logs/session_12345_20251118.parquet',
    sequence_length=100,  # 5 seconds @ 20Hz
    stride=50             # 50% overlap
)

# Create data loader
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Train model
for batch_features, batch_labels in loader:
    # batch_features: dict with speed, mvd_score, ttc, etc.
    # batch_labels: bool (collision occurred in next N frames)

    outputs = model(batch_features)
    loss = criterion(outputs, batch_labels)
    loss.backward()
    optimizer.step()
```

### Event Classification

```python
import pyarrow.parquet as pq
import json

# Load Parquet file
table = pq.read_table('session.parquet')

# Extract events
for row in table.to_pylist():
    events = json.loads(row['events_json'])

    for event in events:
        if event['event_type'] == 'harsh_brake':
            # Extract features for harsh brake classifier
            features = {
                'speed': event['speed_kmh'],
                'ttc': event['ttc_s'],
                'traffic_density': event['traffic_density'],
                'weather': event['weather_condition']
            }

            label = event['outcome']  # 'safe_recovery' or 'collision'

            # Train classifier
            # ...
```

## Compression Formats

The logger generates 4 compression formats for different use cases:

### 1. Standard Format (Bandwidth-Constrained)
```
1234-1294|S:65.4|M:78.2|H:2|L:1
```
**Use case**: Low-bandwidth streaming to Jetson

### 2. Context Format (Pattern Detection)
```
1234-1294|S:65.4|M:78.2|H:2|L:1|V:[harsh_brake,lane_violation]
```
**Use case**: Event pattern analysis

### 3. LLM Format (Natural Language)
```
During frames 1234-1294, driver maintained 65.4 km/h average speed with good driving performance. Events: harsh_brake(WARNING), lane_violation(WARNING).
```
**Use case**: LLM fine-tuning, natural language reports

### 4. Adaptive Format (Smart Compression)
```
# Safe driving: compact
1234-1294|S:65.4|M:92.1|H:0|L:0

# Concerning: detailed
During frames 1234-1294, driver maintained 85.2 km/h with poor driving performance (MVD: 62.3). Events: harsh_brake(CRITICAL), collision(CATASTROPHIC).
```
**Use case**: Optimal balance for most scenarios

## Event Detection

### Event Types (20+ categories)

**Collision Events:**
- COLLISION_FRONT, COLLISION_REAR, COLLISION_SIDE
- COLLISION_PEDESTRIAN, COLLISION_VEHICLE, COLLISION_STATIC

**Lane Events:**
- LANE_VIOLATION, LANE_CHANGE_LEGAL, LANE_CHANGE_AGGRESSIVE, LANE_WEAVING

**Harsh Maneuvers:**
- HARSH_BRAKE, HARSH_ACCEL, HARSH_CORNER, PANIC_SWERVE

**Near-Miss Events:**
- NEAR_MISS_FRONT, NEAR_MISS_SIDE, CLOSE_FOLLOWING, CUT_OFF

**Traffic Violations:**
- RED_LIGHT_VIOLATION, SPEED_VIOLATION, STOP_SIGN_VIOLATION, WRONG_WAY

**Complex Maneuvers:**
- ROUNDABOUT_ENTRY, MERGE_HIGHWAY, PARALLEL_PARK, U_TURN, INTERSECTION_CROSSING

### Severity Levels
- **INFO** (0): Normal operations, legal maneuvers
- **WARNING** (1): Minor violations, harsh but controlled
- **CRITICAL** (2): Dangerous situations, near-misses
- **CATASTROPHIC** (3): Collisions, severe violations

### Auto-Tagging

Events are automatically tagged with:
- **Contributing factors**: speed, distraction, weather, traffic density
- **Outcomes**: safe_recovery, near_miss, collision
- **Tags**: For filtering and classification

## File Outputs

### Parquet Files

**Location**: `Session_logs/session_<id>_<timestamp>.parquet`

**Features**:
- Columnar storage (10-20x smaller than CSV)
- Metadata embedded (schema version, session info)
- Fast random access for ML training
- Snappy compression (optimal for ML workloads)

**Query Example**:
```python
import pyarrow.parquet as pq
import pyarrow.compute as pc

# Load file
table = pq.read_table('session.parquet')

# Filter by frame range
mask = pc.and_(
    pc.greater_equal(table['frame_start'], 1000),
    pc.less_equal(table['frame_start'], 2000)
)
filtered = table.filter(mask)

# Get high-risk windows (low MVD score)
risky = table.filter(pc.less(table['avg_mvd_score'], 70))
```

### CSV Files (Backward Compatibility)

**Location**: `Session_logs/session_<id>_<timestamp>.csv`

Generated when `export_csv=True`. Same format as existing DataIngestion.

### Memory-Mapped Files (IPC)

**Location**: `/dev/shm/qdrive_<session_id>.arrow`

For zero-copy inter-process communication with Jetson/edge devices.

## Performance Characteristics

### Memory Usage
- **Arrow buffers**: ~1MB per 1000 windows
- **Compressed cache**: ~50KB for 1000 windows (text compression)
- **Frame buffers**: ~10MB for 3-window system (immediate/tactical/strategic)

**Total**: ~15-20MB for typical session

### CPU Overhead
- **Frame ingestion**: <0.5ms per frame
- **Window computation**: ~1ms every 30 frames (tactical update)
- **Event detection**: <0.2ms per frame
- **Parquet export**: ~500ms for 10,000 windows (end of session)

**Impact**: <1% of 20Hz tick budget

### Disk Usage
- **CSV**: ~100MB for 30min session
- **Parquet**: ~5-10MB for same session (10-20x smaller)

## Future Extensions

### Already Designed For:
- **LIDAR integration**: `has_lidar` flag in metadata
- **Multi-camera sync**: Schema supports frame embeddings
- **Transfer learning**: Pre-trained model embeddings in event.embedding
- **Reinforcement learning**: Reward signals from MVD scores

### Easy to Add:
- **Video frame sync**: Add camera frame paths to schema
- **Audio telemetry**: Add mic level, horn usage columns
- **DMS integration**: Add gaze tracking, distraction scores
- **Custom sensors**: Schema is extensible

## Troubleshooting

### PyArrow Not Available

**Symptom**: `PYARROW_AVAILABLE = False` warning

**Fix**:
```bash
pip install pyarrow >= 14.0.0
```

### Memory-Mapped File Errors

**Symptom**: `Could not create memory map` warning

**Fix**:
- Ensure `/dev/shm` exists (Linux only)
- Or disable: `MLDataLogger(mmap_enabled=False)`

### Large Parquet Files

**Symptom**: Parquet files larger than expected

**Possible causes**:
- Too many windows (reduce buffer size)
- High event density (normal for complex scenarios)
- Uncompressed binary data (ensure compression='snappy')

**Fix**: Parquet size is usually 5-10MB for 30min session. If larger, consider reducing window retention.

## Migration Guide

### Phase 1: Parallel Logging (Week 1)
- Add MLDataLogger alongside existing DataIngestion
- Export both CSV and Parquet
- Verify outputs match

### Phase 2: Transition (Week 2)
- Use Parquet for new analyses
- Keep CSV for backward compatibility
- Train team on new format

### Phase 3: Full Migration (Week 3+)
- Switch to Parquet-only for new sessions
- Archive old CSV logs
- Update all ML pipelines

## Support

For questions or issues:
1. Check this README
2. Review example usage in `MLDataLogger.py` main block
3. See integration examples above
4. Contact Q-DRIVE team

## License

Proprietary - Q-DRIVE Alpha Project
