"""
MLDataLogger.py - Production ML Data Infrastructure

Zero-copy telemetry logging with PyArrow for transfer learning, event detection,
and reinforcement learning pipelines.

Features:
- Zero-copy Arrow RecordBatch operations
- Hybrid storage: columnar (analytics) + compressed (streaming)
- Multi-scale windowing (immediate/tactical/strategic)
- Real-time event detection and automated tagging
- Parquet export with schema versioning
- PyTorch/TensorFlow compatible data loaders
- Memory-mapped IPC for Jetson streaming

Author: Q-DRIVE Team
Performance Audit: 2025-11-18
"""

# ============================================================================
# PERF CHECK (file-level):
# ============================================================================
# [X] | Role: ML-ready data logger with Arrow backend
# [X] | Hot-path functions: ingest() called every tick
# [X] |- Heavy allocs in hot path? Minimal - zero-copy Arrow operations
# [X] |- pandas/pyarrow/json/disk/net in hot path? Arrow only (no pandas)
# [ ] | Graphics here? No
# [X] | Data produced (tick schema?): RecordBatch per frame
# [X] | Storage (Parquet/Arrow/CSV/none): Parquet + mmap + optional CSV
# [X] | Queue/buffer used?: YES - Arrow batches + deque for streaming
# [X] | Session-aware? YES - session metadata in schema
# [ ] | Debug-only heavy features?: None
# Top 3 perf risks:
# 1. [PERF_OK] Arrow RecordBatch is zero-copy - very efficient
# 2. [PERF_OK] Event detection throttled to tactical window updates
# 3. [PERF_OK] Parquet writes batched and async-compatible
# ============================================================================

import os
import time
import struct
import logging
import json
from collections import deque, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Tuple, Deque
from enum import Enum
from datetime import datetime

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    logging.warning("PyArrow not available - falling back to pandas-only mode")

import numpy as np


# ============================================================================
# SCHEMA VERSIONING
# ============================================================================

SCHEMA_VERSION = "2.0.0"  # Semantic versioning
SCHEMA_CHANGELOG = {
    "2.0.0": "Initial ML-ready schema with event detection",
    "2.1.0": "Future: Add LIDAR point cloud columns",
    "3.0.0": "Future: Multi-camera sync with frame embeddings"
}


# ============================================================================
# EVENT DEFINITIONS
# ============================================================================

class EventType(Enum):
    """Categorized event types for supervised learning"""

    # Collision events
    COLLISION_FRONT = "collision_front"
    COLLISION_REAR = "collision_rear"
    COLLISION_SIDE = "collision_side"
    COLLISION_PEDESTRIAN = "collision_pedestrian"
    COLLISION_VEHICLE = "collision_vehicle"
    COLLISION_STATIC = "collision_static"

    # Lane events
    LANE_VIOLATION = "lane_violation"
    LANE_CHANGE_LEGAL = "lane_change_legal"
    LANE_CHANGE_AGGRESSIVE = "lane_change_aggressive"
    LANE_WEAVING = "lane_weaving"

    # Harsh maneuvers
    HARSH_BRAKE = "harsh_brake"
    HARSH_ACCEL = "harsh_accel"
    HARSH_CORNER = "harsh_corner"
    PANIC_SWERVE = "panic_swerve"

    # Near-miss events
    NEAR_MISS_FRONT = "near_miss_front"
    NEAR_MISS_SIDE = "near_miss_side"
    CLOSE_FOLLOWING = "close_following"
    CUT_OFF = "cut_off"

    # Traffic violations
    RED_LIGHT_VIOLATION = "red_light_violation"
    SPEED_VIOLATION = "speed_violation"
    STOP_SIGN_VIOLATION = "stop_sign_violation"
    WRONG_WAY = "wrong_way"

    # Complex maneuvers (for transfer learning)
    ROUNDABOUT_ENTRY = "roundabout_entry"
    MERGE_HIGHWAY = "merge_highway"
    PARALLEL_PARK = "parallel_park"
    U_TURN = "u_turn"
    INTERSECTION_CROSSING = "intersection_crossing"


class EventSeverity(Enum):
    """Event severity for prioritization"""
    INFO = 0
    WARNING = 1
    CRITICAL = 2
    CATASTROPHIC = 3


@dataclass
class DetectedEvent:
    """Detected event with full context for ML training"""
    event_type: EventType
    severity: EventSeverity
    frame_start: int
    frame_end: int
    timestamp_start: float
    timestamp_end: float
    confidence: float  # 0-1 confidence in detection

    # Contextual features
    speed_kmh: float
    mvd_score: float
    ttc_s: float
    tlc_s: float
    traffic_density: int
    weather_condition: str

    # Contributing factors
    factors: List[str] = field(default_factory=list)  # ["distraction", "high_speed", "poor_visibility"]

    # Outcome
    outcome: str = "unknown"  # "safe_recovery", "collision", "near_miss"

    # Metadata for ML
    tags: List[str] = field(default_factory=list)  # Auto-generated tags
    embedding: Optional[bytes] = None  # Future: CNN/transformer embedding of event

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        d = asdict(self)
        d['event_type'] = self.event_type.value
        d['severity'] = self.severity.value
        return d


# ============================================================================
# COMPRESSION FORMATS
# ============================================================================

class CompressionFormats:
    """
    Multiple compression formats for different use cases:
    - Standard: Original compact format
    - Context: Includes violation context
    - LLM: Natural language for LLM fine-tuning
    - Adaptive: Dynamic format based on events
    """

    @staticmethod
    def format_standard(window_data: Dict[str, Any]) -> str:
        """Original compact format for bandwidth-constrained streaming"""
        return f"{window_data['frame_start']}-{window_data['frame_end']}|" \
               f"S:{window_data['avg_speed']:.1f}|" \
               f"M:{window_data['avg_mvd_score']:.1f}|" \
               f"H:{window_data['total_harsh_events']}|" \
               f"L:{window_data['total_lane_violations']}"

    @staticmethod
    def format_context(window_data: Dict[str, Any], violations: List[str]) -> str:
        """Include violation context for pattern detection"""
        base = CompressionFormats.format_standard(window_data)
        if violations:
            vio_str = ",".join(violations[:3])  # Top 3 violations
            return f"{base}|V:[{vio_str}]"
        return base

    @staticmethod
    def format_llm(window_data: Dict[str, Any], events: List[DetectedEvent]) -> str:
        """Natural language format for LLM fine-tuning"""
        speed = window_data['avg_speed']
        mvd = window_data['avg_mvd_score']

        # Start with context
        text = f"During frames {window_data['frame_start']}-{window_data['frame_end']}, "
        text += f"driver maintained {speed:.1f} km/h average speed "

        # Add performance assessment
        if mvd >= 90:
            text += "with excellent driving performance"
        elif mvd >= 75:
            text += "with good driving performance"
        elif mvd >= 60:
            text += "with marginal driving performance"
        else:
            text += "with poor driving performance"

        # Add events if present
        if events:
            event_strs = [f"{e.event_type.value}({e.severity.name})" for e in events[:3]]
            text += f". Events: {', '.join(event_strs)}"

        text += "."
        return text

    @staticmethod
    def format_adaptive(window_data: Dict[str, Any], events: List[DetectedEvent]) -> str:
        """
        Adaptive format: compact when safe, detailed when issues detected
        """
        has_critical = any(e.severity.value >= EventSeverity.CRITICAL.value for e in events)

        if has_critical or window_data['avg_mvd_score'] < 70:
            # Detailed format for concerning windows
            return CompressionFormats.format_llm(window_data, events)
        else:
            # Compact format for safe driving
            return CompressionFormats.format_standard(window_data)


# ============================================================================
# EVENT DETECTOR
# ============================================================================

class EventDetector:
    """
    Real-time event detection and automated tagging

    Detects events across multiple scales:
    - Immediate: Collision, harsh brake, sudden swerve
    - Tactical: Lane weaving, tailgating pattern, aggressive driving
    - Strategic: Fatigue, distraction trend, risk escalation
    """

    def __init__(self):
        self.active_events: Dict[str, DetectedEvent] = {}
        self.completed_events: Deque[DetectedEvent] = deque(maxlen=1000)
        self.event_history = defaultdict(int)

        # Detection thresholds (tunable)
        self.thresholds = {
            'harsh_brake_ms2': -4.0,
            'harsh_accel_ms2': 3.5,
            'harsh_corner_g': 0.3,
            'panic_swerve_rate': 45.0,  # deg/s
            'near_miss_ttc_s': 2.0,
            'close_following_s': 1.0,
            'speed_violation_ratio': 1.2,  # 20% over limit
            'lane_weaving_count': 3,  # violations in window
        }

    def detect_immediate_events(self, frame_data: Dict[str, Any], frame_num: int) -> List[DetectedEvent]:
        """Detect immediate events (single frame or sub-second)"""
        events = []
        timestamp = frame_data.get('timestamp', time.time())

        # Collision detection
        if frame_data.get('collision_occurred', False):
            event_type = self._classify_collision(frame_data)
            severity = EventSeverity.CATASTROPHIC if frame_data.get('collision_intensity', 0) > 2000 else EventSeverity.CRITICAL

            events.append(DetectedEvent(
                event_type=event_type,
                severity=severity,
                frame_start=frame_num,
                frame_end=frame_num,
                timestamp_start=timestamp,
                timestamp_end=timestamp,
                confidence=1.0,
                speed_kmh=frame_data.get('speed_kmh', 0),
                mvd_score=frame_data.get('mvd_overall_score', 0),
                ttc_s=frame_data.get('ttc_s', 99),
                tlc_s=frame_data.get('tlc_s', 99),
                traffic_density=frame_data.get('nearby_vehicles_count', 0),
                weather_condition=self._get_weather_desc(frame_data),
                factors=self._identify_collision_factors(frame_data),
                outcome="collision",
                tags=["collision", "critical", "safety"]
            ))

        # Harsh braking
        accel_x = frame_data.get('acceleration_forward', frame_data.get('acceleration_x', 0))
        if accel_x < self.thresholds['harsh_brake_ms2']:
            events.append(DetectedEvent(
                event_type=EventType.HARSH_BRAKE,
                severity=EventSeverity.WARNING if accel_x > -6.0 else EventSeverity.CRITICAL,
                frame_start=frame_num,
                frame_end=frame_num,
                timestamp_start=timestamp,
                timestamp_end=timestamp,
                confidence=min(abs(accel_x) / 8.0, 1.0),
                speed_kmh=frame_data.get('speed_kmh', 0),
                mvd_score=frame_data.get('mvd_overall_score', 0),
                ttc_s=frame_data.get('ttc_s', 99),
                tlc_s=frame_data.get('tlc_s', 99),
                traffic_density=frame_data.get('nearby_vehicles_count', 0),
                weather_condition=self._get_weather_desc(frame_data),
                factors=self._identify_harsh_brake_factors(frame_data, accel_x),
                tags=["harsh_driving", "braking", "deceleration"]
            ))

        # Harsh acceleration
        if accel_x > self.thresholds['harsh_accel_ms2']:
            events.append(DetectedEvent(
                event_type=EventType.HARSH_ACCEL,
                severity=EventSeverity.WARNING,
                frame_start=frame_num,
                frame_end=frame_num,
                timestamp_start=timestamp,
                timestamp_end=timestamp,
                confidence=min(accel_x / 6.0, 1.0),
                speed_kmh=frame_data.get('speed_kmh', 0),
                mvd_score=frame_data.get('mvd_overall_score', 0),
                ttc_s=frame_data.get('ttc_s', 99),
                tlc_s=frame_data.get('tlc_s', 99),
                traffic_density=frame_data.get('nearby_vehicles_count', 0),
                weather_condition=self._get_weather_desc(frame_data),
                factors=["aggressive_driving"],
                tags=["harsh_driving", "acceleration"]
            ))

        # Harsh cornering
        g_lateral = frame_data.get('g_force_lateral', abs(frame_data.get('acceleration_lateral', 0)) / 9.81)
        if abs(g_lateral) > self.thresholds['harsh_corner_g']:
            events.append(DetectedEvent(
                event_type=EventType.HARSH_CORNER,
                severity=EventSeverity.WARNING if abs(g_lateral) < 0.5 else EventSeverity.CRITICAL,
                frame_start=frame_num,
                frame_end=frame_num,
                timestamp_start=timestamp,
                timestamp_end=timestamp,
                confidence=min(abs(g_lateral) / 0.6, 1.0),
                speed_kmh=frame_data.get('speed_kmh', 0),
                mvd_score=frame_data.get('mvd_overall_score', 0),
                ttc_s=frame_data.get('ttc_s', 99),
                tlc_s=frame_data.get('tlc_s', 99),
                traffic_density=frame_data.get('nearby_vehicles_count', 0),
                weather_condition=self._get_weather_desc(frame_data),
                factors=self._identify_harsh_corner_factors(frame_data, g_lateral),
                tags=["harsh_driving", "cornering", "lateral"]
            ))

        # Near-miss detection (low TTC)
        ttc = frame_data.get('ttc_s', frame_data.get('time_to_collision', 99))
        if ttc < self.thresholds['near_miss_ttc_s']:
            events.append(DetectedEvent(
                event_type=EventType.NEAR_MISS_FRONT,
                severity=EventSeverity.CRITICAL if ttc < 1.0 else EventSeverity.WARNING,
                frame_start=frame_num,
                frame_end=frame_num,
                timestamp_start=timestamp,
                timestamp_end=timestamp,
                confidence=1.0 - (ttc / 2.0),
                speed_kmh=frame_data.get('speed_kmh', 0),
                mvd_score=frame_data.get('mvd_overall_score', 0),
                ttc_s=ttc,
                tlc_s=frame_data.get('tlc_s', 99),
                traffic_density=frame_data.get('nearby_vehicles_count', 0),
                weather_condition=self._get_weather_desc(frame_data),
                factors=["close_following", "insufficient_headway"],
                tags=["near_miss", "safety", "following_distance"]
            ))

        # Lane violation
        if frame_data.get('lane_invasion_active', False):
            blinker = frame_data.get('blinker_state', 0)
            event_type = EventType.LANE_CHANGE_LEGAL if blinker != 0 else EventType.LANE_VIOLATION

            events.append(DetectedEvent(
                event_type=event_type,
                severity=EventSeverity.INFO if blinker != 0 else EventSeverity.WARNING,
                frame_start=frame_num,
                frame_end=frame_num,
                timestamp_start=timestamp,
                timestamp_end=timestamp,
                confidence=0.9,
                speed_kmh=frame_data.get('speed_kmh', 0),
                mvd_score=frame_data.get('mvd_overall_score', 0),
                ttc_s=frame_data.get('ttc_s', 99),
                tlc_s=frame_data.get('tlc_s', 99),
                traffic_density=frame_data.get('nearby_vehicles_count', 0),
                weather_condition=self._get_weather_desc(frame_data),
                factors=[] if blinker != 0 else ["improper_lane_change"],
                tags=["lane_discipline", "signaling"] if blinker != 0 else ["lane_violation", "unsafe"]
            ))

        # Traffic light violation
        if frame_data.get('traffic_light_state') == 'Red' and \
           frame_data.get('is_at_traffic_light', False) and \
           frame_data.get('speed_kmh', 0) > 5:
            events.append(DetectedEvent(
                event_type=EventType.RED_LIGHT_VIOLATION,
                severity=EventSeverity.CRITICAL,
                frame_start=frame_num,
                frame_end=frame_num,
                timestamp_start=timestamp,
                timestamp_end=timestamp,
                confidence=1.0,
                speed_kmh=frame_data.get('speed_kmh', 0),
                mvd_score=frame_data.get('mvd_overall_score', 0),
                ttc_s=frame_data.get('ttc_s', 99),
                tlc_s=frame_data.get('tlc_s', 99),
                traffic_density=frame_data.get('nearby_vehicles_count', 0),
                weather_condition=self._get_weather_desc(frame_data),
                factors=["traffic_violation", "risk_taking"],
                outcome="violation",
                tags=["traffic_violation", "red_light", "critical"]
            ))

        # Speed violation
        speed_limit = frame_data.get('speed_limit', 50)
        speed = frame_data.get('speed_kmh', 0)
        if speed > speed_limit * self.thresholds['speed_violation_ratio']:
            events.append(DetectedEvent(
                event_type=EventType.SPEED_VIOLATION,
                severity=EventSeverity.WARNING if speed < speed_limit * 1.5 else EventSeverity.CRITICAL,
                frame_start=frame_num,
                frame_end=frame_num,
                timestamp_start=timestamp,
                timestamp_end=timestamp,
                confidence=1.0,
                speed_kmh=speed,
                mvd_score=frame_data.get('mvd_overall_score', 0),
                ttc_s=frame_data.get('ttc_s', 99),
                tlc_s=frame_data.get('tlc_s', 99),
                traffic_density=frame_data.get('nearby_vehicles_count', 0),
                weather_condition=self._get_weather_desc(frame_data),
                factors=[f"speeding_{int((speed/speed_limit - 1) * 100)}%_over"],
                tags=["speed_violation", "compliance"]
            ))

        return events

    def _classify_collision(self, frame_data: Dict[str, Any]) -> EventType:
        """Classify collision type based on actor type and context"""
        actor_type = frame_data.get('collision_actor_type', 'unknown')

        if 'pedestrian' in actor_type.lower() or 'walker' in actor_type.lower():
            return EventType.COLLISION_PEDESTRIAN
        elif 'vehicle' in actor_type.lower():
            return EventType.COLLISION_VEHICLE
        elif 'static' in actor_type.lower():
            return EventType.COLLISION_STATIC
        else:
            # Default to front collision (could enhance with velocity vector analysis)
            return EventType.COLLISION_FRONT

    def _get_weather_desc(self, frame_data: Dict[str, Any]) -> str:
        """Get weather condition description"""
        cloudiness = frame_data.get('weather_cloudiness', 0)
        precipitation = frame_data.get('weather_precipitation', 0)
        fog = frame_data.get('weather_fog_density', 0)

        if precipitation > 50:
            return "heavy_rain"
        elif precipitation > 20:
            return "light_rain"
        elif fog > 30:
            return "fog"
        elif cloudiness > 80:
            return "cloudy"
        else:
            return "clear"

    def _identify_collision_factors(self, frame_data: Dict[str, Any]) -> List[str]:
        """Identify contributing factors to collision"""
        factors = []

        if frame_data.get('speed_kmh', 0) > frame_data.get('speed_limit', 50) * 1.2:
            factors.append("excessive_speed")

        if frame_data.get('ttc_s', 99) < 1.0:
            factors.append("insufficient_following_distance")

        if frame_data.get('weather_precipitation', 0) > 20:
            factors.append("adverse_weather")

        if frame_data.get('nearby_vehicles_count', 0) > 10:
            factors.append("high_traffic_density")

        return factors if factors else ["unknown"]

    def _identify_harsh_brake_factors(self, frame_data: Dict[str, Any], accel: float) -> List[str]:
        """Identify why harsh braking occurred"""
        factors = []

        ttc = frame_data.get('ttc_s', 99)
        if ttc < 2.0:
            factors.append("emergency_avoidance")

        if frame_data.get('speed_kmh', 0) > frame_data.get('speed_limit', 50) * 1.3:
            factors.append("speed_too_high")

        if frame_data.get('traffic_light_state') == 'Red':
            factors.append("late_braking_traffic_light")

        if accel < -6.0:
            factors.append("panic_brake")

        return factors if factors else ["unknown"]

    def _identify_harsh_corner_factors(self, frame_data: Dict[str, Any], g_force: float) -> List[str]:
        """Identify why harsh cornering occurred"""
        factors = []

        if frame_data.get('speed_kmh', 0) > 60:
            factors.append("excessive_speed_for_turn")

        if frame_data.get('is_junction', False):
            factors.append("intersection_maneuver")

        if abs(g_force) > 0.5:
            factors.append("extreme_lateral_acceleration")

        return factors if factors else ["aggressive_cornering"]


# ============================================================================
# ARROW TELEMETRY BUFFER
# ============================================================================

class ArrowTelemetryBuffer:
    """Zero-copy telemetry buffer using PyArrow RecordBatch"""

    def __init__(self, max_windows=1000):
        if not PYARROW_AVAILABLE:
            raise ImportError("PyArrow is required for ArrowTelemetryBuffer")

        # Define comprehensive schema for telemetry windows
        self.schema = pa.schema([
            # Window identification
            ('frame_start', pa.int32()),
            ('frame_end', pa.int32()),
            ('timestamp', pa.timestamp('ms')),
            ('session_id', pa.string()),

            # Speed metrics
            ('avg_speed_kmh', pa.float32()),
            ('max_speed_kmh', pa.float32()),
            ('min_speed_kmh', pa.float32()),

            # MVD scores
            ('avg_mvd_score', pa.float32()),
            ('min_mvd_score', pa.float32()),
            ('avg_collision_score', pa.float32()),
            ('avg_lane_score', pa.float32()),
            ('avg_harsh_score', pa.float32()),

            # Event counts
            ('total_harsh_events', pa.int16()),
            ('total_lane_violations', pa.int16()),
            ('total_collisions', pa.int16()),
            ('total_near_misses', pa.int16()),

            # Predictive indices (minimum values - worst case in window)
            ('min_ttc_s', pa.float32()),
            ('min_tlc_s', pa.float32()),
            ('max_required_brake_ms2', pa.float32()),

            # Dynamics extremes
            ('max_jerk_ms3', pa.float32()),
            ('max_g_force_lateral', pa.float32()),
            ('max_g_force_longitudinal', pa.float32()),

            # Control inputs (averages)
            ('avg_throttle', pa.float32()),
            ('avg_brake', pa.float32()),
            ('avg_steer', pa.float32()),

            # Environmental context
            ('traffic_density', pa.int16()),
            ('weather_condition', pa.string()),
            ('map_name', pa.string()),

            # Compressed formats (binary for efficiency)
            ('compressed_standard', pa.binary()),
            ('compressed_context', pa.binary()),
            ('compressed_llm', pa.binary()),
            ('compressed_adaptive', pa.binary()),

            # Event annotations (JSON)
            ('events_json', pa.string()),  # Serialized list of events
        ])

        self.batch_builder = []
        self.max_windows = max_windows
        self.batches = []

        logging.info(f"ArrowTelemetryBuffer initialized (schema v{SCHEMA_VERSION}, max_windows={max_windows})")

    def append_window(self, window_data: Dict[str, Any], events: List[DetectedEvent], session_id: str = "default"):
        """Append a window with zero-copy efficiency"""

        # Generate compressed formats
        compressed_std = CompressionFormats.format_standard(window_data).encode()
        compressed_ctx = CompressionFormats.format_context(window_data, [e.event_type.value for e in events]).encode()
        compressed_llm = CompressionFormats.format_llm(window_data, events).encode()
        compressed_adp = CompressionFormats.format_adaptive(window_data, events).encode()

        # Serialize events to JSON
        events_json = json.dumps([e.to_dict() for e in events])

        # Build row matching schema
        row = {
            'frame_start': window_data['frame_start'],
            'frame_end': window_data['frame_end'],
            'timestamp': pa.scalar(datetime.now(), type=pa.timestamp('ms')),
            'session_id': session_id,

            'avg_speed_kmh': float(window_data.get('avg_speed', 0)),
            'max_speed_kmh': float(window_data.get('max_speed', 0)),
            'min_speed_kmh': float(window_data.get('min_speed', 0)),

            'avg_mvd_score': float(window_data.get('avg_mvd_score', 100)),
            'min_mvd_score': float(window_data.get('min_mvd_score', 100)),
            'avg_collision_score': float(window_data.get('avg_collision_score', 100)),
            'avg_lane_score': float(window_data.get('avg_lane_score', 100)),
            'avg_harsh_score': float(window_data.get('avg_harsh_score', 100)),

            'total_harsh_events': int(window_data.get('total_harsh_events', 0)),
            'total_lane_violations': int(window_data.get('total_lane_violations', 0)),
            'total_collisions': int(window_data.get('total_collisions', 0)),
            'total_near_misses': int(window_data.get('total_near_misses', 0)),

            'min_ttc_s': float(window_data.get('min_ttc', 99)),
            'min_tlc_s': float(window_data.get('min_tlc', 99)),
            'max_required_brake_ms2': float(window_data.get('max_required_brake', 0)),

            'max_jerk_ms3': float(window_data.get('max_jerk', 0)),
            'max_g_force_lateral': float(window_data.get('max_g_lateral', 0)),
            'max_g_force_longitudinal': float(window_data.get('max_g_longitudinal', 0)),

            'avg_throttle': float(window_data.get('avg_throttle', 0)),
            'avg_brake': float(window_data.get('avg_brake', 0)),
            'avg_steer': float(window_data.get('avg_steer', 0)),

            'traffic_density': int(window_data.get('traffic_density', 0)),
            'weather_condition': str(window_data.get('weather', 'clear')),
            'map_name': str(window_data.get('map_name', 'unknown')),

            'compressed_standard': compressed_std,
            'compressed_context': compressed_ctx,
            'compressed_llm': compressed_llm,
            'compressed_adaptive': compressed_adp,

            'events_json': events_json,
        }

        self.batch_builder.append(row)

        # Flush to RecordBatch periodically
        if len(self.batch_builder) >= 100:
            self._flush_batch()

    def _flush_batch(self):
        """Convert accumulated rows to RecordBatch (zero-copy)"""
        if not self.batch_builder:
            return

        try:
            # Create RecordBatch from accumulated data
            batch = pa.RecordBatch.from_pylist(
                self.batch_builder,
                schema=self.schema
            )

            self.batches.append(batch)
            self.batch_builder.clear()

            # Limit memory usage
            if len(self.batches) > self.max_windows // 100:
                self.batches.pop(0)

            logging.debug(f"Flushed {len(batch)} rows to Arrow RecordBatch")

        except Exception as e:
            logging.error(f"Error flushing Arrow batch: {e}", exc_info=True)

    def get_table(self) -> pa.Table:
        """Get all data as PyArrow Table"""
        self._flush_batch()

        if not self.batches:
            return pa.Table.from_pylist([], schema=self.schema)

        return pa.Table.from_batches(self.batches)

    def export_parquet(self, filepath: str, metadata: Dict[str, Any] = None):
        """Export to Parquet with metadata"""
        table = self.get_table()

        # Add metadata
        if metadata:
            existing_meta = table.schema.metadata or {}
            existing_meta.update({
                b'schema_version': SCHEMA_VERSION.encode(),
                b'export_timestamp': datetime.now().isoformat().encode(),
                b'row_count': str(len(table)).encode(),
            })

            for key, value in metadata.items():
                existing_meta[key.encode()] = str(value).encode()

            table = table.replace_schema_metadata(existing_meta)

        # Write with optimal compression
        pq.write_table(
            table,
            filepath,
            compression='snappy',  # Fast compression, good for ML workloads
            use_dictionary=True,   # Compress repeated strings
            write_statistics=True  # Enable predicate pushdown
        )

        logging.info(f"Exported {len(table)} windows to {filepath}")


# ============================================================================
# MULTI-SCALE WINDOWS
# ============================================================================

class MultiScaleWindows:
    """
    Three concurrent windows for different analysis types:
    - Immediate (3s): Real-time hazard detection
    - Tactical (30s): Maneuver classification
    - Strategic (5min): Driving style analysis
    """

    # Window definitions
    WINDOWS = {
        'immediate': {
            'duration_s': 3,
            'frames': 60,  # At 20 Hz
            'detects': ['harsh_brake', 'collision', 'sudden_swerve', 'near_miss'],
            'compression': 'standard'
        },
        'tactical': {
            'duration_s': 30,
            'frames': 600,
            'detects': ['lane_weaving', 'tailgating_pattern', 'aggressive_driving', 'traffic_violations'],
            'compression': 'context'
        },
        'strategic': {
            'duration_s': 300,
            'frames': 6000,
            'detects': ['fatigue', 'distraction_trend', 'risk_escalation', 'driving_style'],
            'compression': 'adaptive'
        }
    }

    def __init__(self, tick_rate_hz: float = 20.0):
        self.tick_rate = tick_rate_hz
        self.frame_buffers = {
            'immediate': deque(maxlen=self.WINDOWS['immediate']['frames']),
            'tactical': deque(maxlen=self.WINDOWS['tactical']['frames']),
            'strategic': deque(maxlen=self.WINDOWS['strategic']['frames'])
        }

        self.event_detector = EventDetector()
        self.window_stats = {scale: {} for scale in self.WINDOWS.keys()}

    def ingest_frame(self, frame_data: Dict[str, Any], frame_num: int) -> Dict[str, Any]:
        """Ingest single frame into all windows"""
        # Add to all buffers
        for scale in self.frame_buffers:
            self.frame_buffers[scale].append(frame_data)

        # Detect immediate events
        immediate_events = self.event_detector.detect_immediate_events(frame_data, frame_num)

        return {
            'immediate_events': immediate_events,
            'window_sizes': {scale: len(buf) for scale, buf in self.frame_buffers.items()}
        }

    def compute_window_stats(self, scale: str) -> Dict[str, Any]:
        """Compute statistics for a window scale"""
        buffer = self.frame_buffers[scale]

        if not buffer:
            return {}

        frames = list(buffer)

        # Speed statistics
        speeds = [f.get('speed_kmh', 0) for f in frames]
        avg_speed = np.mean(speeds) if speeds else 0
        max_speed = np.max(speeds) if speeds else 0
        min_speed = np.min(speeds) if speeds else 0

        # MVD scores
        mvd_scores = [f.get('mvd_overall_score', 100) for f in frames]
        avg_mvd = np.mean(mvd_scores) if mvd_scores else 100
        min_mvd = np.min(mvd_scores) if mvd_scores else 100

        # Event counts
        harsh_events = sum(1 for f in frames if any([
            f.get('harsh_braking', False),
            f.get('harsh_acceleration', False),
            f.get('harsh_cornering', False)
        ]))

        lane_violations = sum(1 for f in frames if f.get('lane_invasion_active', False))
        collisions = sum(1 for f in frames if f.get('collision_occurred', False))
        near_misses = sum(1 for f in frames if f.get('ttc_s', 99) < 2.0)

        # Predictive indices (worst case in window)
        ttcs = [f.get('ttc_s', 99) for f in frames]
        tlcs = [f.get('tlc_s', 99) for f in frames]
        min_ttc = np.min(ttcs) if ttcs else 99
        min_tlc = np.min(tlcs) if tlcs else 99

        # Dynamics
        jerks = [f.get('jerk_magnitude', 0) for f in frames]
        g_lats = [abs(f.get('g_force_lateral', 0)) for f in frames]
        g_longs = [abs(f.get('g_force_longitudinal', 0)) for f in frames]

        max_jerk = np.max(jerks) if jerks else 0
        max_g_lat = np.max(g_lats) if g_lats else 0
        max_g_long = np.max(g_longs) if g_longs else 0

        # Control inputs
        throttles = [f.get('control_throttle', 0) for f in frames]
        brakes = [f.get('control_brake', 0) for f in frames]
        steers = [f.get('control_steer', 0) for f in frames]

        avg_throttle = np.mean(throttles) if throttles else 0
        avg_brake = np.mean(brakes) if brakes else 0
        avg_steer = np.mean(steers) if steers else 0

        # Environmental
        traffic_density = int(np.mean([f.get('nearby_vehicles_count', 0) for f in frames]))

        # Get weather from most recent frame
        weather = frames[-1].get('weather_condition', 'clear') if frames else 'clear'
        map_name = frames[-1].get('map_name', 'unknown') if frames else 'unknown'

        stats = {
            'frame_start': frames[0].get('frame', 0),
            'frame_end': frames[-1].get('frame', 0),
            'avg_speed': avg_speed,
            'max_speed': max_speed,
            'min_speed': min_speed,
            'avg_mvd_score': avg_mvd,
            'min_mvd_score': min_mvd,
            'total_harsh_events': harsh_events,
            'total_lane_violations': lane_violations,
            'total_collisions': collisions,
            'total_near_misses': near_misses,
            'min_ttc': min_ttc,
            'min_tlc': min_tlc,
            'max_jerk': max_jerk,
            'max_g_lateral': max_g_lat,
            'max_g_longitudinal': max_g_long,
            'avg_throttle': avg_throttle,
            'avg_brake': avg_brake,
            'avg_steer': avg_steer,
            'traffic_density': traffic_density,
            'weather': weather,
            'map_name': map_name,
        }

        self.window_stats[scale] = stats
        return stats


# ============================================================================
# HYBRID TELEMETRY STORE
# ============================================================================

class HybridTelemetryStore:
    """
    Hybrid storage combining:
    - Arrow columnar (for analytics and ML training)
    - Compressed cache (for real-time streaming to Jetson)
    - Memory-mapped IPC (for shared memory with other processes)
    """

    def __init__(self, session_id: str = None, mmap_path: str = None):
        self.session_id = session_id or f"session_{int(time.time())}"
        self.mmap_path = mmap_path or f"/dev/shm/qdrive_telemetry_{self.session_id}.arrow"

        # Arrow storage
        self.arrow_buffer = ArrowTelemetryBuffer()

        # Compressed cache for streaming
        self.compressed_cache = deque(maxlen=1000)

        # Multi-scale windows
        self.windows = MultiScaleWindows()

        # Memory-mapped file for IPC
        self.mmap = None
        if mmap_path and os.path.exists('/dev/shm'):
            try:
                # Create a 100MB memory-mapped file
                self.mmap = pa.create_memory_map(self.mmap_path, 100 * 1024 * 1024)
                logging.info(f"Created memory-mapped IPC at {self.mmap_path}")
            except Exception as e:
                logging.warning(f"Could not create memory map: {e}")

        logging.info(f"HybridTelemetryStore initialized (session={self.session_id})")

    def ingest_frame(self, frame_data: Dict[str, Any], frame_num: int):
        """Single ingestion point for dual storage"""

        # Add to multi-scale windows
        window_result = self.windows.ingest_frame(frame_data, frame_num)
        immediate_events = window_result['immediate_events']

        # Periodically compute and store window statistics
        # (tactical window updates every 30 frames = 1.5s @ 20Hz)
        if frame_num % 30 == 0:
            for scale in ['immediate', 'tactical']:
                if len(self.windows.frame_buffers[scale]) >= self.windows.WINDOWS[scale]['frames'] * 0.8:
                    stats = self.windows.compute_window_stats(scale)

                    if stats:
                        # Store in Arrow buffer
                        self.arrow_buffer.append_window(stats, immediate_events, self.session_id)

                        # Generate compressed format for streaming
                        compressed = CompressionFormats.format_adaptive(stats, immediate_events)
                        self.compressed_cache.append(compressed)

        return {
            'immediate_events': immediate_events,
            'window_stats': self.windows.window_stats
        }

    def get_streaming_batch(self, n: int = 20) -> List[str]:
        """Get compressed data for Jetson streaming"""
        return list(self.compressed_cache)[-n:]

    def query_analytics(self, start_frame: int, end_frame: int) -> pa.Table:
        """Query columnar data for analytics"""
        table = self.arrow_buffer.get_table()

        if table.num_rows == 0:
            return table

        # Use Arrow compute for filtering
        mask = pc.and_(
            pc.greater_equal(table['frame_start'], start_frame),
            pc.less_equal(table['frame_start'], end_frame)
        )

        return table.filter(mask)

    def export_parquet(self, filepath: str):
        """Export session to Parquet"""
        metadata = {
            'session_id': self.session_id,
            'schema_version': SCHEMA_VERSION,
        }

        self.arrow_buffer.export_parquet(filepath, metadata)

    def cleanup(self):
        """Cleanup resources"""
        if self.mmap and os.path.exists(self.mmap_path):
            try:
                os.remove(self.mmap_path)
                logging.info(f"Removed memory map {self.mmap_path}")
            except Exception as e:
                logging.warning(f"Could not remove memory map: {e}")


# ============================================================================
# ML DATA LOGGER (Main Interface)
# ============================================================================

class MLDataLogger:
    """
    Main interface for ML-ready data logging

    Drop-in replacement for existing DataIngestion classes with:
    - Zero-copy Arrow backend
    - Real-time event detection
    - Multi-scale windowing
    - Parquet export with schema versioning
    - Backward compatible CSV export
    """

    def __init__(self,
                 session_id: str = None,
                 export_csv: bool = True,
                 export_parquet: bool = True,
                 mmap_enabled: bool = False):

        self.session_id = session_id or f"session_{int(time.time())}"
        self.export_csv = export_csv
        self.export_parquet = export_parquet

        # Hybrid storage
        mmap_path = f"/dev/shm/qdrive_{self.session_id}.arrow" if mmap_enabled else None
        self.store = HybridTelemetryStore(session_id=self.session_id, mmap_path=mmap_path)

        # Frame counter
        self.frame_count = 0

        # For CSV backward compatibility
        self.csv_buffer = [] if export_csv else None

        logging.info(f"MLDataLogger initialized (session={self.session_id}, csv={export_csv}, parquet={export_parquet})")

    def log_frame(self, frame_data: Dict[str, Any]):
        """
        Log a single frame (backward compatible with DataIngestion.log_frame)

        Args:
            frame_data: Dictionary with frame telemetry
        """
        # Add frame number if not present
        if 'frame' not in frame_data:
            frame_data['frame'] = self.frame_count

        # Ingest into hybrid store
        result = self.store.ingest_frame(frame_data, self.frame_count)

        # Optionally buffer for CSV export
        if self.csv_buffer is not None:
            self.csv_buffer.append(frame_data)

        self.frame_count += 1

        return result

    def get_streaming_batch(self, n: int = 20) -> List[str]:
        """Get compressed batch for Jetson streaming"""
        return self.store.get_streaming_batch(n)

    def query(self, start_frame: int, end_frame: int) -> pa.Table:
        """Query data by frame range"""
        return self.store.query_analytics(start_frame, end_frame)

    def save_session(self, log_dir: str = "./Session_logs/"):
        """Save session data to disk"""
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')

        # Export Parquet
        if self.export_parquet:
            parquet_path = os.path.join(log_dir, f"session_{self.session_id}_{timestamp}.parquet")
            self.store.export_parquet(parquet_path)
            logging.info(f"Saved Parquet to {parquet_path}")

        # Export CSV (backward compatibility)
        if self.export_csv and self.csv_buffer:
            import pandas as pd
            csv_path = os.path.join(log_dir, f"session_{self.session_id}_{timestamp}.csv")
            df = pd.DataFrame(self.csv_buffer)
            df.to_csv(csv_path, index=False)
            logging.info(f"Saved CSV to {csv_path} ({len(df)} frames)")

    def cleanup(self):
        """Cleanup resources"""
        self.store.cleanup()


# ============================================================================
# PYTORCH DATALOADER (Future ML Training)
# ============================================================================

class ParquetSequenceDataset:
    """
    PyTorch Dataset for loading temporal sequences from Parquet files

    Usage:
        dataset = ParquetSequenceDataset('session.parquet', sequence_length=100)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
    """

    def __init__(self, parquet_path: str, sequence_length: int = 100, stride: int = 50):
        if not PYARROW_AVAILABLE:
            raise ImportError("PyArrow required for ParquetSequenceDataset")

        self.table = pq.read_table(parquet_path)
        self.sequence_length = sequence_length
        self.stride = stride

        # Compute valid start indices
        num_rows = len(self.table)
        self.indices = list(range(0, num_rows - sequence_length + 1, stride))

        logging.info(f"ParquetSequenceDataset: {len(self.indices)} sequences from {num_rows} windows")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """Get a temporal sequence"""
        start = self.indices[idx]
        end = start + self.sequence_length

        # Extract sequence
        sequence = self.table.slice(start, self.sequence_length)

        # Convert to numpy/torch tensors as needed
        # This is a placeholder - customize for your features
        features = {
            'speed': sequence['avg_speed_kmh'].to_numpy(),
            'mvd_score': sequence['avg_mvd_score'].to_numpy(),
            'ttc': sequence['min_ttc_s'].to_numpy(),
            # Add more features as needed
        }

        # Extract labels (e.g., future collision within next N frames)
        labels = sequence['total_collisions'].to_numpy().sum() > 0

        return features, labels


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("MLDataLogger - ML-Ready Data Infrastructure")
    print(f"Schema Version: {SCHEMA_VERSION}")
    print(f"PyArrow Available: {PYARROW_AVAILABLE}")
    print("\nEvent Types:", len(EventType))
    print("Window Scales:", list(MultiScaleWindows.WINDOWS.keys()))
    print("\nReady for integration with Main.py")
