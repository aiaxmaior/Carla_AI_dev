"""
Core.Data - ML-Ready Data Logging and Event Detection

This package provides high-performance data logging optimized for:
- Transfer learning pipelines
- Event detection and classification
- Reinforcement learning
- Real-time streaming to edge devices

Author: Q-DRIVE Team
Performance Audit: 2025-11-18
"""

# ============================================================================
# PERF CHECK (file-level):
# ============================================================================
# [X] | Role: Package __init__ for ML data infrastructure
# [ ] | Hot-path functions: None (import-time only)
# [ ] |- Heavy allocs in hot path? N/A
# [ ] |- pandas/pyarrow/json/disk/net in hot path? No
# [ ] | Graphics here? No
# [ ] | Data produced (tick schema?): None
# [ ] | Storage (Parquet/Arrow/CSV/none): None
# ============================================================================

from .MLDataLogger import (
    MLDataLogger,
    ArrowTelemetryBuffer,
    HybridTelemetryStore,
    MultiScaleWindows,
    EventDetector,
    CompressionFormats
)

__all__ = [
    'MLDataLogger',
    'ArrowTelemetryBuffer',
    'HybridTelemetryStore',
    'MultiScaleWindows',
    'EventDetector',
    'CompressionFormats'
]
