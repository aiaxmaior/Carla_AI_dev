"""
Q-DRIVE Cortex Core Package
===========================

This package contains the core simulation modules for Q-DRIVE Cortex.

Subpackages:
- Sensors: Sensor management and event detection
- Vision: Computer vision and perception systems
- Controls: Vehicle control and input handling
- Simulation: Physics simulation and world management
"""
# ============================================================================
# PERF CHECK (file-level):
# ============================================================================
# [X] | Role: Package __init__ (imports only, NOT in hot path)
# [ ] | Hot-path functions: None (import-time only)
# [ ] |- Heavy allocs in hot path? N/A - import-time only
# [ ] |- pandas/pyarrow/json/disk/net in hot path? No
# [ ] | Graphics here? No
# [ ] | Data produced (tick schema?): None
# [ ] | Storage (Parquet/Arrow/CSV/none): None
# [ ] | Queue/buffer used?: No
# [ ] | Session-aware? No
# [ ] | Debug-only heavy features?: None
# Top 3 perf risks:
# 1. [PERF_OK] __init__.py - package imports only (import-time)
# 2. [PERF_OK] try/except blocks prevent import failures
# 3. [PERF_OK] No performance concerns
# ============================================================================

__version__ = "1.5.0"
__author__ = "Q-DRIVE Cortex Team"

# Import only specific items from subpackages to avoid conflicts
__all__ = []

try:
    from .Sensors import (
        LaneViolationState,
        LaneChangeState, 
        LaneViolationStateMachine,
        LaneManagement,
        LaneInvasionSensor,
        CollisionSensor,
        GnssSensor
    )
    __all__.extend([
        'LaneViolationState', 'LaneChangeState', 'LaneViolationStateMachine',
        'LaneManagement', 'LaneInvasionSensor', 'CollisionSensor', 'GnssSensor'
    ])
except ImportError:
    pass

try:
    from .Vision import VisionPerception, DMS_Module
    __all__.extend(['VisionPerception', 'DMS_Module'])
except ImportError:
    pass
    
try:
    from .Simulation import MVDFeatureExtractor
    __all__.append('MVDFeatureExtractor')
except ImportError:
    pass
    
try:
    from .Controls import DualControl, SteeringModel, DynamicMapping
    __all__.extend(['DualControl', 'SteeringModel', 'DynamicMapping'])
except ImportError:
    pass
