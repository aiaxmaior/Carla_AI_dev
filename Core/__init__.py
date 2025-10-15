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
    __all__.append('VisionPerception','DMS_Module')
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
