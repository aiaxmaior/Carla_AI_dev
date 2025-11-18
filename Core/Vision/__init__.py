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

"""
Q-DRIVE Cortex Vision Package
============================

This package contains computer vision and perception systems for Q-DRIVE Cortex.

The Vision package implements advanced perception algorithms that enable the simulator
to understand the driving environment and provide intelligent feedback. It serves as
the "perfect AI" that knows ground truth about the simulation world.

Key Components:

Perception Systems:
- VisionPerception: Main computer vision system using CARLA ground truth
- Object detection and tracking for vehicles, pedestrians, and obstacles
- Real-time distance and trajectory analysis

Environmental Analysis:
- Traffic flow monitoring and analysis
- Lane detection and road geometry understanding
- Hazard identification and risk assessment

Data Processing:
- Multi-threaded image processing pipeline
- Efficient bounding box and overlay rendering
- Performance-optimized computer vision algorithms

Integration:
- Seamless integration with HUD overlay systems  
- Real-time data feeding to scoring and analytics systems
- Ground truth validation for AI training datasets
"""

__version__ = "1.5.0"
__author__ = "Q-DRIVE Cortex Team"

# Import core vision classes
try:
    from .VisionPerception import VisionPerception
    _vision_perception_available = True
except ImportError:
    _vision_perception_available = False
    VisionPerception = None

try:
    from .DMS_Module import AlertLevel, EyeMetrics, DriverState, DMS
    _dms_tracking_available = True
except ImportError:
    _dms_tracking_available = False
    DMS_Module = None
    pass

# Future vision modules
try:
    from .ObjectTracker import ObjectTracker, TrackedObject
    _object_tracking_available = True
except ImportError:
    _object_tracking_available = False
    ObjectTracker = None
    TrackedObject = None

try:
    from .LaneDetection import LaneDetector, LaneGeometry
    _lane_detection_available = True
except ImportError:
    _lane_detection_available = False
    LaneDetector = None
    LaneGeometry = None

try:
    from .HazardAnalysis import HazardDetector, RiskAssessment
    _hazard_analysis_available = True
except ImportError:
    _hazard_analysis_available = False
    pass

__all__ = []

if _vision_perception_available:
    __all__.append('VisionPerception')
if _dms_tracking_available:
    __all__.append(['DMS', 'AlertLevel', 'EyeMetrics', 'DriverState'])
if _object_tracking_available:
    __all__.extend(['ObjectTracker', 'TrackedObject'])
if _lane_detection_available:
    __all__.extend(['LaneDetector', 'LaneGeometry'])
if _hazard_analysis_available:
    __all__.extend(['HazardDetector', 'RiskAssessment'])