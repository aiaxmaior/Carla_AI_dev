"""
Q-DRIVE Cortex Vision Package
=============================
(…original docstring kept…)
"""

from __future__ import annotations

__version__ = "1.5.0"
__author__ = "Q-DRIVE Cortex Team"

# -----------------------------
# Availability flags (defaults)
# -----------------------------
_vision_perception_available = False
_dms_tracking_available = False
_object_tracking_available = False
_lane_detection_available = False
_hazard_analysis_available = False

# -----------------------------
# Core vision classes
# -----------------------------
# NOTE: VisionPerception.py defines class `Perception` (HUD imports this).
try:
    from .VisionPerception import Perception
    _vision_perception_available = True
except ImportError:
    Perception = None

# Optional: DMS module
try:
    from .DMS_Module import AlertLevel, EyeMetrics, DriverState, DMS
    _dms_tracking_available = True
except ImportError:
    AlertLevel = EyeMetrics = DriverState = DMS = None
    _dms_tracking_available = False

# Optional: Object tracking
try:
    from .ObjectTracker import ObjectTracker, TrackedObject
    _object_tracking_available = True
except ImportError:
    ObjectTracker = TrackedObject = None
    _object_tracking_available = False

# Optional: Lane detection
try:
    from .LaneDetection import LaneDetector, LaneGeometry
    _lane_detection_available = True
except ImportError:
    LaneDetector = LaneGeometry = None
    _lane_detection_available = False

# Optional: Hazard analysis
try:
    from .HazardAnalysis import HazardDetector, RiskAssessment
    _hazard_analysis_available = True
except ImportError:
    HazardDetector = RiskAssessment = None
    _hazard_analysis_available = False

# -----------------------------
# Public API
# -----------------------------
__all__ = []
if _vision_perception_available:
    __all__.append("Perception")
if _dms_tracking_available:
    __all__.extend(["DMS", "AlertLevel", "EyeMetrics", "DriverState"])
if _object_tracking_available:
    __all__.extend(["ObjectTracker", "TrackedObject"])
if _lane_detection_available:
    __all__.extend(["LaneDetector", "LaneGeometry"])
if _hazard_analysis_available:
    __all__.extend(["HazardDetector", "RiskAssessment"])

# Optional: quick availability map for debugging
availability = {
    "Perception": _vision_perception_available,
    "DMS": _dms_tracking_available,
    "ObjectTracker": _object_tracking_available,
    "LaneDetection": _lane_detection_available,
    "HazardAnalysis": _hazard_analysis_available,
}