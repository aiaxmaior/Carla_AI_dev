"""
Core Controls Package - Q-DRIVE Cortex

Vehicle control and input handling systems.
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

# Import from existing control files with explicit error handling
__all__ = []

try:
    from .controls_queue import DualControl
    __all__.append('DualControl')
except ImportError as e:
    print(f"Warning: Could not import DualControl from controls_queue.py: {e}")

try:
    from .Steering import SteeringModel, VehicleParams  
    __all__.extend(['SteeringModel', 'VehicleParams'])
except ImportError as e:
    print(f"Warning: Could not import from Steering.py: {e}")

try:
    from .dynamic_mapping import DynamicMapping
    __all__.append('DynamicMapping')
except ImportError as e:
    print(f"Warning: Could not import DynamicMapping from dynamic_mapping.py: {e}")

try:
    from .MozaArduinoVirtualGamepad import HardwareVirtualGamepad
    from .MozaVirtualGamepad import MozaVirtualGamepad
    __all__.extend(['HardwareVirtualGamepad', 'MozaVirtualGamepad'])
except ImportError as e:
    print(f"Warning: Could not import from MozaArduinoVirtualGamepad.py or MozaVirtualGamepad.py: {e}")
# Explicitly DO NOT import DynamicMonitor (that belongs in Utility.Monitor)
# If you see this error: "AttributeError: module 'Core.Controls' has no attribute 'DynamicMonitor'"
# It means something is trying to import DynamicMonitor from the wrong place.
# DynamicMonitor should be imported from Utility.Monitor, not Core.Controls.