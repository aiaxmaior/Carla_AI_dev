"""
Core Controls Package - Q-DRIVE Cortex

Vehicle control and input handling systems.
"""

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