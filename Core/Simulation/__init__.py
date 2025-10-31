
"""
Core Simulation Package - Q-DRIVE Cortex

Physics simulation and driver scoring systems.
"""

# Import from existing MVD.py
try:
    from .MVD import MVDFeatureExtractor
except ImportError as e:
    print(f"Warning: Could not import MVDFeatureExtractor: {e}")
    MVDFeatureExtractor = None

try:
    from .DataIngestion import DataIngestion
except ImportError as e:
    print(f"Warning: Could not import DataIngestion: {e}")
    DataIngestion = None  

try:
    from .WindowProcessor import WindowProcessor
except ImportError as e:
    print(f"Warning: Could not import WindowProcessor: {e}")
    WindowProcessor = None


# Public API
__all__ = []

if MVDFeatureExtractor:
    __all__.append('MVDFeatureExtractor')

if DataIngestion:
    __all__.append('DataIngestion')

if WindowProcessor:
    __all__.append('WindowProcessor')