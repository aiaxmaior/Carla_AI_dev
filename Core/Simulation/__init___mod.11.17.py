
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

# Public API
__all__ = []

if MVDFeatureExtractor:
    __all__.append('MVDFeatureExtractor')