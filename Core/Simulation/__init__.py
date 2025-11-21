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