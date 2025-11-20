"""
Q-DRIVE Cortex Utility Package
============================

Utilities for monitor management, hardware interfacing, and UI components.
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

__version__ = "1.0.0"

# Monitor utilities
try:
    from .Monitor.DynamicMonitor import DynamicMonitor
    from .Monitor.monitor_config import MonitorConfigUtility
    from Font.FontIconLibrary import FontLibrary, IconLibrary

except ImportError:
    pass

__all__ = ['DynamicMonitor', 'MonitorConfigUtility','FontLibrary', 'IconLibrary']
