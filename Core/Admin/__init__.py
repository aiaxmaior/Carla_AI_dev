"""
Core.Admin - Administrator Panel Module

Provides a GUI panel for simulator administrators to:
- Configure vehicle hyperparameters
- View real-time driver performance metrics
- Set up simulation parameters (optional replacement for PreWelcomeSelect)
"""

from .AdminPanel import (
    AdminPanel,
    AdminPanelMode,
    VehicleHyperparameters,
    DriverPerformanceMetrics,
)

__all__ = [
    "AdminPanel",
    "AdminPanelMode",
    "VehicleHyperparameters",
    "DriverPerformanceMetrics",
]
