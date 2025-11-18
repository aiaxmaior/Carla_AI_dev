# ============================================================================
# PERF CHECK (file-level):
# ============================================================================
# [X] | Role: Package __init__ (imports, helper functions, NOT in hot path)
# [ ] | Hot-path functions: None (import-time + utility functions)
# [ ] |- Heavy allocs in hot path? N/A - import-time only
# [ ] |- pandas/pyarrow/json/disk/net in hot path? No
# [ ] | Graphics here? No
# [ ] | Data produced (tick schema?): None
# [ ] | Storage (Parquet/Arrow/CSV/none): None
# [ ] | Queue/buffer used?: No
# [ ] | Session-aware? No
# [ ] | Debug-only heavy features?: None
# Top 3 perf risks:
# 1. [PERF_OK] __init__.py - package imports + utility functions
# 2. [PERF_OK] Helper functions (create_sensor_suite, etc.) NOT in hot path
# 3. [PERF_OK] No performance concerns
# ============================================================================

"""
Q-DRIVE Cortex Sensor Package
============================

This package contains intelligent sensor wrappers that transform raw CARLA sensor data 
into classified driving events for the Q-DRIVE Cortex safety evaluation system.

Architecture:
- Raw CARLA sensors provide low-level data (e.g., "line crossed")
- State machines classify events into meaningful categories (e.g., "SEVERE violation")
- Event data flows to MVD.py for scoring and HUD.py for user feedback

Key Components:

State Machines:
- LaneViolationStateMachine: Classifies lane violations by severity
- LaneManagement: Evaluates lane change safety and signaling

Sensor Wrappers:
- LaneInvasionSensor: Integrates lane violation and maneuver evaluation
- CollisionSensor: Detects and classifies vehicle collisions
- GnssSensor: Provides GPS positioning data

Enumerations:
- LaneViolationState: NORMAL, MINOR, MODERATE, SEVERE, CRITICAL
- LaneChangeState: NORMAL, SIGNALLED, UNSIGNALLED, UNSAFE
"""

__version__ = "1.5.0"
__author__ = "Q-DRIVE Cortex Team"

# Import sensor state enumerations
from .Sensors import (
    LaneViolationState,
    LaneChangeState
)

# Import core sensor classes  
from .Sensors import (
    LaneViolationStateMachine,
    LaneManagement,
    LaneInvasionSensor,
    CollisionSensor,
    GnssSensor
)

# Public API exports
__all__ = [
    # Enumerations
    'LaneViolationState',
    'LaneChangeState',
    
    # State Machines
    'LaneViolationStateMachine', 
    'LaneManagement',
    
    # Sensor Wrappers
    'LaneInvasionSensor',
    'CollisionSensor',
    'GnssSensor',
]

# Sensor configuration constants
SENSOR_DEFAULTS = {
    'lane_management': {
        'proximity_threshold': 15.0,  # meters
        'radar_range': 100.0,         # meters  
        'state_timeout': 3.0,         # seconds
    },
    'collision': {
        'cooldown_period': 2.0,       # seconds between collision detections
        'min_impact_threshold': 0.1,  # minimum impulse to register
    },
    'lane_violation': {
        'reset_frame_count': 5,       # frames before auto-reset to NORMAL
        'violation_timeout': 2.0,     # seconds
    }
}

def create_sensor_suite(parent_actor, hud_instance, controller=None, lane_manager=None):
    """
    Convenience factory function to create a complete sensor suite for a vehicle.
    
    Args:
        parent_actor: CARLA vehicle actor to attach sensors to
        hud_instance: HUD instance for event reporting
        controller: DualControl instance for blinker state (optional)
        lane_manager: Custom LaneManagement instance (optional)
        
    Returns:
        dict: Dictionary containing all initialized sensor instances
        
    Example:
        sensors = create_sensor_suite(player_vehicle, hud, controller)
        lane_sensor = sensors['lane_invasion']
        collision_sensor = sensors['collision']
    """
    
    sensors = {}
    
    try:
        # Create lane management if not provided
        if lane_manager is None and controller is not None:
            lane_manager = LaneManagement(parent_actor, hud_instance, controller)
        
        # Lane invasion sensor (integrates with lane management)
        sensors['lane_invasion'] = LaneInvasionSensor(
            parent_actor, 
            hud_instance, 
            lane_manager
        )
        
        # Collision detection
        sensors['collision'] = CollisionSensor(parent_actor, hud_instance)
        
        # GPS/GNSS positioning  
        sensors['gnss'] = GnssSensor(parent_actor, hud_instance)
        
        # Store lane manager reference if created
        if lane_manager:
            sensors['lane_manager'] = lane_manager
            
    except Exception as e:
        import logging
        logging.error(f"Failed to create sensor suite: {e}")
        return {}
    
    return sensors

def destroy_sensor_suite(sensors):
    """
    Safely destroy all sensors in a sensor suite.
    
    Args:
        sensors (dict): Sensor suite dictionary from create_sensor_suite()
    """
    for name, sensor in sensors.items():
        try:
            if hasattr(sensor, 'destroy'):
                sensor.destroy()
        except Exception as e:
            import logging
            logging.warning(f"Failed to destroy sensor '{name}': {e}")

# Sensor state classification helpers
def get_violation_severity_color(state: LaneViolationState) -> tuple:
    """
    Get RGB color tuple for a lane violation severity level.
    
    Args:
        state: LaneViolationState enum value
        
    Returns:
        tuple: (R, G, B) color values (0-255)
    """
    color_map = {
        LaneViolationState.NORMAL: (0, 255, 0),      # Green
        LaneViolationState.MINOR: (255, 255, 0),     # Yellow  
        LaneViolationState.MODERATE: (255, 165, 0),  # Orange
        LaneViolationState.SEVERE: (255, 69, 0),     # Red-Orange
        LaneViolationState.CRITICAL: (255, 0, 0),    # Red
    }
    return color_map.get(state, (128, 128, 128))  # Gray fallback

def get_lane_change_severity_color(state: LaneChangeState) -> tuple:
    """
    Get RGB color tuple for a lane change evaluation.
    
    Args:
        state: LaneChangeState enum value
        
    Returns:
        tuple: (R, G, B) color values (0-255)  
    """
    color_map = {
        LaneChangeState.NORMAL: (0, 255, 0),      # Green
        LaneChangeState.SIGNALLED: (0, 255, 0),   # Green  
        LaneChangeState.UNSIGNALLED: (255, 165, 0), # Orange
        LaneChangeState.UNSAFE: (255, 0, 0),      # Red
    }
    return color_map.get(state, (128, 128, 128))  # Gray fallback

def get_violation_severity_score(state: LaneViolationState) -> float:
    """
    Get numeric penalty score for a lane violation severity.
    
    Args:
        state: LaneViolationState enum value
        
    Returns:
        float: Penalty score (higher = worse)
    """
    score_map = {
        LaneViolationState.NORMAL: 0.0,
        LaneViolationState.MINOR: 1.0,
        LaneViolationState.MODERATE: 3.0,  
        LaneViolationState.SEVERE: 5.0,
        LaneViolationState.CRITICAL: 10.0,
    }
    return score_map.get(state, 0.0)

# Module-level logging setup
def setup_sensor_logging(level='INFO'):
    """
    Configure logging for the sensor package.
    
    Args:
        level (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    """
    import logging
    
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[SENSORS] %(levelname)s: %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

# Package metadata for introspection
SENSOR_TYPES = {
    'lane_invasion': {
        'description': 'Detects and classifies lane line crossings',
        'output_states': ['NORMAL', 'MINOR', 'MODERATE', 'SEVERE', 'CRITICAL'],
        'carla_sensor': 'sensor.other.lane_invasion'
    },
    'collision': {
        'description': 'Detects vehicle collisions and impact intensity',
        'output_data': ['collided', 'actor_type', 'intensity'],
        'carla_sensor': 'sensor.other.collision'  
    },
    'gnss': {
        'description': 'Provides GPS positioning data',
        'output_data': ['latitude', 'longitude'],
        'carla_sensor': 'sensor.other.gnss'
    },
    'lane_management': {
        'description': 'Evaluates lane change safety using radar',
        'output_states': ['NORMAL', 'SIGNALLED', 'UNSIGNALLED', 'UNSAFE'],
        'carla_sensor': 'sensor.other.radar'
    }
}

def get_sensor_info(sensor_type: str = None) -> dict:
    """
    Get information about available sensor types.
    
    Args:
        sensor_type (str, optional): Specific sensor type to query
        
    Returns:
        dict: Sensor information
    """
    if sensor_type:
        return SENSOR_TYPES.get(sensor_type, {})
    return SENSOR_TYPES

# Initialize package-level logger
_logger = setup_sensor_logging()
_logger.info(f"Q-DRIVE Cortex Sensor Package v{__version__} loaded")