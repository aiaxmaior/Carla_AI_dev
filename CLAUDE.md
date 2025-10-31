# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Q-DRIVE Alpha is a CARLA-based driving simulation system for driver safety evaluation and Motor Vehicle Dynamics (MVD) scoring. The system integrates custom hardware controls (Moza steering wheel, Arduino sensors), multi-monitor panoramic displays, computer vision perception, and real-time driver performance analysis.

**Key Components:**
- CARLA simulator integration with custom vehicle physics
- Hardware-in-the-loop control system with virtual gamepad bridge
- Multi-screen rendering (quad-monitor panoramic or single-screen with PIP)
- Real-time MVD scoring system (Collision Avoidance, Lane Management, Harsh Driving)
- Vision perception system for object detection and tracking
- Driver Monitoring System (DMS) integration
- Session data logging and predictive safety indices

## Development Environment

**Conda Environment:** `carla`
```bash
conda activate carla
```

**Main Dependencies:**
- Python 3.x
- CARLA Simulator (requires CARLA_ROOT environment variable)
- pygame, numpy, pandas
- carla Python API
- evdev (for hardware input devices)
- tensorrt, tensorrt_llm (for AI/ML components)

## Common Commands

### Running the Simulation

**Standard launch:**
```bash
./Start.sh
```

**With custom arguments:**
```bash
python Main.py --host localhost --port 2000 --sync --num-vehicles 20 --num-pedestrians 30 --carla-root $CARLA_ROOT
```

**Developer mode (skip selection screens):**
```bash
python Main.py --dev --sync --carla-root $CARLA_ROOT
```

**Single-screen mode:**
```bash
python Main.py --single --sync --carla-root $CARLA_ROOT
```

**Custom map loading:**
```bash
python Main.py --map Town10HD --sync --carla-root $CARLA_ROOT
```

**Load OpenDRIVE map:**
```bash
python Main.py --xodr-path /path/to/map.xodr --sync --carla-root $CARLA_ROOT
```

### Hardware Setup

**Configure input devices (Moza steering wheel, Arduino sensors):**
```bash
python Tools/evdev_autotest.py
```

This creates `configs/joystick_mappings/input_devices.json` with device mappings.

### CARLA Server

**Launch CARLA server separately:**
```bash
./CarlaUE4.sh -carla-rpc-port=2000 -quality-level=Epic -ResX=3840 -ResY=2160 -RenderOffScreen
```

**Launch in windowed mode:**
```bash
./CarlaUE4.sh -carla-rpc-port=2000 -windowed -ResX=1920 -ResY=1080
```

## Architecture

### Core Package Structure (`Core/`)

The Core package contains modular simulation subsystems:

**`Core.Sensors`** - Sensor management and event detection
- `LaneInvasionSensor` - Detects lane violations
- `CollisionSensor` - Tracks collisions with intensity
- `GnssSensor` - GPS positioning
- `LaneManagement` - Lane change state machine

**`Core.Vision`** - Computer vision and perception
- `VisionPerception` (Perception class) - Ground-truth object detection using CARLA world state
- `DMS_Module` - Driver Monitoring System for gaze tracking and distraction detection
- Projects 3D world objects to 2D camera space with bounding boxes

**`Core.Controls`** - Vehicle control and input handling
- `DualControl` (controls_queue.py) - Main controller integrating keyboard, gamepad, and hardware inputs
- `DynamicMapping` (dynamic_mapping.py) - Runtime joystick button/axis mapping
- `MozaArduinoVirtualGamepad` - Hardware bridge creating virtual gamepad from physical devices
- `SteeringModel` (Steering.py) - Custom steering physics and Ackermann geometry

**`Core.Simulation`** - Physics simulation and scoring
- `MVDFeatureExtractor` (MVD.py) - Motor Vehicle Dynamics scoring engine
- `DataIngestion` - Per-frame data collection and CSV export
- `WindowProcessor` - Multi-window rendering coordination
- `LaneDetection`, `ObjectTracker`, `HazardAnalysis` - Perception pipeline modules

**`Core.Websockets`** - Network communication
- `jetson_ws_server.py` - WebSocket server for Jetson Nano integration

### Main Simulation Components

**`Main.py`** - Entry point and main game loop
- Initializes CARLA client connection
- Handles monitor configuration (single vs quad-screen)
- Runs pre-welcome selection, title screen, vehicle/map selection
- Main tick loop with synchronous mode support
- Session management (restart vs exit)
- Cleanup and monitor restoration

**`World.py`** - World state and actor management
- Spawns player vehicle with custom physics
- Manages traffic (vehicles and pedestrians via TrafficManager)
- Creates sensor suite (cameras, collision, lane invasion, GNSS)
- Loads vehicle configurations from `configs/vehicles/*.json`
- Handles weather presets
- Implements single-screen camera system (`_SingleScreenCameras`)

**`HUD.py`** - Heads-up display rendering
- Multi-monitor or single-screen layouts
- Real-time telemetry (speed, RPM, gear, steering angle)
- MVD score display (Collision Avoidance, Lane Management, Harsh Driving)
- Predictive indices (TTC, TLC, brake requirement)
- Event notifications (collisions, lane violations, seatbelt warnings)
- Vision overlay integration

**`TitleScreen.py`** - Vehicle and map selection UI
- Interactive menu using joystick or keyboard
- Vehicle library browsing with preview images
- Map selection with thumbnails
- Persistent key mapping for menu navigation

**`Helpers.py`** - UI utilities
- `EndScreen` - Post-session results and MVD score breakdown
- `PersistentWarningManager` - Manages HUD warnings with timing
- `BlinkingAlert` - Animated alert rendering
- `HelpText` - Keyboard/controller help overlay

### Configuration System

**`configs/vehicles/*.json`** - Vehicle-specific parameters
- Physics overrides (mass, drag, torque curves)
- Steering geometry (max angle, Ackermann radius)
- Engine parameters (max RPM, gear ratios)

**`configs/joystick_mappings/*.json`** - Input device mappings
- Button/axis IDs for steering wheels, pedals, auxiliary controls
- Auto-generated by `Tools/evdev_autotest.py`

**`configs/scenarios/*.json`** - Predefined driving scenarios
- Spawn points, traffic configurations
- Scenario-specific objectives

**`penalty_config/*.json`** - MVD scoring penalties
- Collision severity thresholds
- Lane violation penalties
- Harsh acceleration/braking/cornering thresholds

### Data Flow

1. **Input**: Hardware devices (Moza wheel, Arduino) → `MozaArduinoVirtualGamepad` → virtual gamepad → `DualControl`
2. **Control**: `DualControl` processes inputs → applies to `carla.VehicleControl` → CARLA physics
3. **Perception**: CARLA sensors + `VisionPerception` → object lists with bounding boxes
4. **Scoring**: `MVDFeatureExtractor` evaluates collision, lane, and harsh driving metrics
5. **Logging**: `DataIngestion` captures per-frame telemetry → CSV export
6. **Display**: `HUD` renders telemetry + scores + vision overlay → pygame surface

### Monitor Management

The `Utility.Monitor.DynamicMonitor` class handles multi-monitor setup:
- Detects connected displays via xrandr
- Arranges monitors horizontally for panoramic view
- Restores original layout on exit
- Supports single-screen fallback with PIP rear camera

### Vehicle Physics Customization

Vehicles are customized via `World.load_vehicle_config(vehicle_id)`:
- Loads JSON from `configs/vehicles/{vehicle_id}.json`
- Overrides CARLA blueprint defaults (often inaccurate)
- Applies custom physics control (mass, center of mass, wheels, torque curve)
- Essential for realistic handling with hardware steering wheels

### MVD Scoring System

Three primary components (start at 100, decrease with penalties):

1. **Collision Avoidance**
   - Major collision (>2000 intensity): catastrophic failure
   - Minor collision: moderate penalty
   - Near-miss detection planned

2. **Lane Management**
   - Lane violations without blinkers
   - Lane changes with proper signaling (reduced penalty)
   - Dwelling in violation state increases penalty

3. **Harsh Driving**
   - Excessive longitudinal acceleration/braking
   - Excessive lateral acceleration (cornering)
   - Speed limit violations
   - Seatbelt compliance

Penalties configured in JSON files under `penalty_config/`.

## Development Patterns

### Adding New Sensors

1. Create sensor class in `Core/Sensors/`
2. Implement sensor callback and data storage
3. Instantiate in `World.py` constructor
4. Add to `World.destroy_all_actors()` cleanup

### Extending MVD Scoring

1. Edit `Core/Simulation/MVD.py`
2. Add new score component in `__init__`
3. Implement scoring logic in `update_scores()`
4. Update `get_mvd_datalog_metrics()` for logging
5. Add penalties to JSON config

### Custom Vehicle Configuration

1. Create `configs/vehicles/{vehicle_id}.json`
2. Define carla_blueprint, display_name, image path
3. Set physics overrides (mass, drag, torque, steering)
4. Add entry to `VehicleLibrary.py`
5. Test with `--dev` mode to skip selection screen

### Hardware Integration

1. Identify device with `evdev_probe.py` or `lsusb`
2. Run `Tools/evdev_autotest.py` to capture button/axis mappings
3. Modify `MozaArduinoVirtualGamepad.py` for new device types
4. Test with `Tools/vgamepad_pygame_test.py`

## Key CLI Arguments

- `--sync`: Enable synchronous mode (recommended for deterministic physics)
- `--dev`: Developer mode (skip menus, use default vehicle/map)
- `--single`: Force single-screen layout
- `--screens N`: Number of logical displays
- `--display N`: Display index for main window
- `--carla-root PATH`: CARLA installation directory
- `--no-launch-carla`: Don't auto-launch CARLA server
- `--num-vehicles N`: Traffic vehicle count
- `--num-pedestrians N`: Pedestrian count
- `--map NAME`: Map name (e.g., Town10HD, Town01)
- `--xodr-path FILE`: Load custom OpenDRIVE map
- `--mvd-config FILE`: MVD penalty configuration JSON
- `--quality LEVEL`: CARLA render quality (Low, Medium, High, Epic)
- `--windowed`: Launch CARLA in windowed mode
- `--ResX X --ResY Y`: CARLA render resolution
- `--vision-compare`: Show vision overlay split view
- `--record-vision-demo FILE.mp4`: Record vision demo video

## Important Notes

- **Synchronous mode (`--sync`) is critical** for reproducible physics and timing
- CARLA server takes ~10s to start; connection retries are automatic
- Hardware device mapping is cached in `configs/joystick_mappings/input_devices.json`
- Session logs are saved to `Session_logs/` with timestamps
- Monitor layout is automatically restored on exit (even on crash)
- Vehicle physics must be customized per-vehicle due to CARLA blueprint inaccuracies
- The system expects 4 monitors for quad layout; auto-falls back to single-screen if <4 detected
- Conda environment `carla` must be activated before running

## File Naming Conventions

- `*.py.pre-modular-backup`: Files before Core package migration
- `*.py.backup`: General backup files (safe to ignore)
- `*_backup.py`: Alternative backup naming
- `(1).py` suffix: Duplicate files from migrations

## Testing and Debugging

**Check display configuration:**
```bash
python Tools/check_displays.py
```

**Test evdev devices:**
```bash
python Tools/evdev_probe.py
```

**Test virtual gamepad:**
```bash
python Tools/vgamepad_pygame_test.py
```

**Enable verbose logging:**
```bash
python Main.py -v --debug
```

**CARLA connection issues:**
- Verify CARLA_ROOT is set correctly
- Check CARLA server is running on correct port (default 2000)
- Increase timeout in `Main.py` if server is slow to start

**Hardware not detected:**
- Run `Tools/evdev_autotest.py` with `--verbose`
- Check device permissions (may need udev rules)
- Verify device appears in `/dev/input/`
