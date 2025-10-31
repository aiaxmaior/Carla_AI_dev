# Multi-GPU Camera Rendering Strategy for Q-DRIVE Alpha

**Date:** 2025-10-30
**Status:** Research & Design Phase
**Priority:** High - Performance Critical

---

## Executive Summary

Q-DRIVE Alpha currently runs all camera rendering on a single GPU, creating a bottleneck. This document outlines strategies to distribute camera workload across two GPUs using CARLA's native multi-GPU support.

**Current Bottleneck:**
- 4 panoramic cameras (left_side, left_dash, right_dash, right_side)
- 1 rearview camera
- 4 semantic segmentation cameras (optional, per-camera)
- All rendering on GPU 0
- Future sensor requirements (depth, instance segmentation) will exacerbate the issue

**Goal:** Offload 2+ cameras to GPU 1, improve frame rate, enable additional sensor types.

---

## Current Architecture Analysis

### Camera Creation Pipeline

**Location:** [HUD.py:1022-1120](HUD.py#L1022-L1120) - `CameraManager.__init__()`

**Current Flow:**
1. All cameras spawn via `self.world.spawn_actor()` using the **primary client's world object**
2. Cameras attach to vehicle (`attach_to=self._parent`)
3. Each camera has a callback that feeds into threaded image queues
4. Image processing happens in Python threads, but **GPU rendering** is CARLA-side

**Key Finding:** There is **no explicit GPU assignment** in the current code. All sensors inherit the server's default GPU (GPU 0).

### Current Camera Setup

```python
# From HUD.py CameraManager.__init__
for name, settings in self.config['panoramic_setup'].items():
    # RGB Camera
    cam_bp = bp_library.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(self.single_monitor_dim[0]))
    cam_bp.set_attribute("image_size_y", str(self.single_monitor_dim[1]))
    cam_bp.set_attribute("fov", str(settings['fov']))

    xform = initial_view_set[name]
    rgb = self.world.spawn_actor(cam_bp, xform, attach_to=self._parent)  # ← All use same world
    self.sensors[name] = rgb

    # Semantic Segmentation (optional)
    if settings.get("seg", False):
        seg_bp = bp_library.find("sensor.camera.semantic_segmentation")
        # ... similar spawn on same world
```

**Cameras:**
- `left_side_cam` (queue: 'left_side')
- `left_dash_cam` (queue: 'main')
- `right_dash_cam` (queue: 'right_dash')
- `right_side_cam` (queue: 'right_side')
- `rearview_cam` (separate spawn)
- Optional: 4x semantic segmentation cameras (paired 1:1 with RGB)

---

## CARLA Multi-GPU Options

### Option 1: Multi-Server Architecture (RECOMMENDED)

**How it Works:**
- Primary server runs physics (no GPU): `./CarlaUE4.sh -nullrhi -carla-primary-port=2002`
- Secondary server on GPU 0: `./CarlaUE4.sh -carla-rpc-port=3000 -carla-primary-host=127.0.0.1 -carla-primary-port=2002 -ini:[/Script/Engine.RendererSettings]:r.GraphicsAdapter=0`
- Secondary server on GPU 1: `./CarlaUE4.sh -carla-rpc-port=4000 -carla-primary-host=127.0.0.1 -carla-primary-port=2002 -ini:[/Script/Engine.RendererSettings]:r.GraphicsAdapter=1`

**Sensor Distribution:**
- Primary server **automatically** distributes sensors across connected secondary servers
- Transparent to client - data flows directly from secondary servers to Python client
- **Limitation:** Distribution is automatic/round-robin, not user-controlled per-sensor

**Pros:**
- Official CARLA feature (stable)
- Automatic synchronization (forces sync mode)
- Transparent data flow to client
- Primary server offloads all rendering

**Cons:**
- Cannot explicitly assign specific sensors to specific GPUs
- Need to start 3 CARLA processes (1 primary + 2 secondary)
- Round-robin distribution may not be optimal (e.g., might split heavy cameras unevenly)

### Option 2: Multi-Client Architecture

**How it Works:**
- Single CARLA server on GPU 0
- Client 1 connects and creates sensors 1-3 (GPU 0)
- Client 2 connects **with GPU affinity** and creates sensors 4-6 (GPU 1)

**Pros:**
- Explicit control over which sensors go to which GPU
- Only 1 CARLA server process
- Can assign "heavy" sensors (segmentation, depth) to dedicated GPU

**Cons:**
- **NOT officially supported** for GPU distribution
- CARLA server still renders everything on its assigned GPU(s) regardless of which client created the sensor
- Clients control creation, not rendering hardware assignment
- **This won't work** - sensors render on server-side GPU pool

### Option 3: Hybrid - Server-Side Scripts with Multi-GPU

**How it Works:**
- Use CARLA's multi-server setup
- Client connects to primary server
- Use CARLA Python API's server-side scripting to control which secondary server gets which sensor
- Requires CARLA source modifications or plugin development

**Status:** Experimental, would require deep CARLA customization.

---

## Recommended Implementation Strategy

### Phase 1: Multi-Server Setup (Immediate)

**Action:** Deploy CARLA multi-GPU server architecture

**Implementation Steps:**

1. **Server Launch Script (`scripts/launch_carla_multigpu.sh`):**
   ```bash
   #!/bin/bash
   # Launch primary server (no rendering)
   $CARLA_ROOT/CarlaUE4.sh -nullrhi -carla-primary-port=2002 &
   PRIMARY_PID=$!

   sleep 5  # Wait for primary to initialize

   # Launch secondary on GPU 0
   CUDA_VISIBLE_DEVICES=0 $CARLA_ROOT/CarlaUE4.sh \
     -carla-rpc-port=3000 \
     -carla-primary-host=127.0.0.1 \
     -carla-primary-port=2002 \
     -ini:[/Script/Engine.RendererSettings]:r.GraphicsAdapter=0 &
   GPU0_PID=$!

   sleep 5

   # Launch secondary on GPU 1
   CUDA_VISIBLE_DEVICES=1 $CARLA_ROOT/CarlaUE4.sh \
     -carla-rpc-port=4000 \
     -carla-primary-host=127.0.0.1 \
     -carla-primary-port=2002 \
     -ini:[/Script/Engine.RendererSettings]:r.GraphicsAdapter=1 &
   GPU1_PID=$!

   echo "Primary PID: $PRIMARY_PID"
   echo "GPU0 Secondary PID: $GPU0_PID"
   echo "GPU1 Secondary PID: $GPU1_PID"

   wait
   ```

2. **Client Connection Update (`Main.py`):**
   ```python
   # Connect to PRIMARY server (not secondary)
   client = carla.Client(args.host, 2002)  # Primary port
   client.set_timeout(300.0)

   # Sensors will be automatically distributed across GPU 0 and GPU 1 secondaries
   ```

3. **No Code Changes Required in HUD.py:**
   - Sensor creation code remains unchanged
   - Primary server handles distribution automatically
   - Data flows transparently to client

4. **Verification:**
   - Monitor GPU usage: `nvidia-smi dmon -i 0,1 -s u`
   - Both GPUs should show rendering activity
   - Expected distribution: ~2-3 cameras per GPU

**Expected Distribution (Automatic Round-Robin):**
- GPU 0: `left_side_cam`, `right_dash_cam`, `rearview_cam` (+ their seg cameras)
- GPU 1: `left_dash_cam`, `right_side_cam` (+ their seg cameras)

**Note:** Exact distribution is CARLA-controlled and may vary. Monitor with `nvidia-smi`.

---

### Phase 2: Optimization & Sensor Expansion (Future)

Once multi-GPU is confirmed working:

1. **Add Depth Cameras** (for advanced perception):
   - Spawn on existing world instance
   - CARLA will distribute across GPUs automatically
   - Example: depth sensor for left_dash_cam

2. **Add Instance Segmentation** (for object tracking):
   - High GPU cost - benefits most from multi-GPU
   - Consider limiting to 1-2 critical cameras initially

3. **Benchmark Different Configurations:**
   - Test with/without semantic segmentation
   - Measure FPS impact of additional sensors
   - Adjust `sensor_tick` rates if needed (currently 0.066s for seg)

4. **Consider Manual Distribution (Advanced):**
   - If automatic distribution is suboptimal, explore CARLA source modifications
   - Potential PR to CARLA: sensor-level GPU hints via blueprints
   - Alternative: Spawn sensors in batches with deliberate delays to influence round-robin

---

## Code Changes Required

### Minimal Changes (Phase 1)

**File:** `Main.py` (or wherever CARLA server is launched)

**Change:** Update `--carla-launch` behavior to use multi-GPU script instead of single server.

**Before:**
```python
# Launch single CARLA server
subprocess.Popen([
    f"{carla_root}/CarlaUE4.sh",
    "-carla-rpc-port=2000",
    ...
])
```

**After:**
```python
# Launch multi-GPU CARLA setup
subprocess.Popen([
    "./scripts/launch_carla_multigpu.sh"
])
```

**File:** Connection logic (ensure port 2002 for primary)

```python
client = carla.Client(args.host, 2002)  # Primary server port
```

### No Changes Needed

- `HUD.py` - Sensor spawning stays the same
- `World.py` - Physics/vehicle code unchanged
- `DynamicMonitor.py` - Display config unaffected
- All sensor callbacks and data pipelines remain identical

---

## Testing Plan

### Test 1: Baseline (Single GPU)
1. Run current setup: `./Start.sh`
2. Measure FPS: `nvidia-smi dmon -i 0 -s u`
3. Record: GPU utilization, frame time, CPU usage

### Test 2: Multi-GPU Deployment
1. Launch: `./scripts/launch_carla_multigpu.sh`
2. Wait for all 3 servers to initialize (~30s)
3. Run simulation: `python Main.py --host localhost --port 2002 --sync`
4. Monitor both GPUs: `nvidia-smi dmon -i 0,1 -s u -d 1`
5. Verify both GPUs show activity (>10% utilization)

### Test 3: Sensor Load Comparison
| Configuration | GPU 0 Load | GPU 1 Load | FPS | Notes |
|---------------|------------|------------|-----|-------|
| Single GPU (baseline) | ~80-95% | 0% | ? | Current |
| Multi-GPU (4 RGB) | ~40-50% | ~40-50% | ? | Expected improvement |
| Multi-GPU (4 RGB + 4 SEG) | ~60-70% | ~60-70% | ? | With segmentation |
| Multi-GPU (4 RGB + 4 SEG + 2 DEPTH) | ~70-80% | ~70-80% | ? | Future expansion |

### Success Criteria
- Both GPUs utilized during rendering (>10% each)
- FPS improvement of 15-30% vs single GPU
- No visual artifacts or frame drops
- Synchronization maintained (tick alignment)

---

## Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Automatic distribution suboptimal | Medium | Monitor and benchmark; consider manual workarounds if severe |
| Increased VRAM usage (3 processes) | Low | Monitor VRAM; primary server uses minimal VRAM with `-nullrhi` |
| Synchronization issues | High | CARLA auto-enables sync mode; test thoroughly with traffic |
| Startup complexity | Medium | Wrap in robust script with health checks and PID management |
| Port conflicts | Low | Use unique ports; add conflict detection to launch script |

---

## Alternative Approaches (Considered & Rejected)

### 1. CUDA Stream Partitioning in Python
**Idea:** Use CUDA streams to split rendering workload in client-side PyGame.
**Why Rejected:** PyGame/NumPy rendering is CPU-bound; GPU work happens in CARLA server.

### 2. Multi-Process Python Client
**Idea:** Run 2 Python client processes, each connecting to separate CARLA servers.
**Why Rejected:** Physics synchronization nightmare; would need 2 separate simulations.

### 3. Per-Camera GPU Hints in Blueprint
**Idea:** Add GPU affinity hints to camera blueprints.
**Why Rejected:** Not supported by CARLA; would require engine modifications.

---

## Next Steps

1. **Create launch script** (`scripts/launch_carla_multigpu.sh`)
2. **Update Start.sh** to detect and use multi-GPU mode when 2+ GPUs available
3. **Test with minimal setup** (just 4 RGB cameras, no segmentation)
4. **Benchmark and document** results
5. **If successful:** Enable semantic segmentation and re-test
6. **If successful:** Design depth/instance camera additions

---

## Questions for Further Investigation

1. **Can we influence round-robin?**
   - Test: Spawn cameras in specific order to bias distribution
   - Test: Delay between spawns to force different servers

2. **Segmentation camera impact?**
   - Do seg cameras count toward distribution separately?
   - Or are they treated as paired with RGB?

3. **Dynamic re-distribution?**
   - Can sensors be destroyed and re-spawned mid-session?
   - Would CARLA re-balance across GPUs?

4. **VRAM limits?**
   - Each secondary server loads full world geometry
   - Test with large maps (Town10HD) to verify VRAM headroom

---

## References

- [CARLA Multi-GPU Documentation](https://carla.readthedocs.io/en/latest/adv_multigpu/)
- [CARLA Synchrony & Time-step](https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/)
- [GitHub Issue #3772: Multiple GPUs for Multiple Camera Sensors](https://github.com/carla-simulator/carla/issues/3772)

---

## Appendix A: GPU Monitoring Commands

**Real-time monitoring:**
```bash
# Both GPUs, 1-second updates
nvidia-smi dmon -i 0,1 -s ucm -d 1

# Detailed per-process view
nvidia-smi pmon -i 0,1 -d 1
```

**One-time snapshot:**
```bash
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv
```

**Log to file:**
```bash
nvidia-smi dmon -i 0,1 -s ucm -o T -f gpu_log.csv
```

---

## Appendix B: Launch Script Template

See `scripts/launch_carla_multigpu.sh` (to be created).

Key features:
- Health checks for each server
- PID file management
- Graceful shutdown handler
- Port conflict detection
- GPU availability verification

---

**End of Document**
