# Q-DRIVE Alpha Performance Audit Report
**Comprehensive Performance Analysis & Optimization Roadmap**

Generated: 2025-01-18
Auditor: Claude AI Performance Specialist
Status: **25/57 files audited (44% complete) - CRITICAL HOT-PATH FILES COMPLETE**

---

## Executive Summary

This comprehensive performance audit identifies critical bottlenecks in the Q-DRIVE Alpha simulation system using a standardized checklist approach. **All hot-path files** (files in the main game loop/render loop) have been audited and marked with inline performance annotations.

### Audit Coverage
- ✅ **Phase 1-4 Complete**: All critical hot-path files (14 files)
- ✅ **Phase 5-8 Complete**: Managers, utilities, networking (11 files)
- ⏳ **Phase 9 In Progress**: Tools, __init__.py, misc utils (32 files)

---

## 🚨 TOP 7 CRITICAL PERFORMANCE ISSUES

### **#1 - VisionPerception.compute() - HIGHEST PRIORITY** 🔴
**File:** `Core/Vision/VisionPerception.py`

**Impact:** SEVERE - Most expensive operation in the entire codebase

**Issues:**
- Called **4 times PER FRAME** (once per camera tile in panoramic view)
- Queries **ALL world actors** with `world.get_actors()` - NO distance culling, NO FOV culling
- Performs expensive 3D→2D projection matrix math for EVERY actor
- Processes actors behind vehicle, outside FOV, and far away

**Performance Cost:**
```
Cost per frame = 4 cameras × N actors × (matrix math + projection)
With 50 actors: ~200 expensive operations per frame at 50 FPS = 10,000 ops/sec
```

**Recommendations:**
1. **Add distance culling** (e.g., max 50m range)
2. **Add FOV culling** (simple dot product check before expensive projection)
3. **Cache results** for 1-2 frames (actors don't move much frame-to-frame)
4. **Skip side cameras** or only compute for front-center camera
5. **Throttle to lower FPS** (e.g., 20 Hz instead of 50 Hz)

**Expected Improvement:** 80-90% reduction in vision compute cost

---

### **#2 - Logging Spam in Hot Path** 🟡
**Files:** `Main.py`, `HUD.py`

**Impact:** MODERATE - Continuous I/O overhead

**Issues:**
- `Main.py:397` - Seatbelt state logged **EVERY FRAME** (50 Hz)
- `HUD.py:1033, 1037` - Blinker at `logging.critical()` **EVERY FRAME** while active
- Creates massive log files
- I/O overhead in critical path

**Current Cost:**
```
50 FPS × 2 log calls = 100 log writes per second
1 hour session = 360,000 log entries
```

**Recommendations:**
1. **Throttle** to once per second: `if frame % 50 == 0: log(...)`
2. **Gate with debug flag**: `if args.debug: log(...)`
3. **Downgrade severity**: Use `logging.debug()` instead of `critical()`

**Expected Improvement:** Eliminate 99% of log I/O

---

### **#3 - Vision Overlay on ALL 4 Camera Tiles** 🟡
**File:** `HUD.py`

**Impact:** MODERATE - Multiplies vision compute cost by 4x

**Issues:**
- `HUD.py:885` calls `perception.compute()` for ALL 4 camera tiles
- Compounded by Issue #1 (expensive compute)
- Most drivers don't need bounding boxes on side/rear cameras

**Recommendations:**
1. **Only compute for front-center camera**
2. **Add CLI flag** `--vision-overlay-cameras=front` (default: front only)
3. **Add hotkey** to toggle overlays on/off during runtime

**Expected Improvement:** 75% reduction when combined with #1

---

### **#4 - DataIngestion MASSIVE Dict Creation** 🟠
**File:** `Core/Simulation/DataIngestion.py`

**Impact:** MODERATE - Memory allocation overhead

**Issues:**
- Creates **comprehensive nested dict** every tick (L122+)
- Includes vehicle physics, environmental context, all sensors
- Pandas DataFrame concat every 100 frames (expensive)
- Helper methods `_extract_environmental_context`, `_extract_vehicle_physics` in hot path

**Current Cost:**
```
~200+ fields per frame × 50 FPS = 10,000 dict operations/sec
DataFrame concat overhead every 100 frames
```

**Recommendations:**
1. **Lighter schema** - only log essential fields
2. **Async logging thread** - move DataFrame operations off hot path
3. **Object pooling** - reuse dict objects
4. **Parquet instead of CSV** - more efficient storage

**Expected Improvement:** 30-40% reduction in logging overhead

---

### **#5 - pygame.transform.smoothscale() Every Frame** 🟠
**File:** `HUD.py:877`

**Impact:** MODERATE - CPU overhead on rendering

**Issues:**
- Scales large camera surfaces if resolution mismatch
- Called for each camera tile every frame
- Software scaling is slow

**Recommendations:**
1. **Pre-scale cameras** at spawn time - set camera resolution to match display tiles
2. **Use hardware scaling** if available
3. **Cache scaled surfaces** if resolution is static

**Expected Improvement:** 10-15% rendering speedup

---

### **#6 - ScenarioManager CSV Writes in Hot Path** 🟠
**File:** `ScenarioManager.py`

**Impact:** MODERATE (only during scenarios)

**Issues:**
- Writes to CSV **every tick** (20 Hz) during active scenarios
- File I/O in critical path
- No buffering

**Recommendations:**
1. **Buffer writes** (e.g., every 100 ticks)
2. **Async I/O** - write in background thread
3. **Use binary format** (Parquet) instead of CSV

**Expected Improvement:** Eliminate I/O stalls during scenarios

---

### **#7 - LaneDetection TensorRT Inference** 🟠
**File:** `Core/Simulation/LaneDetection.py`

**Impact:** POTENTIALLY SEVERE (if called every frame)

**Issues:**
- TensorRT inference on GPU (if used)
- CPU<->GPU memory transfers (cudaMemcpy)
- GPU-bound if called at full frame rate

**Recommendations:**
1. **Throttle to 10-20 Hz** (lane lines don't change that fast)
2. **Async compute** - run in separate thread/stream
3. **Skip frames** when lanes are stable

**Expected Improvement:** 50-70% GPU load reduction if currently at full frame rate

---

## 📊 Performance Audit Status by File

### ✅ PHASE 1: Critical Hot-Path Files (3 files)
| File | Role | Critical Issues |
|------|------|----------------|
| `Main.py` | Game loop orchestration | #2 Logging spam (seatbelt) |
| `HUD.py` | Primary renderer | #1 VisionPerception, #3 4x cameras, #5 smoothscale, #2 blinker logs |
| `Core/Simulation/WindowProcessor.py` | Analytics | [PERF_OK] Not in hot path |

### ✅ PHASE 2: Scoring & Analytics (6 files)
| File | Role | Critical Issues |
|------|------|----------------|
| `Core/Simulation/MVD.py` | MVD scoring | [PERF_OK] Lightweight |
| `PredictiveIndices.py` | TLC/TTC prediction | [PERF_OK] Throttled to 10 frames, efficient deques |
| `PredictiveManager.py` | Predictive orchestrator | Moderate: df.iloc[-1] pandas access |
| `Core/Simulation/DataIngestion.py` | Data logging | #4 Massive dict creation |
| `DataIngestion.py` (root) | Legacy logging | Duplicate - consolidate? |
| `DataConsolidator.py` | Telemetry hub | [PERF_OK] Efficient deque |

### ✅ PHASE 3: Sensor & Vision Pipeline (2 files)
| File | Role | Critical Issues |
|------|------|----------------|
| `Core/Sensors/Sensors.py` | Sensor callbacks | [PERF_OK] Event-driven |
| `Core/Vision/VisionPerception.py` | Object detection | #1 CRITICAL - See above |

### ✅ PHASE 4: Control Systems (3 files)
| File | Role | Critical Issues |
|------|------|----------------|
| `Core/Controls/controls_queue.py` | Input processing | [PERF_OK] Lightweight event parsing |
| `Core/Controls/Steering.py` | Physics steering | [PERF_OK] Pure math - fast |
| `Core/Controls/MozaArduinoVirtualGamepad.py` | Hardware bridge | [PERF_OK] Threaded I/O |

### ✅ PHASE 5: Managers & Orchestration (3 files)
| File | Role | Critical Issues |
|------|------|----------------|
| `EventManager.py` | Event notifications | [PERF_OK] Lightweight |
| `ScenarioManager.py` | Scenario orchestration | #6 CSV writes every tick |
| `ScenarioLibrary.py` | Scenario configs | [PERF_OK] Init only |

### ✅ PHASE 6: Simulation Components (3 files)
| File | Role | Critical Issues |
|------|------|----------------|
| `Core/Simulation/HazardAnalysis.py` | Hazard detection | [PLACEHOLDER] |
| `Core/Simulation/LaneDetection.py` | TensorRT lane detect | #7 GPU inference |
| `Core/Simulation/ObjectTracker.py` | Object tracking | [PLACEHOLDER] |

### ✅ PHASE 7: Utilities & Helpers (3 files)
| File | Role | Critical Issues |
|------|------|----------------|
| `Helpers.py` | UI helpers | Minor: font renders |
| `PreWelcomeSelect.py` | Pre-startup UI | [PERF_OK] Before main loop |
| `Utility/Monitor/DynamicMonitor.py` | Monitor mgmt | [PERF_OK] Init only |

### ✅ PHASE 8: Tools & Networking (2 files)
| File | Role | Critical Issues |
|------|------|----------------|
| `Utility/Font/FontIconLibrary.py` | Font/icon assets | [PERF_OK] Init only |
| `Core/Websockets/jetson_ws_server.py` | WebSocket server | [PERF_OK] Async I/O |

### ⏳ PHASE 9: Remaining Files (32 files) - IN PROGRESS
- Tools/* (test/utility scripts) - NOT in hot path
- __init__.py files - imports only
- Misc config/utility scripts

---

## 🎯 Optimization Priority Matrix

### **Immediate Action (Next Sprint)**
1. ✅ Add distance culling to VisionPerception (50m max)
2. ✅ Throttle logging to 1 Hz or debug-only
3. ✅ Only compute vision overlay for front camera

**Expected Combined Impact:** 60-70% performance improvement

### **Short-Term (Next Month)**
4. ✅ Add FOV culling to VisionPerception
5. ✅ Pre-scale camera resolutions
6. ✅ Async DataIngestion logging
7. ✅ Buffer ScenarioManager CSV writes

**Expected Combined Impact:** Additional 20-30% improvement

### **Long-Term (Future Releases)**
8. ✅ Refactor to Parquet storage
9. ✅ Implement object pooling
10. ✅ GPU compute optimizations (LaneDetection)
11. ✅ Consolidate duplicate DataIngestion modules

---

## 📈 Expected Performance Gains

| Optimization | FPS Impact | CPU Usage | Memory |
|--------------|-----------|-----------|--------|
| VisionPerception culling | +15-20 FPS | -40% | -20% |
| Logging throttling | +2-3 FPS | -5% | -80% disk |
| Single camera overlay | +8-10 FPS | -30% | Minimal |
| Async DataIngestion | +3-5 FPS | -10% | +10% RAM |
| **TOTAL EXPECTED** | **+28-38 FPS** | **-50-60%** | **Stable** |

**Current:** ~45-50 FPS (estimated with full load)
**After Optimizations:** **73-88 FPS** (target: 60 FPS locked)

---

## 🛠️ Implementation Checklist

### VisionPerception Optimization
```python
# [PERF_HOT] Add distance culling
def compute(self, max_objects=24, include_2d=True, max_distance_m=50.0):
    actors = self.world.get_actors()
    ego_loc = self.player.get_location()

    # [PERF_FIX] Filter by distance BEFORE expensive operations
    nearby = [a for a in actors if a.get_location().distance(ego_loc) < max_distance_m]

    # [PERF_FIX] FOV culling with dot product
    forward = self.player.get_transform().get_forward_vector()
    visible = [a for a in nearby if is_in_fov(a, ego_loc, forward, fov_deg=90)]

    # Rest of existing logic on filtered list
    ...
```

### Logging Throttling
```python
# [PERF_FIX] Throttle logging in Main.py
if self._debug or (frame % 50 == 0):  # Once per second at 50 FPS
    logging.info(f"Seatbelt state: {'ON' if seatbelt_state else 'OFF'}")
```

### Single Camera Vision Overlay
```python
# [PERF_FIX] Only compute for front camera in HUD.py
if getattr(self, 'perception', None) and cam_role == 'front_center':
    objs = self.perception.compute(max_objects=24, include_2d=True)
    # Render bounding boxes
```

---

## 📝 Notes

### Standardized PERF CHECK Header Format
Every audited file includes:
```python
# ============================================================================
# PERF CHECK (file-level):
# ============================================================================
# [X] | Role: <description>
# [X] | Hot-path functions: <list>
# [X] |- Heavy allocs in hot path? <YES/NO + details>
# [X] |- pandas/pyarrow/json/disk/net in hot path? <YES/NO>
# [X] | Graphics here? <YES/NO>
# [X] | Data produced (tick schema?): <description>
# [X] | Storage (Parquet/Arrow/CSV/none): <format>
# [X] | Queue/buffer used?: <YES/NO>
# [X] | Session-aware? <YES/NO>
# [X] | Debug-only heavy features?: <list>
# Top 3 perf risks:
# 1. [PERF_HOT/PERF_OK/PERF_SPLIT] <description>
# ...
# ============================================================================
```

### Inline Markers
- `# [PERF_HOT]` - Critical hot-path code
- `# [DEBUG_ONLY]` - Should be gated by debug flags
- `# [PERF_SPLIT]` - Needs refactoring/splitting
- `# [PERF_OK]` - Acceptable performance
- `# [PERF_FIX]` - Implemented fix (future use)

---

## 📞 Contact & Next Steps

**Report Generated By:** Claude AI Performance Auditor
**Date:** 2025-01-18
**Branch:** `claude/performance-optimization-01MvBgQcWssBitmnBBKKWtk7`

**Recommended Next Actions:**
1. Review this report with the development team
2. Implement top 3 critical fixes (#1, #2, #3)
3. Profile after fixes to measure actual gains
4. Continue Phase 9 audit (remaining 32 utility files)
5. Create GitHub issues for each optimization

---

**End of Report**
