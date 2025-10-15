# DMS Module Quick Reference Card

**Module:** `DMS_Module.py` | **Version:** 2025-10-14 | **GPU-Accelerated**

---

## 🚀 Quick Start

```python
from Core.Vision.DMS_Module import DMS, check_gpu_availability

# Check GPU status
check_gpu_availability()

# Initialize DMS
dms = DMS(camera_index=0, use_gpu=True)
dms.start()

# Get latest state
state = dms.get_latest_state()

# Access scores
print(f"Attention: {state.attention_score:.0%}")
print(f"Alert: {state.alert_level.name}")
```

---

## 📊 Output Data Structure

```python
state = dms.get_latest_state()

# Primary Scores (0.0 to 1.0)
state.attention_score      # Higher = better attention
state.drowsiness_score     # Higher = more drowsy
state.distraction_score    # Higher = more distracted

# Alert Level (enum)
state.alert_level          # NORMAL, CAUTION, WARNING, CRITICAL

# Eye Data (per eye)
state.left_eye.center           # (x, y) pixels
state.left_eye.radius           # pixels
state.left_eye.aspect_ratio     # 0.0-1.0+
state.left_eye.is_closed        # bool
state.left_eye.iris_position    # (-1 to 1, -1 to 1)

# Aggregate Metrics
state.eyes_closed_duration_ms   # milliseconds
state.blink_rate               # blinks/minute
state.gaze_vector              # (x, y, z) unit vector
state.head_pose                # (yaw, pitch, roll) degrees

# Event Flags
state.microsleep_detected      # bool - CRITICAL
state.looking_at_phone        # bool
state.head_down_event         # bool

# Metadata
state.timestamp               # Unix timestamp
```

---

## 🎯 Score Interpretation

### Attention Score
| Range | Meaning | Action |
|-------|---------|--------|
| 0.9-1.0 | Excellent attention | None |
| 0.7-0.9 | Good attention | Monitor |
| 0.5-0.7 | Reduced attention | Alert |
| 0.3-0.5 | Poor attention | Warn |
| 0.0-0.3 | Critical inattention | Intervene |

### Drowsiness Score
| Range | Meaning | Indicators |
|-------|---------|------------|
| 0.0-0.2 | Fully alert | Normal blinking |
| 0.2-0.5 | Mild fatigue | Slow blinks, head nods |
| 0.5-0.7 | Significant fatigue | Long closures, poor blink rate |
| 0.7-0.9 | Dangerous drowsiness | Very long closures, nodding |
| 1.0 | Microsleep event | Eyes closed >500ms |

### Distraction Score
| Range | Meaning | Indicators |
|-------|---------|------------|
| 0.0-0.3 | Focused | Gaze forward, head straight |
| 0.3-0.4 | Briefly distracted | Quick glance away |
| 0.4-0.7 | Significantly distracted | Head turned, gaze away |
| 0.7-1.0 | Completely distracted | Multiple indicators |

### Alert Levels
| Level | Score Range | Response |
|-------|-------------|----------|
| NORMAL | < 0.3 | Continue monitoring |
| CAUTION | 0.3-0.5 | Log event, prepare alert |
| WARNING | 0.5-0.7 | Visual/audio alert |
| CRITICAL | ≥ 0.7 | Immediate intervention |

---

## ⚙️ Default Thresholds

```python
# Eye Detection
eye_closed_threshold = 0.2        # Aspect ratio

# Timing
microsleep_ms = 500               # Milliseconds
normal_blink_min = 100            # Milliseconds
normal_blink_max = 400            # Milliseconds

# Blink Rate
normal_blink_rate_min = 10        # Blinks/minute
normal_blink_rate_max = 30        # Blinks/minute

# Head Pose
distraction_angle_deg = 30        # Yaw degrees
head_down_angle_deg = -15         # Pitch degrees

# Gaze
gaze_forward_threshold = 0.8      # Z-component
gaze_down_threshold = -0.3        # Y-component
gaze_side_threshold = 0.3         # X-component

# Alert Levels
alert_caution = 0.3               # Score threshold
alert_warning = 0.5               # Score threshold
alert_critical = 0.7              # Score threshold
```

---

## 🚨 Critical Events

### Microsleep Detection
```python
if state.microsleep_detected:
    # Eyes closed > 500ms
    # drowsiness_score = 1.0 (override)
    # alert_level = CRITICAL
    log_critical_event("MICROSLEEP", state.eyes_closed_duration_ms)
```

### Phone Use
```python
if state.looking_at_phone:
    # Gaze down and to side
    # distraction_score += 0.3
    log_event("PHONE_USE", state.gaze_vector)
```

### Head Down
```python
if state.head_down_event:
    # Head pitch < -15°
    # drowsiness_score += 0.3
    log_event("HEAD_DOWN", state.head_pose)
```

---

## 📈 Scoring Formulas

### Distraction Score
```python
score = 0.0

# Not looking forward
if gaze_vector[2] < 0.8:
    score += 0.3

# Head turned
if abs(head_pose[0]) > 30:
    score += 0.4

# Phone zone
if gaze[1] < -0.3 and abs(gaze[0]) > 0.3:
    score += 0.3

distraction_score = min(score, 1.0)
```

### Drowsiness Score
```python
score = 0.0

# Long closure
if eyes_closed_duration_ms > 200:
    score += min(eyes_closed_duration_ms / 1000, 0.5)

# Microsleep (override)
if eyes_closed_duration_ms > 500:
    score = 1.0

# Abnormal blink rate
if blink_rate < 10 or blink_rate > 30:
    score += 0.2

# Head down
if head_pose[1] < -15:
    score += 0.3

# Repeated microsleeps
if microsleep_count > 2 (in last second):
    score += 0.2

drowsiness_score = min(score, 1.0)
```

### Attention Score
```python
attention_score = 1.0 - max(distraction_score, drowsiness_score)
```

---

## 🔧 Configuration

### Adjust Thresholds
```python
dms = DMS(camera_index=0)

# Modify calibration
dms.calibration['microsleep_ms'] = 600  # More lenient
dms.calibration['distraction_angle_deg'] = 25  # More strict
dms.calibration['eye_closed_threshold'] = 0.18  # More sensitive
```

### GPU Configuration
```python
# Auto-select GPU (prompts user)
dms = DMS(camera_index=0, use_gpu=True)

# Force specific GPU
dms = DMS(camera_index=0, use_gpu=True, gpu_device=1)

# CPU-only mode
dms = DMS(camera_index=0, use_gpu=False)
```

---

## 📝 Logging Example

```python
import json
import logging

# Setup data logger
data_logger = logging.getLogger('dms.telemetry')
handler = logging.FileHandler('dms_data.jsonl')
handler.setFormatter(logging.Formatter('%(message)s'))
data_logger.addHandler(handler)
data_logger.setLevel(logging.INFO)

# Log state
def log_state(state):
    data = {
        "timestamp": state.timestamp,
        "attention": state.attention_score,
        "drowsiness": state.drowsiness_score,
        "distraction": state.distraction_score,
        "alert_level": state.alert_level.name,
        "microsleep": state.microsleep_detected,
        "looking_at_phone": state.looking_at_phone,
        "head_down": state.head_down_event,
        "eyes_closed_ms": state.eyes_closed_duration_ms,
        "blink_rate": state.blink_rate,
        "gaze": state.gaze_vector,
        "head_pose": state.head_pose
    }
    data_logger.info(json.dumps(data))

# In processing loop
while running:
    state = dms.get_latest_state()
    if state:
        log_state(state)
```

---

## 🎬 Common Use Cases

### Simple Monitoring
```python
dms = DMS(camera_index=0)
dms.start()

while True:
    state = dms.get_latest_state()
    if state and state.alert_level.name != "NORMAL":
        print(f"ALERT: {state.alert_level.name}")
        print(f"Attention: {state.attention_score:.0%}")
```

### Event-Based Alerts
```python
prev_level = AlertLevel.NORMAL

while True:
    state = dms.get_latest_state()

    # Alert level changed
    if state.alert_level != prev_level:
        handle_alert_change(prev_level, state.alert_level)
        prev_level = state.alert_level

    # Critical events
    if state.microsleep_detected:
        trigger_immediate_alert()

    if state.looking_at_phone:
        show_phone_warning()
```

### Data Collection
```python
states = []

for i in range(1000):  # Collect 1000 frames
    state = dms.get_latest_state()
    if state:
        states.append(state)

# Analyze
avg_attention = np.mean([s.attention_score for s in states])
microsleep_count = sum(1 for s in states if s.microsleep_detected)
```

---

## 🔍 Debugging

### Check GPU Status
```python
from Core.Vision.DMS_Module import check_gpu_availability
check_gpu_availability()
```

### Enable Debug Visualization
```python
dms = DMS(camera_index=0)
dms.start()

while True:
    frame = dms.get_debug_frame()
    if frame is not None:
        cv2.imshow('DMS Debug', frame)

    if cv2.waitKey(1) == ord('q'):
        break
```

### Check Performance
```python
print(f"FPS: {dms.fps:.1f}")
print(f"Queue size: {dms.state_queue.qsize()}")
print(f"History length: {len(dms._state_history)}")
```

---

## 📚 Additional Documentation

- **Full Analysis:** [DMS_DATA_FLOW_ANALYSIS.md](DMS_DATA_FLOW_ANALYSIS.md)
- **Decision Trees:** [DMS_DECISION_TREE.md](DMS_DECISION_TREE.md)
- **Source Code:** [DMS_Module.py](DMS_Module.py)

---

## 🆘 Troubleshooting

| Issue | Solution |
|-------|----------|
| No GPU detected | Check nvidia-smi, verify CUDA installation |
| Low FPS | Reduce resolution, check GPU usage |
| False positives | Adjust calibration thresholds |
| No face detected | Check lighting, camera position |
| High CPU usage | Enable GPU acceleration |

---

**Quick Reference Version:** 1.0
**Last Updated:** 2025-10-14
