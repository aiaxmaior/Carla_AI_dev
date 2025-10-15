# DMS Decision Tree & Logic Flow

## Visual Decision Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CAMERA FRAME INPUT                          │
│                         (640x480 @ 30fps)                          │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    MEDIAPIPE FACE DETECTION                         │
│                         Face Mesh (468 landmarks)                   │
└────────────┬────────────────────────────────────┬───────────────────┘
             │                                    │
             ▼                                    ▼
    ┌────────────────┐                  ┌────────────────┐
    │  LEFT EYE      │                  │  RIGHT EYE     │
    │  Processing    │                  │  Processing    │
    └───────┬────────┘                  └───────┬────────┘
            │                                   │
            │  ┌───────────────────────────────┘
            │  │
            ▼  ▼
    ┌──────────────────────────────────────────────────┐
    │           EYE METRICS EXTRACTION                 │
    │  • Iris center & radius                          │
    │  • Eye aspect ratio (height/width)               │
    │  • Iris position (normalized -1 to 1)            │
    └────────────────────┬─────────────────────────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │   IS EYE CLOSED?       │
            │  aspect_ratio < 0.2    │
            └──────┬─────────┬───────┘
                   │         │
              YES  │         │  NO
                   │         │
                   ▼         ▼
         ┌─────────────┐   ┌─────────────┐
         │ is_closed = │   │ is_closed = │
         │    TRUE     │   │    FALSE    │
         └──────┬──────┘   └──────┬──────┘
                │                 │
                └────────┬────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      BLINK & CLOSURE DETECTION                      │
└─────────────────────────────────────────────────────────────────────┘
                         │
                    Both Eyes Closed?
                         │
            ┌────────────┼────────────┐
            │ YES                     │ NO
            ▼                         ▼
    Start/Continue Timer      Eyes Reopened After Closure?
    eyes_closed_duration_ms          │
            │                    ┌───┴───┐
            │               YES  │       │  NO
            │                    ▼       ▼
            │              Record Blink  Reset Timer
            │              (if 100-400ms)
            │
            ▼
    Closure Duration > 500ms?
            │
        ┌───┴───┐
    YES │       │  NO
        ▼       ▼
    MICROSLEEP  Continue
    FLAG SET    Monitoring
        │
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     HEAD POSE ESTIMATION                            │
│  • Yaw (left/right turn): -30° to +30°                             │
│  • Pitch (up/down tilt): -20° to +20°                              │
│  • Roll (side tilt): calculated from landmarks                      │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      GAZE VECTOR ESTIMATION                         │
│  Combines: Iris position + Head pose                                │
│  Output: 3D unit vector (x, y, z)                                  │
│    Forward = (0, 0, 1)                                             │
│    Down = (0, -1, 0)                                               │
│    Right = (1, 0, 0)                                               │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        SCORE CALCULATION                            │
└─────────────────────────────────────────────────────────────────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
                    ▼                 ▼
        ┌───────────────────┐  ┌───────────────────┐
        │  DISTRACTION      │  │  DROWSINESS       │
        │  SCORE            │  │  SCORE            │
        └───────┬───────────┘  └───────┬───────────┘
                │                      │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │   ATTENTION SCORE    │
                │  = 1.0 - max(D, D)   │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │  ALERT LEVEL         │
                │  DETERMINATION       │
                └──────────────────────┘
```

---

## Distraction Score Decision Tree

```
DISTRACTION SCORE = 0.0
        │
        ├─→ [CHECK 1] Gaze Forward?
        │   gaze_vector[2] < 0.8?
        │       ├─→ YES: score += 0.3
        │       └─→ NO:  continue
        │
        ├─→ [CHECK 2] Head Turned?
        │   |head_pose[0]| > 30°?
        │       ├─→ YES: score += 0.4
        │       └─→ NO:  continue
        │
        └─→ [CHECK 3] Looking at Phone Zone?
            (gaze[1] < -0.3) AND (|gaze[0]| > 0.3)?
                ├─→ YES: score += 0.3
                │        looking_at_phone = TRUE
                └─→ NO:  continue

FINAL: distraction_score = min(score, 1.0)

┌─────────────────────────────────────────────┐
│ Distraction Score Interpretation            │
├─────────────────────────────────────────────┤
│ 0.0 - 0.3  │ Focused on road (NORMAL)      │
│ 0.3 - 0.4  │ Briefly distracted (CAUTION)  │
│ 0.4 - 0.7  │ Significantly off-task (WARN) │
│ 0.7 - 1.0  │ Completely distracted (CRIT)  │
└─────────────────────────────────────────────┘

MAXIMUM POSSIBLE: 1.0 (all three conditions met)
```

---

## Drowsiness Score Decision Tree

```
DROWSINESS SCORE = 0.0
        │
        ├─→ [CHECK 1] Eyes Closed Duration?
        │   eyes_closed_duration_ms > 200?
        │       ├─→ YES: score += min(duration/1000, 0.5)
        │       └─→ NO:  continue
        │
        ├─→ [CHECK 2] Microsleep Detected?
        │   eyes_closed_duration_ms > 500?
        │       ├─→ YES: score = 1.0 ← OVERRIDE ALL
        │       │        microsleep_detected = TRUE
        │       │        CRITICAL ALERT
        │       └─→ NO:  continue
        │
        ├─→ [CHECK 3] Abnormal Blink Rate?
        │   (blink_rate < 10) OR (blink_rate > 30)?
        │       ├─→ YES: score += 0.2
        │       └─→ NO:  continue
        │
        ├─→ [CHECK 4] Head Nodding/Down?
        │   head_pose[1] < -15°?
        │       ├─→ YES: score += 0.3
        │       │        head_down_event = TRUE
        │       └─→ NO:  continue
        │
        └─→ [CHECK 5] Repeated Microsleeps? (Trend Analysis)
            microsleep_count_last_30_frames > 2?
                ├─→ YES: score += 0.2
                └─→ NO:  continue

FINAL: drowsiness_score = min(score, 1.0)

┌─────────────────────────────────────────────┐
│ Drowsiness Score Interpretation             │
├─────────────────────────────────────────────┤
│ 0.0 - 0.2  │ Fully alert (NORMAL)          │
│ 0.2 - 0.5  │ Mild fatigue (CAUTION)        │
│ 0.5 - 0.7  │ Significant fatigue (WARNING) │
│ 0.7 - 1.0  │ Dangerous drowsiness (CRIT)   │
│ 1.0 EXACT  │ MICROSLEEP EVENT (CRITICAL)   │
└─────────────────────────────────────────────┘

MAXIMUM POSSIBLE: 1.0 (microsleep override)
```

---

## Alert Level Determination

```
INPUT: max_score = max(distraction_score, drowsiness_score)

                    ┌─────────────────┐
                    │   max_score     │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
         < 0.3          < 0.5          < 0.7        ≥ 0.7
              │              │              │              │
              ▼              ▼              ▼              ▼
        ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
        │ NORMAL  │    │ CAUTION │    │ WARNING │    │CRITICAL │
        └─────────┘    └─────────┘    └─────────┘    └─────────┘
              │              │              │              │
              ▼              ▼              ▼              ▼
        No action      Monitor         Alert         Immediate
        required       driver          driver        intervention

        Color:         Color:         Color:         Color:
        GREEN          YELLOW         ORANGE         RED

┌────────────────────────────────────────────────────────────────────┐
│                     Alert Level Details                            │
├────────────────────────────────────────────────────────────────────┤
│ NORMAL (0.0-0.3)                                                   │
│   • Driver is attentive and alert                                 │
│   • No intervention needed                                         │
│   • Continue monitoring                                            │
│                                                                    │
│ CAUTION (0.3-0.5)                                                 │
│   • Early signs of distraction or fatigue                         │
│   • Log event for analysis                                        │
│   • Prepare for potential alert                                   │
│                                                                    │
│ WARNING (0.5-0.7)                                                 │
│   • Significant distraction or drowsiness detected                │
│   • Visual/audio alert to driver                                 │
│   • Increase monitoring frequency                                 │
│   • Log detailed state                                            │
│                                                                    │
│ CRITICAL (0.7-1.0)                                                │
│   • Dangerous level of inattention                                │
│   • Immediate alerts (visual, audio, haptic)                      │
│   • Consider autonomous intervention                              │
│   • Full telemetry logging                                        │
│   • May trigger emergency protocols                              │
└────────────────────────────────────────────────────────────────────┘
```

---

## Event Flag Logic

### Microsleep Detection

```
┌──────────────────────────────────────┐
│  Both eyes closed continuously?      │
└─────────────┬────────────────────────┘
              │
              ▼
     Start timer: eye_closed_start
              │
              ▼
     Track duration: eyes_closed_duration_ms
              │
              ▼
     Duration > 500ms?
              │
         ┌────┴────┐
     YES │         │ NO
         ▼         ▼
    ┌─────────┐  Continue
    │MICROSLEEP│  monitoring
    │ FLAG=TRUE│
    └────┬────┘
         │
         ▼
    drowsiness_score = 1.0 (override)
    alert_level = CRITICAL
    LOG EVENT with duration
```

### Phone Use Detection

```
┌──────────────────────────────────────┐
│  Check Gaze Vector Position          │
└─────────────┬────────────────────────┘
              │
              ▼
     gaze_vector[1] < -0.3?  (Looking down?)
              │
         ┌────┴────┐
     YES │         │ NO
         ▼         └──→ looking_at_phone = FALSE
    |gaze[0]| > 0.3?  (Looking to side?)
         │
    ┌────┴────┐
YES │         │ NO
    ▼         └──→ looking_at_phone = FALSE
┌─────────────┐
│PHONE FLAG   │
│  = TRUE     │
└──────┬──────┘
       │
       ▼
  distraction_score += 0.3
  LOG EVENT: "Phone use detected"
```

### Head Down Event

```
┌──────────────────────────────────────┐
│  Check Head Pitch Angle              │
└─────────────┬────────────────────────┘
              │
              ▼
     head_pose[1] < -15°?  (Tilted down?)
              │
         ┌────┴────┐
     YES │         │ NO
         ▼         └──→ head_down_event = FALSE
    ┌─────────────┐
    │HEAD DOWN    │
    │ FLAG = TRUE │
    └──────┬──────┘
           │
           ▼
    drowsiness_score += 0.3
    LOG EVENT: "Head nodding detected"
    Possible causes:
      • Nodding off (drowsiness)
      • Looking at phone/lap
      • Adjusting controls
```

---

## Blink Classification

```
Eyes Close
    │
    ▼
Start Timer
    │
    ▼
Eyes Reopen
    │
    ▼
Calculate Duration
    │
    ├──→ Duration < 100ms
    │    └──→ TOO FAST (ignore, likely noise)
    │
    ├──→ Duration 100-400ms
    │    └──→ NORMAL BLINK ✓
    │         • Record in blink history
    │         • Update blink rate
    │
    ├──→ Duration 400-500ms
    │    └──→ LONG BLINK ⚠
    │         • May indicate fatigue
    │         • Contributes to drowsiness score
    │
    └──→ Duration > 500ms
         └──→ MICROSLEEP 🚨
              • Set microsleep flag
              • Override drowsiness = 1.0
              • CRITICAL alert
```

---

## Trend Analysis (Aggregate Metrics)

```
┌────────────────────────────────────────────────────────────┐
│  STATE HISTORY BUFFER: 150 frames (5 seconds @ 30fps)     │
└───────────────────────────┬────────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │ Analyze Last 30 Frames  │
              │ (1 second window)       │
              └────────────┬────────────┘
                           │
                           ▼
              Count microsleep events in window
                           │
                           ▼
              microsleep_count > 2?
                           │
                      ┌────┴────┐
                  YES │         │ NO
                      ▼         └──→ No trend adjustment
          ┌────────────────────┐
          │ PATTERN DETECTED!  │
          │ Multiple microsleep│
          │ in short time      │
          └─────────┬──────────┘
                    │
                    ▼
          drowsiness_score += 0.2
          LOG: "Repeated microsleep pattern"
          CRITICAL: Severe fatigue detected
```

---

## Score Combination Logic

```
┌───────────────┐        ┌───────────────┐
│  DISTRACTION  │        │  DROWSINESS   │
│     SCORE     │        │     SCORE     │
│   (0.0-1.0)   │        │   (0.0-1.0)   │
└───────┬───────┘        └───────┬───────┘
        │                        │
        └────────┬───────────────┘
                 │
                 ▼
        ┌────────────────┐
        │   MAX( D1, D2 )│
        └────────┬───────┘
                 │
                 ▼
        ┌────────────────┐
        │ ALERT LEVEL    │
        │ (4 thresholds) │
        └────────┬───────┘
                 │
          ┌──────┴──────┐
          │             │
          ▼             ▼
    ┌─────────┐   ┌─────────┐
    │ATTENTION│   │  ALERT  │
    │  SCORE  │   │  LEVEL  │
    │ = 1-max │   │ N/C/W/C │
    └─────────┘   └─────────┘

WHY MAX?
  • Takes the worse of the two conditions
  • Driver cannot be both drowsy AND distracted safely
  • Conservative approach for safety
  • Single unified alert state

ATTENTION SCORE FORMULA:
  attention_score = 1.0 - max(distraction_score, drowsiness_score)

  Examples:
    • Distraction=0.2, Drowsy=0.1 → Attention=0.8 (Good)
    • Distraction=0.5, Drowsy=0.3 → Attention=0.5 (Reduced)
    • Distraction=0.2, Drowsy=0.8 → Attention=0.2 (Poor)
```

---

## Summary of All Decisions

| Decision Point | Location | Input | Output | Threshold |
|----------------|----------|-------|--------|-----------|
| Eye Closed? | Line 434 | aspect_ratio | is_closed (bool) | < 0.2 |
| Valid Blink? | Line 500-505 | closure duration | record blink | 100-400ms |
| Microsleep? | Line 496 | closure duration | flag + score=1.0 | > 500ms |
| Not Forward? | Line 521 | gaze.z | distraction +0.3 | < 0.8 |
| Head Turned? | Line 526 | |head_pose.yaw| | distraction +0.4 | > 30° |
| Phone Zone? | Line 562 | gaze x,y | flag + dist +0.3 | complex |
| Long Closure? | Line 541 | closure duration | drowsy +0-0.5 | > 200ms |
| Abnormal Blink? | Line 549 | blink_rate | drowsy +0.2 | <10 or >30 |
| Head Down? | Line 553 | head_pose.pitch | flag + drowsy +0.3 | < -15° |
| Trend Pattern? | Line 588 | microsleep count | drowsy +0.2 | > 2/sec |
| Alert Normal? | Line 569 | max_score | NORMAL | < 0.3 |
| Alert Caution? | Line 571 | max_score | CAUTION | 0.3-0.5 |
| Alert Warning? | Line 573 | max_score | WARNING | 0.5-0.7 |
| Alert Critical? | Line 575 | max_score | CRITICAL | ≥ 0.7 |

---

**Total Decision Points:** 14
**Total Thresholds:** 13 configurable parameters
**Output Metrics:** 15+ fields per frame

