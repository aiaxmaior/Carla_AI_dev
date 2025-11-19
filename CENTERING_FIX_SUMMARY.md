# UI Centering Fix Summary

## Problem
The HUD panel and welcome/selection screens were not centering properly on the ultrawide monitor in a 3-monitor + 1 admin monitor setup.

## Root Cause
1. The pygame `--display` parameter was pointing to the wrong monitor (was using `--display 2`, should use `--display 1`)
2. Centering calculations were using single panel width instead of full window width
3. Pygame monitor indexing differs from xrandr ordering

## Monitor Setup

### Physical Layout (xrandr names):
- **HDMI-0** @ x=0: 3440x1440 (admin/coding monitor - not used for simulation)
- **DP-4** @ x=3440: 1920x1080 (simulation left camera)
- **DP-0** @ x=5360: 3840x1080 (32:9 ultrawide for center cameras)
- **DP-2** @ x=9200: 1920x1080 (simulation right camera)

### Pygame Monitor Indexing:
- pygame Monitor 0 = HDMI-0 (3440x1440)
- pygame Monitor 1 = DP-0 (3840x1080) ← **Ultrawide - use `--display 1`**
- pygame Monitor 2 = DP-2 (1920x1080)
- pygame Monitor 3 = DP-4 (1920x1080)

## Solution

### 1. Use `--display 1` Instead of `--display 2`
The pygame window should start on **pygame Monitor 1** (which is DP-0, the ultrawide).

**Updated in:** `_QDrive_alpha.sh` line 58

### 2. Center UI Elements on Full Window Width
Changed centering calculation from:
```python
center_x = panel_x0 + (panel_w // 2)  # Centers on single panel
```

To:
```python
center_x = self._W // 2  # Centers on full window width
```

**Files modified:**
- `TitleScreen.py` line 564: `self.center_x = self._W // 2`
- `TitleScreen.py` line 290: `center_x = _W // 2` (consolidated_select_screen function)

### 3. Coordinate System
When using `--display 1`, the pygame window coordinate system is:
- Window width: 7680px (4 × 1920px logical monitors)
- Window (0, 0) = left edge of DP-0 (ultrawide)
- Window center (3840, 540) = **center of the ultrawide** ✓

Logical panel layout within pygame window:
```
Panel 0: x=0    to x=1920  (left half of ultrawide)
Panel 1: x=1920 to x=3840  (right half of ultrawide)
Panel 2: x=3840 to x=5760  (DP-2 monitor)
Panel 3: x=5760 to x=7680  (DP-4 monitor)
```

Center at x=3840 is perfectly positioned at the middle of the ultrawide (between Panel 0 and Panel 1).

## Usage

### Recommended Command:
```bash
./_QDrive_alpha.sh --sync --res 1920x1080
```

The `--display 1` is now included by default in the script.

### Manual Override:
```bash
python Main.py --sync --res 1920x1080 --display 1 --host localhost --port 2000
```

## Files Changed
1. `TitleScreen.py` - Changed centering to use full window width
2. `_QDrive_alpha.sh` - Added `--display 1` as default
3. `Tools/diagnose_display_layout.py` - Created diagnostic tool

## Backups Created
- `TitleScreen.py.backup_centering`
- `HUD.py.backup_centering`
- `PreWelcomeSelect.py.backup_centering`

## Testing
Use the diagnostic tool to verify centering:
```bash
conda run -n carla python Tools/diagnose_display_layout.py 1
```

The MAGENTA center line should appear at the center of the ultrawide monitor.

## Notes
- PreWelcomeSelect.py creates its own 1500x800 window and doesn't need fixing
- HUD.py already uses proper centering (was not modified in this fix)
- The setup assumes monitors are pre-configured to target resolutions before pygame initialization
