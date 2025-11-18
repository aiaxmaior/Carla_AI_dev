# CORRECTED VISIONPERCEPTION INITIALIZATION FOR WORLD.PY
# ========================================================

## Issues Found in Your Code Snippet:

1. **Typo:** `cam_actors` (plural) vs `cam_actor` (singular)
2. **Type mismatch:** Setting `cam_actor = [cam_actor, cam_actor_r]` but class expects single camera
3. **Typo in VisionPerception.py line 53:** `name_ti_id` should be `name_to_id`
4. **Typo in VisionPerception.py line 58:** `min_visibility_ration` should be `min_visibility_ratio`
5. **Wrong attribute name:** You wrote `occluion_check_enabled`, should be `occlusion_check_enabled`

---

## CORRECTED VERSION FOR WORLD.PY

### Option 1: Use LEFT_DASH Camera Only (Recommended)

Use the left-dash camera as the reference for VisionPerception since it's your primary forward view.

```python
# In World.py, after camera_manager is initialized

try:
    cm = self.world.camera_manager
    
    # Get the left-dash camera (primary forward view)
    cam_actor = cm.sensors.get('left_dash_cam')
    
    if cam_actor:
        if not hasattr(self, 'perception') or self.perception is None:
            # Initialize VisionPerception with left-dash camera
            self.perception = Perception(
                self.world,
                image_width=1920,
                image_height=1080,
                camera_actor=cam_actor
            )
            
            # Enable occlusion checking (FIXED TYPO)
            self.perception.occlusion_check_enabled = True
            
            # Set visibility thresholds
            self.perception.min_visible_pixels = 50
            self.perception.min_visibility_ratio = 0.1  # Accept if 10%+ visible
            
            logging.info(f"[Vision] Perception initialized with left_dash camera")
            logging.info(f"[Vision] FOV={self.perception.fov_deg:.1f}°, "
                        f"Resolution={self.perception.image_width}x{self.perception.image_height}")
        else:
            # Update camera reference if perception already exists
            self.perception.set_camera(cam_actor)
            logging.info(f"[Vision] Camera reference updated")
            
except Exception as e:
    logging.error(f"[Vision] Failed to initialize Perception: {e}")
    self.perception = None
```

**Why left-dash only?**
- VisionPerception uses a single camera's FOV and intrinsics
- Left-dash is your primary forward view (where most action happens)
- Simpler and cleaner than trying to merge two camera views

---

### Option 2: Switch Between Cameras Dynamically

If you want to use BOTH cameras at different times:

```python
# In World.py

try:
    cm = self.world.camera_manager
    
    # Get both dash cameras
    self.left_dash_cam = cm.sensors.get('left_dash_cam')
    self.right_dash_cam = cm.sensors.get('right_dash_cam')
    
    # Initialize with left camera by default
    active_camera = self.left_dash_cam
    
    if active_camera:
        if not hasattr(self, 'perception') or self.perception is None:
            self.perception = Perception(
                self.world,
                image_width=1920,
                image_height=1080,
                camera_actor=active_camera
            )
            
            # Enable occlusion
            self.perception.occlusion_check_enabled = True
            self.perception.min_visible_pixels = 50
            self.perception.min_visibility_ratio = 0.1
            
            logging.info(f"[Vision] Perception initialized (multi-camera mode)")
        else:
            self.perception.set_camera(active_camera)
            
except Exception as e:
    logging.error(f"[Vision] Initialization failed: {e}")
    self.perception = None

# Add method to switch cameras
def switch_perception_camera(self, camera_name='left_dash_cam'):
    """Switch VisionPerception to different camera"""
    if not hasattr(self, 'perception') or self.perception is None:
        return
    
    cm = self.world.camera_manager
    cam_actor = cm.sensors.get(camera_name)
    
    if cam_actor:
        self.perception.set_camera(cam_actor)
        logging.info(f"[Vision] Switched to {camera_name}")
```

**Usage:**
```python
# In game loop, switch cameras based on context
if looking_right:
    world_obj.switch_perception_camera('right_dash_cam')
else:
    world_obj.switch_perception_camera('left_dash_cam')
```

---

### Option 3: Dual Perception Instances (Advanced)

If you need BOTH cameras active simultaneously:

```python
# In World.py

try:
    cm = self.world.camera_manager
    
    left_cam = cm.sensors.get('left_dash_cam')
    right_cam = cm.sensors.get('right_dash_cam')
    
    # Create separate Perception instance for each camera
    self.perception_left = None
    self.perception_right = None
    
    if left_cam:
        self.perception_left = Perception(
            self.world,
            image_width=1920,
            image_height=1080,
            camera_actor=left_cam
        )
        self.perception_left.occlusion_check_enabled = True
        self.perception_left.min_visible_pixels = 50
        self.perception_left.min_visibility_ratio = 0.1
        logging.info(f"[Vision] Left perception initialized")
    
    if right_cam:
        self.perception_right = Perception(
            self.world,
            image_width=1920,
            image_height=1080,
            camera_actor=right_cam
        )
        self.perception_right.occlusion_check_enabled = True
        self.perception_right.min_visible_pixels = 50
        self.perception_right.min_visibility_ratio = 0.1
        logging.info(f"[Vision] Right perception initialized")
    
except Exception as e:
    logging.error(f"[Vision] Dual perception setup failed: {e}")

# Usage in game loop
def get_visible_objects(self):
    """Get objects visible in both cameras"""
    objects = []
    
    if self.perception_left:
        left_objects = self.perception_left.compute(max_objects=32, include_2d=True)
        objects.extend([{**obj, 'camera': 'left'} for obj in left_objects])
    
    if self.perception_right:
        right_objects = self.perception_right.compute(max_objects=32, include_2d=True)
        objects.extend([{**obj, 'camera': 'right'} for obj in right_objects])
    
    # Remove duplicates (same actor in both cameras)
    seen_ids = set()
    unique_objects = []
    for obj in objects:
        if obj['track_id'] not in seen_ids:
            unique_objects.append(obj)
            seen_ids.add(obj['track_id'])
    
    return unique_objects
```

---

## REQUIRED FIXES IN VISIONPERCEPTION.PY

Your uploaded VisionPerception.py has typos that need fixing:

```python
# Line 53 - TYPO FIX
# OLD:
self.name_ti_id[k.lower()] = v

# NEW:
self.name_to_id[k.lower()] = v


# Line 58 - TYPO FIX
# OLD:
self.min_visibility_ration = 0.15

# NEW:
self.min_visibility_ratio = 0.15
```

---

## SEMANTIC SEGMENTATION SETUP (For Occlusion Detection)

If you want occlusion detection to work, you need to feed segmentation data to VisionPerception:

```python
# In HUD.py CameraManager, add semantic segmentation camera

def _spawn_segmentation_camera(self):
    """Spawn semantic segmentation camera for VisionPerception occlusion"""
    bp_library = self.world.get_blueprint_library()
    seg_bp = bp_library.find("sensor.camera.semantic_segmentation")
    
    # Match left-dash camera settings
    seg_bp.set_attribute("image_size_x", "1920")
    seg_bp.set_attribute("image_size_y", "1080")
    seg_bp.set_attribute("fov", "90")
    
    # Spawn at same location as left-dash
    transform = self.sensors['left_dash_cam'].get_transform()
    
    self.seg_camera = self.world.spawn_actor(
        seg_bp,
        transform,
        attach_to=self._parent
    )
    
    # Listen for segmentation data
    self.seg_camera.listen(self._on_segmentation_image)
    
    logging.info("[Vision] Segmentation camera spawned for occlusion detection")

def _on_segmentation_image(self, image):
    """Process segmentation image for VisionPerception"""
    # Convert to numpy array
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    seg_tags = array[:, :, 2]  # Red channel = semantic tag
    
    # Feed to VisionPerception
    if hasattr(self.world_obj, 'perception') and self.world_obj.perception:
        # For instance segmentation, you'd need a separate camera
        # For now, just semantic
        self.world_obj.perception.set_seg_maps(seg_tags, None)
```

**Call this in CameraManager.__init__():**
```python
# After spawning RGB cameras
if enable_occlusion_detection:
    self._spawn_segmentation_camera()
```

---

## USAGE IN GAME LOOP

```python
# In Main.py game loop

if world_obj.perception:
    # Compute visible objects (with occlusion filtering)
    visible_objects = world_obj.perception.compute(
        max_objects=32,
        include_2d=True,  # Needed for occlusion check
        purity_min=0.0    # No segmentation filtering (just occlusion)
    )
    
    # Draw bounding boxes or process objects
    for obj in visible_objects:
        track_id = obj['track_id']
        cls = obj['cls']
        distance = obj['distance_m']
        rel_speed = obj['rel_speed_mps']
        bbox = obj['bbox_xyxy']  # (x1, y1, x2, y2) in pixels
        
        # Only objects passing occlusion check appear here!
        logging.debug(f"Visible: {cls} at {distance:.1f}m, speed {rel_speed:.1f}m/s")
```

---

## RECOMMENDED APPROACH

**For your Q-DRIVE system, I recommend Option 1** (left-dash camera only):

1. ✅ Simple and clean
2. ✅ Left-dash is your primary forward view
3. ✅ Easy to debug
4. ✅ Sufficient for most safety scenarios

**Why not dual cameras?**
- VisionPerception is designed for single-camera geometry
- Most hazards appear in forward view (left-dash covers it)
- Dual cameras = 2x computation without much benefit for safety

**When to use dual?**
- Lane change assistance (check blind spots)
- Wide-angle hazard detection
- Advanced research scenarios

---

## COMPLETE CORRECTED CODE SNIPPET

```python
# In World.py, where you initialize VisionPerception

try:
    cm = self.camera_manager  # Assuming camera_manager is already set
    
    # Get primary forward camera
    cam_actor = cm.sensors.get('left_dash_cam')
    
    if cam_actor:
        if not hasattr(self, 'perception') or self.perception is None:
            # Initialize VisionPerception
            self.perception = Perception(
                self,  # Pass world_obj (self)
                image_width=1920,
                image_height=1080,
                camera_actor=cam_actor
            )
            
            # Enable occlusion checking (TYPO FIXED)
            self.perception.occlusion_check_enabled = True
            
            # Set visibility thresholds
            self.perception.min_visible_pixels = 50
            self.perception.min_visibility_ratio = 0.1
            
            logging.info(
                f"[Vision] Perception initialized: "
                f"FOV={self.perception.fov_deg:.1f}°, "
                f"Resolution={self.perception.image_width}x{self.perception.image_height}"
            )
        else:
            # Update camera if perception already exists
            self.perception.set_camera(cam_actor)
            logging.info("[Vision] Camera reference updated")
    else:
        logging.warning("[Vision] left_dash_cam not found in sensors")
        self.perception = None
        
except Exception as e:
    logging.error(f"[Vision] Initialization failed: {e}", exc_info=True)
    self.perception = None
```

---

## SUMMARY OF FIXES

**Your code had:**
1. ❌ `cam_actors` typo (should be `cam_actor`)
2. ❌ Tried to pass list of cameras (class expects single camera)
3. ❌ `occluion_check_enabled` typo (missing 's')
4. ❌ VisionPerception.py has `name_ti_id` and `min_visibility_ration` typos

**Corrected version:**
1. ✅ Uses single camera (left_dash)
2. ✅ Fixed all typos
3. ✅ Added proper error handling
4. ✅ Clear logging for debugging

---

Want me to create the complete corrected VisionPerception.py file with the typos fixed?
