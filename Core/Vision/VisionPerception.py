# VisionPerception.py
# ============================================================================
# PERF CHECK (file-level):
# ============================================================================
# [X] | Role: Vision perception - ground-truth object detection & 2D projection
# [X] | Hot-path functions: compute() called from HUD.render() (4x per frame!)
# [X] |- Heavy allocs in hot path? YES - queries all actors + projects to 2D
# [ ] |- pandas/pyarrow/json/disk/net in hot path? No
# [ ] | Graphics here? No (produces bbox data for HUD)
# [X] | Data produced (tick schema?): List of object dicts with bbox_xyxy
# [ ] | Storage (Parquet/Arrow/CSV/none): None
# [ ] | Queue/buffer used?: No
# [X] | Session-aware? No
# [ ] | Debug-only heavy features?: None
# Top 3 perf risks:
# 1. [PERF_HOT] compute() VERY EXPENSIVE - world.get_actors() + matrix math per object
# 2. [PERF_HOT] Called 4 TIMES PER FRAME (once per camera tile) in HUD
# 3. [PERF_HOT] No distance culling, no FOV culling - processes ALL actors every call
# 4. [PERF_SPLIT] CRITICAL: Add culling, caching, or throttle to lower FPS
# ============================================================================

import math
import carla
import logging
import imageio
import carla
import numpy as np

class Perception:
    """
    Ground-truth 'vision' emulation using ONLY existing sim state:
      - No new sensors/actors; uses your camera actor for alignment.
      - Returns per-object: id, class, distance (m), relative LoS speed (m/s), bbox_xyxy.

    PERFORMANCE OPTIMIZATIONS:
      - Configurable distance culling (default 50m instead of 1500m)
      - Frame-based result caching (skip re-computation for N frames)
      - Throttling mechanism (run at reduced frequency)
    """

    def __init__(self, world_obj, image_width=1920, image_height=1080,
                 fov_deg=None, camera_actor=None,
                 max_distance=50.0, cache_frames=2, throttle_interval=1):
        """
        Args:
            max_distance: Maximum detection distance in meters (default: 50m)
            cache_frames: Number of frames to cache results (default: 2)
            throttle_interval: Compute every N frames (default: 1 = every frame)
        """
        self.world_obj = world_obj
        self.world = world_obj.world
        self.player = world_obj.player
        self.camera_actor = camera_actor
        self.image_width = int(image_width)
        self.image_height = int(image_height)

        # PERFORMANCE: Configurable distance culling (was hardcoded to 1500m!)
        self.max_distance = float(max_distance)

        # PERFORMANCE: Result caching to avoid re-computation
        self._cache_frames = int(cache_frames)
        self._cache = {}  # Key: (camera_id, include_2d, max_objects) -> (frame_num, results)
        self._frame_counter = 0

        # PERFORMANCE: Throttling mechanism
        self._throttle_interval = int(throttle_interval)
        self._last_compute_frame = -999

        # If a camera actor is given, prefer its FOV
        if camera_actor is not None and fov_deg is None:
            fov_deg = float(camera_actor.attributes.get("fov", "90"))
        self.fov_deg = float(fov_deg if fov_deg is not None else 90.0)
        self._rebuild_intrinsics()

        # Fallback relative mount (used only if camera_actor is None)
        self._cam_rel_transform = carla.Transform(
            carla.Location(x=0.4, y=0.0, z=1.35),
            carla.Rotation(pitch=-2.0, yaw=0.0, roll=0.0)
        )
        self._actors = [

        ]
        self.seg_tags = None
        self.seg_inst = None
        self.seg_include_ids = set()
        self.seg_exclude_ids = set()

        self.city_labels = {}
        self.name_to_id = {}

        if carla and hasattr(carla, "CityObjectLabel"):
            for k in dir(carla.CityObjectLabel):
                if k.startswith("_"):
                    continue
                try:
                    v = int(getattr(carla.CityObjectLabel,k))
                    self.city_labels[v] = k.lower()
                    self.name_ti_id[k.lower()] = v
                except Exception:
                    pass
    # --- Public --------------------------------------------------------------

    def set_seg_maps(self,tags_map,inst_map):
        # Provide per-pixel semantic/instance maps aligned with camera intrinsics
        self.seg_tags = tags_map
        self.seg_inst = inst_map
    
    def set_set_policy(self, include=None, exclude=None):
        # Include/Exclude class policy using ids (int) or names (str)
        inc,exc = set(),set()
        if include:
            for x in include:
                if isinstance(x,int):
                    inc.add(x)
                else:
                    tid = self.name_to_id.get(str(x).lower())
                    if tid is not None:
                        inc.add(tid)
        if exclude:
            for x in exclude:
                if isinstance(x,int):
                    exc.add(x)
                else:
                    tid = self.name_to_id.get(str(x).lower())
                    if tid is not None:
                        exc.add(tid)
        self.seg_include_ids = inc
        self.seg_exclude_ids = exc

    def _set_majority(self,bb_xyxy):
        # Return  tag, inst, purity for bbox in pixel coordinates
        if self.seg_tags is None or self.seg_inst is None or bb_xyxy is None:
            return None, None, 0.0
        x1,y1,x2,y2 = map(int,bb_xyxy)
        h,w = self.seg_tags.shape[:2]
        x1 = max(0,min(w-1,x1)); x2 = max(0,min(w-1,x2))
        y1 = max(0,min(h-1,y1)); x2 = max(0,min(h-1,y2))
        if x2<=x1 or y2<=y1:
            return None,None,
        roi_t = self.seg_tags[y1:y2,x1:x2]
        roi_i = self.seg_inst[y1:y2,x1:x2]

        if roi_t.size == 0:
            return None,None,0.0
        vals,counts = np.unique(roi_t, return_counts = True)
        if counts.size == 0:
            return None, None,0.0
        tag = int(vals[counts.argmax()])
        inst_vals, inst_counts = np.unique(roi_i[roi_t ==tag], return_count=True)
        inst = int(inst_vals[inst_counts.argmax()]) if inst_counts.size else None
        purity = float(counts.max()) / float(roi_t.size)
        return tag, inst, purity

    def set_camera(self, camera_actor, image_width=None, image_height=None, fov_deg=None):
        self.camera_actor = camera_actor
        if image_width:  self.image_width  = int(image_width)
        if image_height: self.image_height = int(image_height)
        if fov_deg is None and camera_actor is not None:
            fov_deg = float(camera_actor.attributes.get("fov", "90"))
        if fov_deg is not None:
            self.fov_deg = float(fov_deg)
        self._rebuild_intrinsics()

    def set_performance_params(self, max_distance=None, cache_frames=None, throttle_interval=None):
        """
        Configure performance optimization parameters at runtime.

        Args:
            max_distance: Maximum detection distance in meters (None = no change)
            cache_frames: Number of frames to cache results (None = no change)
            throttle_interval: Compute every N frames (None = no change)

        Examples:
            # Maximum performance - very aggressive
            perception.set_performance_params(max_distance=30, cache_frames=3, throttle_interval=2)

            # Balanced (default settings)
            perception.set_performance_params(max_distance=50, cache_frames=2, throttle_interval=1)

            # High quality - less aggressive
            perception.set_performance_params(max_distance=100, cache_frames=1, throttle_interval=1)

            # Original behavior (pre-optimization)
            perception.set_performance_params(max_distance=1500, cache_frames=0, throttle_interval=1)
        """
        if max_distance is not None:
            self.max_distance = float(max_distance)
            logging.info(f"[PERF] VisionPerception max_distance set to {self.max_distance}m")

        if cache_frames is not None:
            self._cache_frames = int(cache_frames)
            if cache_frames == 0:
                self._cache.clear()  # Disable caching
            logging.info(f"[PERF] VisionPerception cache_frames set to {self._cache_frames}")

        if throttle_interval is not None:
            self._throttle_interval = int(throttle_interval)
            logging.info(f"[PERF] VisionPerception throttle_interval set to {self._throttle_interval}")

    def compute(self, max_objects=32, include_2d=False, purity_min=0.0, force_recompute=False):
        """
        Returns list of dicts:
        { track_id, cls, distance_m, rel_speed_mps, bbox_xyxy|None }
        Seg filtering applies only when seg maps are present AND include_2d=True
        (we need the bbox ROI to vote a tag).

        PERFORMANCE: Now with caching and throttling!
        Args:
            force_recompute: Bypass cache/throttling and force fresh computation
        """
        # [PERF_HOT] Increment frame counter
        self._frame_counter += 1

        # [PERF_HOT] Generate cache key based on camera + parameters
        camera_id = self.camera_actor.id if self.camera_actor else "ego"
        cache_key = (camera_id, include_2d, max_objects)

        # [PERF_HOT] Check cache first (skip expensive computation!)
        if not force_recompute and cache_key in self._cache:
            cached_frame, cached_results = self._cache[cache_key]
            frames_since_cache = self._frame_counter - cached_frame

            # Return cached results if still fresh
            if frames_since_cache < self._cache_frames:
                # logging.debug(f"[PERF] VisionPerception cache hit! (age: {frames_since_cache} frames)")
                return cached_results

        # [PERF_HOT] Throttling check - skip computation if called too frequently
        if not force_recompute and self._throttle_interval > 1:
            frames_since_compute = self._frame_counter - self._last_compute_frame
            if frames_since_compute < self._throttle_interval:
                # Return stale cached results or empty list if no cache
                if cache_key in self._cache:
                    return self._cache[cache_key][1]
                return []

        # --- EXPENSIVE COMPUTATION STARTS HERE ---
        # logging.debug(f"[PERF] VisionPerception computing fresh results (frame {self._frame_counter})")

        player = self.player
        if not player or not player.is_alive:
            return []

        cam_world = self._camera_world_transform()
        forward = cam_world.get_forward_vector()
        ego_vel = player.get_velocity()
        ego_loc = cam_world.location

        objs = []
        # [PERF_HOT] This is EXPENSIVE - queries all actors in world!
        for a in self.world.get_actors().filter("*"):
            if a.id == player.id:
                continue
            t = a.type_id or ""
            if not (t.startswith("vehicle.") or t.startswith("walker.pedestrian.")):
                continue

            a_loc = a.get_transform().location
            to_target = a_loc - ego_loc
            dist = (to_target.x**2 + to_target.y**2 + to_target.z**2) ** 0.5

            # [PERF_FIX] Distance culling: was 1500m, now configurable (default 50m)
            if dist < 0.5 or dist > self.max_distance:
                continue

            # Gate by FOV and front-facing
            dot = to_target.x*forward.x + to_target.y*forward.y + to_target.z*forward.z
            if dot <= 0:
                continue
            cosang = max(-1.0, min(1.0, dot / dist))
            ang_deg = math.degrees(math.acos(cosang))
            if ang_deg > (self.fov_deg * 0.5 + 5.0):
                continue

            # Relative speed along line-of-sight
            v = a.get_velocity()
            relx, rely, relz = v.x - ego_vel.x, v.y - ego_vel.y, v.z - ego_vel.z
            losx, losy, losz = to_target.x/dist, to_target.y/dist, to_target.z/dist
            rel_speed = relx*losx + rely*losy + relz*losz

            # Base class from actor type
            cls = "vehicle" if t.startswith("vehicle.") else (
                "pedestrian" if t.startswith("walker.pedestrian.") else "other")

            # Project bbox only if requested (needed for seg ROI)
            bbox_xyxy = self._project_actor_bbox(a, cam_world) if include_2d else   None

            # Build the object dict first
            o = {
                "track_id": int(a.id),
                "cls": cls,
                "distance_m": float(dist),
                "rel_speed_mps": float(rel_speed),
                "bbox_xyxy": bbox_xyxy
            }

            # --- Optional segmentation refinement/filtering ---
            # Only possible if we have seg maps AND a bbox to define the ROI
            if include_2d and self.seg_tags is not None and self.seg_inst is not None and bbox_xyxy is not None:
                tag_id, inst_id, purity = self._seg_majority(bbox_xyxy)

                # Optional purity gate (default 0.0 = disabled)
                if purity_min and purity < purity_min:
                    continue

                # Include/exclude policy by tag id
                if self.seg_include_ids and (tag_id is None or tag_id not in self.seg_include_ids):
                    continue
                if tag_id is not None and tag_id in self.seg_exclude_ids:
                    continue

                # Upgrade class label; prefer seg instance as stable id if you want
                if tag_id is not None:
                    o["cls"] = self.city_labels.get(tag_id, f"class_{tag_id}")
                # keep your actor id as track_id; or uncomment next two lines to prefer seg id when available
                # if inst_id is not None:
                #     o["track_id"] = f"seg:{inst_id}"

            objs.append(o)

        objs.sort(key=lambda d: d["distance_m"])
        results = objs[:max_objects]

        # [PERF_HOT] Cache results for future frames
        self._cache[cache_key] = (self._frame_counter, results)
        self._last_compute_frame = self._frame_counter

        # logging.info(f"[PERF] Computed {len(results)} objects (frame {self._frame_counter})")
        return results
    
    def _rebuild_intrinsics(self):
        self.fx = self.image_width / (2.0 * math.tan(math.radians(self.fov_deg) / 2.0))
        self.fy = self.fx
        self.cx = self.image_width / 2.0
        self.cy = self.image_height / 2.0

    def _camera_world_transform(self):
        if self.camera_actor is not None:
            return self.camera_actor.get_transform()

        # Compose base + relative by application (no Transform * Transform)
        base = self.player.get_transform()
        rel  = self._cam_rel_transform

        # 1) rotate & translate the relative location into world
        world_loc = base.transform(rel.location)

        # 2) add Euler angles for rotation (CARLA rotations are in degrees)
        brot, rrot = base.rotation, rel.rotation
        world_rot = carla.Rotation(
            pitch=brot.pitch + rrot.pitch,
            yaw  =brot.yaw   + rrot.yaw,
            roll =brot.roll  + rrot.roll,
        )
        return carla.Transform(world_loc, world_rot)

    def _project_actor_bbox(self, actor, cam_world):
        """
        Project the actor's 3D bounding-box corners to image space and
        return a tight xyxy box. Returns None if fully off-screen.

        CARLA camera coords: X forward, Y right, Z up.
        Image plane: u right, v down.
        """
        bb = actor.bounding_box
        ext = bb.extent

        # 8 corners in bbox local coords (centered at bb.location)
        corners = [
            carla.Location(x= x, y= y, z= z)
            for x in ( ext.x, -ext.x)
            for y in ( ext.y, -ext.y)
            for z in ( ext.z, -ext.z)
        ]

        actor_tf = actor.get_transform()
        bb_tf = carla.Transform(bb.location, getattr(bb, "rotation", carla.Rotation()))

        # Build world->camera extrinsics as a flat row-major 4x4
        cam_inv = self._inverse_matrix_from_transform(cam_world)

        def mult(m, v):
            # m: list[16] row-major; v: (x,y,z,1)
            return (
                m[0]*v[0] + m[1]*v[1] + m[2]*v[2] + m[3]*v[3],   # Xc
                m[4]*v[0] + m[5]*v[1] + m[6]*v[2] + m[7]*v[3],   # Yc
                m[8]*v[0] + m[9]*v[1] + m[10]*v[2] + m[11]*v[3], # Zc
                m[12]*v[0]+ m[13]*v[1]+ m[14]*v[2]+ m[15]*v[3],
            )

        xs, ys = [], []
        for c in corners:
            # Compose transforms by application (no Transform * Transform):
            # bbox local -> bbox frame -> world
            p_w = actor_tf.transform(bb_tf.transform(c))
            Xc, Yc, Zc, _ = mult(cam_inv, (p_w.x, p_w.y, p_w.z, 1.0))

            # In front of camera means X (forward) > 0
            if Xc <= 0.001:
                continue

            # Pinhole projection (u right, v down)
            u = self.fx * (Yc / Xc) + self.cx         # Y maps to image x
            v = self.fy * (-Zc / Xc) + self.cy        # Z maps to image y (invert because image y goes down)

            xs.append(u); ys.append(v)

        if not xs:
            return None

        x1 = max(0.0, min(xs));           y1 = max(0.0, min(ys))
        x2 = min(self.image_width-1,  max(xs));  y2 = min(self.image_height-1, max(ys))
        if x2 <= x1 or y2 <= y1:
            return None
        return [float(x1), float(y1), float(x2), float(y2)]


    def _inverse_matrix_from_transform(self, t: carla.Transform):
        """
        Build inv([R|t]) = [R^T | -R^T t] as a flat row-major 4x4 list of floats.
        Rotation order matches CARLA's (yaw, pitch, roll).
        """
        import math
        loc = t.location
        rot = t.rotation  # degrees

        cy, sy = math.cos(math.radians(rot.yaw)),   math.sin(math.radians(rot.yaw))
        cp, sp = math.cos(math.radians(rot.pitch)), math.sin(math.radians(rot.pitch))
        cr, sr = math.cos(math.radians(rot.roll)),  math.sin(math.radians(rot.roll))

        # R = Rz(yaw) * Ry(pitch) * Rx(roll)
        R = [
            cy*cp,              cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr,
            sy*cp,              sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr,
            -sp,                cp*sr,              cp*cr
        ]
        # Rt
        Rt = [R[0], R[3], R[6],
            R[1], R[4], R[7],
            R[2], R[5], R[8]]

        tx, ty, tz = float(loc.x), float(loc.y), float(loc.z)
        itx = -(Rt[0]*tx + Rt[1]*ty + Rt[2]*tz)
        ity = -(Rt[3]*tx + Rt[4]*ty + Rt[5]*tz)
        itz = -(Rt[6]*tx + Rt[7]*ty + Rt[8]*tz)

        return [
            float(Rt[0]), float(Rt[1]), float(Rt[2]),  itx,
            float(Rt[3]), float(Rt[4]), float(Rt[5]),  ity,
            float(Rt[6]), float(Rt[7]), float(Rt[8]),  itz,
            0.0,          0.0,          0.0,           1.0
        ]

    def _vehicle_size_category(self, ext):
        """Rough size-based bucket using bbox extents (meters)."""
        L, W, H = 2*ext.x, 2*ext.y, 2*ext.z
        if L < 2.5 and W < 0.8:                return "bicycle/motorcycle"
        if L > 8.0:                            return "bus"
        if L > 6.0 and H > 2.5:                return "truck"
        if 4.5 <= L <= 6.0 and H > 1.8:        return "van"
        return "car"

    def _bearing_and_heading(self, cam_tf, to_target, actor_yaw_deg):
        """Signed bearing of target relative to camera, and actor heading relative to camera yaw."""
        import math
        fwd = cam_tf.get_forward_vector()
        right = cam_tf.get_right_vector()
        dist = math.sqrt(to_target.x**2 + to_target.y**2 + to_target.z**2)
        # bearing magnitude
        cosang = max(-1.0, min(1.0, (to_target.x*fwd.x + to_target.y*fwd.y + to_target.z*fwd.z) / max(dist, 1e-6)))
        ang = math.degrees(math.acos(cosang))
        # left/right sign from right-vector dot
        sign = 1.0 if (to_target.x*right.x + to_target.y*right.y + to_target.z*right.z) >= 0.0 else -1.0
        bearing = sign * ang
        # heading relative: actor yaw minus camera yaw, wrapped to [-180,180]
        cam_yaw = cam_tf.rotation.yaw
        head_rel = ((actor_yaw_deg - cam_yaw + 180.0) % 360.0) - 180.0
        return bearing, head_rel

    def _describe_actor(self, actor, bb, cam_tf, to_target, dist_m, rel_speed_mps):
        """Return a dict of rich, human-friendly fields + a prebuilt label string."""
        import math
        t = actor.type_id or ""
        ax = getattr(actor, "attributes", {}) or {}
        color = ax.get("color")
        model = t.split(".", 1)[1] if "." in t else t  # e.g. "tesla.model3" or "walker.pedestrian.0001"
        v = actor.get_velocity()
        speed = math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)

        # class/kind
        if t.startswith("walker.pedestrian."):
            kind = "pedestrian"
            gait = "running" if speed > 2.2 else ("walking" if speed > 0.5 else "standing")
        elif kind =='vehicle':
            subtype = self._vehicle_size_category(bb.extent)
        #elif: kind in ("")
        
        bearing, heading_rel = self._bearing_and_heading(cam_tf, to_target, actor.get_transform().rotation.yaw)

        # time-to-collision (positive only when closing)
        ttc = dist_m / (-rel_speed_mps) if rel_speed_mps < -0.1 else None

        # physical size (approx, meters)
        size_m = (2*bb.extent.x, 2*bb.extent.y, 2*bb.extent.z)

        # label text for HUD
        if kind == "pedestrian":
            label = f"{gait} ped  {dist_m:.1f}m  brg {bearing:+.0f}°  v {speed:.1f}m/s"
            if ttc is not None: label += f"  TTC {ttc:.1f}s"
        else:
            label = f"{subtype} {model}  {dist_m:.1f}m  brg {bearing:+.0f}°  v {speed:.1f}m/s"
            if ttc is not None: label += f"  TTC {ttc:.1f}s"
            if color: label += f"  {color}"

        return {
            "kind": kind,                         # "vehicle" | "pedestrian"
            "model": model,                       # e.g., "tesla.model3" or "pedestrian.0001"
            "color": color,                       # may be None
            "speed_mps": speed,
            "bearing_deg": bearing,               # signed: +right, -left
            "heading_rel_deg": heading_rel,       # actor heading relative to camera yaw
            "size_m": size_m,                     # (L,W,H)
            "ttc_s": ttc,                         # None if not closing
            "label": label                        # ready-to-draw text
        }
