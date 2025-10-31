"""
LaneDetection.py
TensorRT-optimized lane detection module for Q-DRIVE Alpha.
Supports both Jetson Orin and PC deployment with lightweight segmentation.
"""

import numpy as np
import cv2
import os
from typing import Tuple, Optional, List, Dict
import logging

# TensorRT imports (with fallback)
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    logging.warning("TensorRT not available, falling back to OpenCV/NumPy")

# TensorFlow/TRT imports for model conversion
try:
    import tensorflow as tf
    from tensorflow.python.compiler.tensorrt import trt_convert as trt_tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class TRTSegmentationEngine:
    """
    TensorRT inference engine for lane segmentation.
    Optimized for Jetson Orin and desktop GPUs.
    """
    def __init__(self, engine_path: str, input_shape=(720, 1280, 3)):
        self.logger = logging.getLogger(__name__)
        self.engine_path = engine_path
        self.input_shape = input_shape

        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT not available. Install with: pip install tensorrt")

        # TensorRT logger
        self.trt_logger = trt.Logger(trt.Logger.WARNING)

        # Load engine
        self.engine = None
        self.context = None
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = None

        self._load_engine()

    def _load_engine(self):
        """Load TensorRT engine from file"""
        if not os.path.exists(self.engine_path):
            self.logger.error(f"Engine file not found: {self.engine_path}")
            raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")

        # Load serialized engine
        with open(self.engine_path, 'rb') as f:
            runtime = trt.Runtime(self.trt_logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT engine")

        self.context = self.engine.create_execution_context()

        # Allocate buffers
        for binding in self.engine:
            binding_idx = self.engine.get_binding_index(binding)
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

        # Create CUDA stream
        self.stream = cuda.Stream()

        self.logger.info(f"TensorRT engine loaded: {self.engine_path}")
        self.logger.info(f"Inputs: {len(self.inputs)}, Outputs: {len(self.outputs)}")

    def infer(self, img: np.ndarray) -> np.ndarray:
        """
        Run inference on input image.

        Args:
            img: Input RGB image (HxWxC)

        Returns:
            Binary segmentation mask (HxW)
        """
        # Preprocess
        img_preprocessed = self._preprocess(img)

        # Copy input to device
        np.copyto(self.inputs[0]['host'], img_preprocessed.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)

        # Execute inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # Copy output from device
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()

        # Postprocess
        output = self._postprocess(self.outputs[0]['host'])

        return output

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image for inference"""
        # Resize if needed
        if img.shape[:2] != self.input_shape[:2]:
            img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Channel-first format for TensorRT (CHW)
        img = np.transpose(img, (2, 0, 1))

        return img

    def _postprocess(self, output: np.ndarray) -> np.ndarray:
        """Postprocess network output to binary mask"""
        # Reshape output
        output = output.reshape(self.input_shape[:2])

        # Threshold to binary
        binary_mask = (output > 0.5).astype(np.uint8) * 255

        return binary_mask

    def __del__(self):
        """Cleanup CUDA resources"""
        if self.stream:
            self.stream.synchronize()


class TRTLaneSegmenter:
    """
    TensorRT-accelerated lane segmenter.
    Drop-in replacement for LaneSegmenter with GPU acceleration.
    """
    def __init__(self, engine_path: str = None, use_trt: bool = True):
        self.logger = logging.getLogger(__name__)
        self.use_trt = use_trt and TRT_AVAILABLE and engine_path is not None

        if self.use_trt:
            try:
                self.trt_engine = TRTSegmentationEngine(engine_path)
                self.logger.info("Using TensorRT acceleration for lane segmentation")
            except Exception as e:
                self.logger.warning(f"Failed to load TensorRT engine: {e}, falling back to OpenCV")
                self.use_trt = False
                self.fallback_segmenter = LaneSegmenter()
        else:
            self.fallback_segmenter = LaneSegmenter()
            self.logger.info("Using OpenCV fallback for lane segmentation")

    def segment(self, img, **kwargs):
        """
        Segment image to detect lane pixels.
        Compatible with LaneSegmenter interface.
        """
        if self.use_trt:
            return self.trt_engine.infer(img)
        else:
            return self.fallback_segmenter.segment(img, **kwargs)


class Line:
    """
    Represents a single lane line with tracking history.
    """
    def __init__(self, max_history=20):
        self.detected = False
        self.recent_xfitted = []
        self.bestx = None
        self.best_fit = None
        self.current_fit = [np.array([False])]
        self.radius_of_curvature = None
        self.line_base_pos = None
        self.diffs = np.array([0, 0, 0], dtype='float')
        self.allx = None
        self.ally = None
        self.max_history = max_history

    def reset(self):
        """Reset line detection state"""
        self.detected = False
        self.recent_xfitted = []
        self.bestx = None


class PerspectiveTransform:
    """
    Handles perspective transformation (bird's-eye view) for lane detection.
    """
    def __init__(self, img_width=1280, img_height=720):
        self.img_width = img_width
        self.img_height = img_height
        self.src = None
        self.dst = None
        self.M = None
        self.Minv = None
        self._compute_transform()

    def _compute_transform(self):
        """Compute perspective transformation matrices"""
        # Define the perspective transformation area
        bot_width = 0.76  # percent of bottom
        mid_width = 0.17
        height_pct = 0.66
        bottom_trim = 0.935

        self.src = np.float32([
            [self.img_width * (0.5 - mid_width/2), self.img_height * height_pct],
            [self.img_width * (0.5 + mid_width/2), self.img_height * height_pct],
            [self.img_width * (0.5 + bot_width/2), self.img_height * bottom_trim],
            [self.img_width * (0.5 - bot_width/2), self.img_height * bottom_trim]
        ])

        offset = self.img_width * 0.2
        self.dst = np.float32([
            [offset, 0],
            [self.img_width - offset, 0],
            [self.img_width - offset, self.img_height],
            [offset, self.img_height]
        ])

        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.Minv = cv2.getPerspectiveTransform(self.dst, self.src)

    def warp(self, img):
        """Apply perspective transform (to bird's-eye view)"""
        img_size = (img.shape[1], img.shape[0])
        return cv2.warpPerspective(img, self.M, img_size, flags=cv2.INTER_LINEAR)

    def unwarp(self, img):
        """Apply inverse perspective transform (back to original view)"""
        img_size = (img.shape[1], img.shape[0])
        return cv2.warpPerspective(img, self.Minv, img_size, flags=cv2.INTER_LINEAR)


class LaneSegmenter:
    """
    Lightweight lane segmentation using color/gradient thresholding.
    Optimized for edge devices (no deep learning required).
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def abs_sobel_thresh(self, img, orient='x', sobel_kernel=3, thresh=(0, 255)):
        """Calculate directional gradient"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        else:
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

        grad_binary = np.zeros_like(scaled_sobel)
        grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        return grad_binary

    def color_thresh(self, img, s_thresh=(0, 255), v_thresh=(0, 255)):
        """Apply color space thresholding (HLS + HSV)"""
        # HLS S-channel
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:, :, 2]
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

        # HSV V-channel
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        v_channel = hsv[:, :, 2]
        v_binary = np.zeros_like(v_channel)
        v_binary[(v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1])] = 1

        c_binary = np.zeros_like(s_channel)
        c_binary[(s_binary == 1) & (v_binary == 1)] = 1

        return c_binary

    def segment(self, img, gradx_thresh=(25, 255), grady_thresh=(10, 255),
                s_thresh=(100, 255), v_thresh=(50, 255)):
        """
        Main segmentation pipeline combining gradients and color.
        Returns binary image with lane pixels highlighted.
        """
        gradx = self.abs_sobel_thresh(img, orient='x', thresh=gradx_thresh)
        grady = self.abs_sobel_thresh(img, orient='y', thresh=grady_thresh)
        c_binary = self.color_thresh(img, s_thresh=s_thresh, v_thresh=v_thresh)

        # Combine thresholds
        thresh_binary = np.zeros_like(img[:, :, 0])
        thresh_binary[((gradx == 1) & (grady == 1)) | (c_binary == 1)] = 255

        return thresh_binary


class SlidingWindowDetector:
    """
    Sliding window search for lane line detection.
    """
    def __init__(self, nwindows=9, margin=100, minpix=50):
        self.nwindows = nwindows
        self.margin = margin
        self.minpix = minpix

    def find_lane_pixels(self, binary_warped):
        """
        Use sliding windows to find lane pixels.
        Returns left_fit, right_fit polynomial coefficients and visualization.
        """
        # Histogram of bottom half
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)

        # Find peaks (starting points)
        midpoint = np.int32(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Window parameters
        window_height = np.int32(binary_warped.shape[0] / self.nwindows)
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        # Step through windows
        for window in range(self.nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height

            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin

            # Identify pixels within window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                             (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # Recenter if enough pixels found
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

        # Concatenate indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit polynomial
        if len(leftx) > 0:
            left_fit = np.polyfit(lefty, leftx, 2)
        else:
            left_fit = None

        if len(rightx) > 0:
            right_fit = np.polyfit(righty, rightx, 2)
        else:
            right_fit = None

        return left_fit, right_fit, left_lane_inds, right_lane_inds

    def search_around_poly(self, binary_warped, left_fit, right_fit):
        """
        Search around previous polynomial (faster than sliding window).
        """
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
                          left_fit[2] - self.margin)) &
                         (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
                          left_fit[2] + self.margin)))

        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
                           right_fit[2] - self.margin)) &
                          (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
                           right_fit[2] + self.margin)))

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        if len(leftx) > 0:
            new_left_fit = np.polyfit(lefty, leftx, 2)
        else:
            new_left_fit = left_fit

        if len(rightx) > 0:
            new_right_fit = np.polyfit(righty, rightx, 2)
        else:
            new_right_fit = right_fit

        return new_left_fit, new_right_fit


class LaneGeometry:
    """
    Calculates lane geometry: curvature, vehicle offset, lane width.
    """
    def __init__(self, ym_per_pix=30/720, xm_per_pix=3.7/700):
        self.ym_per_pix = ym_per_pix  # meters per pixel in y
        self.xm_per_pix = xm_per_pix  # meters per pixel in x

    def calculate_curvature(self, ploty, left_fitx, right_fitx):
        """Calculate radius of curvature in meters"""
        y_eval = np.max(ploty)

        # Fit polynomials in world space
        left_fit_cr = np.polyfit(ploty * self.ym_per_pix, left_fitx * self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * self.ym_per_pix, right_fitx * self.xm_per_pix, 2)

        # Calculate curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*self.ym_per_pix +
                        left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*self.ym_per_pix +
                         right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        return left_curverad, right_curverad

    def calculate_vehicle_offset(self, left_fitx, right_fitx, img_width=1280):
        """Calculate vehicle offset from lane center in meters"""
        lane_center = (left_fitx[-1] + right_fitx[-1]) / 2
        vehicle_center = img_width / 2
        offset = (vehicle_center - lane_center) * self.xm_per_pix
        return offset

    def calculate_lane_width(self, left_fitx, right_fitx):
        """Calculate lane width in meters"""
        lane_width_px = np.mean(right_fitx - left_fitx)
        lane_width_m = lane_width_px * self.xm_per_pix
        lane_width_var = np.var(right_fitx - left_fitx)
        return lane_width_m, lane_width_var


class LaneDetector:
    """
    Main lane detection class integrating all components.
    Optimized for real-time performance on Jetson Orin and PC.
    """
    def __init__(self, img_width=1280, img_height=720, camera_matrix=None, dist_coeffs=None,
                 trt_engine_path=None, use_trt=False):
        self.img_width = img_width
        self.img_height = img_height

        # Camera calibration (optional)
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

        # Components
        self.perspective = PerspectiveTransform(img_width, img_height)

        # Segmenter: TensorRT if available and requested, else OpenCV fallback
        if use_trt and trt_engine_path:
            self.segmenter = TRTLaneSegmenter(engine_path=trt_engine_path, use_trt=True)
        else:
            self.segmenter = LaneSegmenter()

        self.window_detector = SlidingWindowDetector()
        self.geometry = LaneGeometry()

        # Lane line trackers
        self.left_line = Line()
        self.right_line = Line()

        # Sanity check thresholds
        self.min_lane_width = 3.0  # meters
        self.max_lane_width = 5.0  # meters
        self.max_lane_var = 500    # pixels
        self.max_fit_diff = 6000   # polynomial fit difference threshold

        self.logger = logging.getLogger(__name__)

    def undistort(self, img):
        """Apply camera distortion correction if calibration available"""
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            return cv2.undistort(img, self.camera_matrix, self.dist_coeffs, None, self.camera_matrix)
        return img

    def detect_lanes(self, img, debug=False):
        """
        Main detection pipeline.

        Args:
            img: Input RGB image
            debug: If True, return debug visualizations

        Returns:
            dict containing lane information and optionally debug images
        """
        # 1. Undistort
        img_undist = self.undistort(img)

        # 2. Segment (threshold)
        img_thresh = self.segmenter.segment(img_undist)

        # 3. Perspective transform (bird's-eye view)
        img_birdeye = self.perspective.warp(img_thresh)

        # 4. Detect lane pixels
        if not self.left_line.detected or not self.right_line.detected:
            # Full sliding window search
            left_fit, right_fit, _, _ = self.window_detector.find_lane_pixels(img_birdeye)
            search_mode = "sliding_window"
        else:
            # Search around previous fit
            left_fit, right_fit = self.window_detector.search_around_poly(
                img_birdeye,
                self.left_line.best_fit[0] if self.left_line.best_fit is not None else self.left_line.current_fit[0],
                self.right_line.best_fit[0] if self.right_line.best_fit is not None else self.right_line.current_fit[0]
            )
            search_mode = "tracking"

        # Handle failed detection
        if left_fit is None or right_fit is None:
            self.logger.warning("Lane detection failed, using previous fit")
            return self._get_previous_result(img_undist)

        # 5. Update line objects
        self.left_line.current_fit = [left_fit]
        self.right_line.current_fit = [right_fit]

        self.left_line.recent_xfitted.append([left_fit])
        self.right_line.recent_xfitted.append([right_fit])

        # Keep history bounded
        if len(self.left_line.recent_xfitted) > self.left_line.max_history:
            self.left_line.recent_xfitted.pop(0)
            self.right_line.recent_xfitted.pop(0)

        # Calculate best fit (moving average)
        if len(self.left_line.recent_xfitted) > 1:
            self.left_line.best_fit = np.mean(np.array(self.left_line.recent_xfitted[-20:]), axis=0)
            self.right_line.best_fit = np.mean(np.array(self.right_line.recent_xfitted[-20:]), axis=0)
        else:
            self.left_line.best_fit = self.left_line.recent_xfitted[-1]
            self.right_line.best_fit = self.right_line.recent_xfitted[-1]

        # 6. Generate lane points
        ploty = np.linspace(0, img_birdeye.shape[0] - 1, img_birdeye.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # 7. Calculate geometry
        left_curverad, right_curverad = self.geometry.calculate_curvature(ploty, left_fitx, right_fitx)
        vehicle_offset = self.geometry.calculate_vehicle_offset(left_fitx, right_fitx, self.img_width)
        lane_width_m, lane_width_var = self.geometry.calculate_lane_width(left_fitx, right_fitx)

        # 8. Sanity check
        self.left_line.diffs = left_fit - self.left_line.best_fit[0]
        self.right_line.diffs = right_fit - self.right_line.best_fit[0]
        lane_continue = np.sum(self.left_line.diffs**2) + np.sum(self.right_line.diffs**2)

        lane_valid = (self.min_lane_width < lane_width_m < self.max_lane_width and
                     lane_width_var < self.max_lane_var and
                     lane_continue < self.max_fit_diff)

        if not lane_valid:
            self.logger.warning(f"Lane sanity check failed: width={lane_width_m:.2f}m, var={lane_width_var:.0f}, diff={lane_continue:.0f}")
            self.left_line.detected = False
            self.right_line.detected = False

            # Revert to best fit
            del self.left_line.recent_xfitted[-1]
            del self.right_line.recent_xfitted[-1]

            left_fit = self.left_line.best_fit[0]
            right_fit = self.right_line.best_fit[0]

            # Recalculate with best fit
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            left_curverad, right_curverad = self.geometry.calculate_curvature(ploty, left_fitx, right_fitx)
            vehicle_offset = self.geometry.calculate_vehicle_offset(left_fitx, right_fitx, self.img_width)
            lane_width_m, lane_width_var = self.geometry.calculate_lane_width(left_fitx, right_fitx)
        else:
            self.left_line.detected = True
            self.right_line.detected = True

        # 9. Create lane overlay
        lane_overlay = self._create_lane_overlay(img_undist, ploty, left_fitx, right_fitx)

        # 10. Prepare result
        result = {
            'overlay': lane_overlay,
            'left_curvature': left_curverad,
            'right_curvature': right_curverad,
            'vehicle_offset': vehicle_offset,
            'lane_width': lane_width_m,
            'lane_valid': lane_valid,
            'search_mode': search_mode,
            'left_fit': left_fit,
            'right_fit': right_fit,
            'ploty': ploty,
            'left_fitx': left_fitx,
            'right_fitx': right_fitx
        }

        if debug:
            result['binary_warped'] = img_birdeye
            result['threshold'] = img_thresh

        return result

    def _create_lane_overlay(self, img_undist, ploty, left_fitx, right_fitx):
        """Create lane visualization overlay"""
        # Create warped blank image
        warp_zero = np.zeros((self.img_height, self.img_width), dtype=np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast points
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw lane
        cv2.fillPoly(color_warp, np.int32([pts]), (0, 255, 0))

        # Draw lane lines
        left_pts = np.transpose(np.vstack((left_fitx, ploty))).astype(np.int32)
        right_pts = np.transpose(np.vstack((right_fitx, ploty))).astype(np.int32)
        cv2.polylines(color_warp, [left_pts], False, (255, 0, 0), thickness=15)
        cv2.polylines(color_warp, [right_pts], False, (255, 0, 0), thickness=15)

        # Unwarp back to original perspective
        newwarp = self.perspective.unwarp(color_warp)

        # Combine with original image
        result = cv2.addWeighted(img_undist, 1, newwarp, 0.3, 0)

        return result

    def _get_previous_result(self, img_undist):
        """Return result using previous best fit when detection fails"""
        if len(self.left_line.recent_xfitted) == 0:
            # No previous data available
            return {
                'overlay': img_undist,
                'left_curvature': 0,
                'right_curvature': 0,
                'vehicle_offset': 0,
                'lane_width': 0,
                'lane_valid': False,
                'search_mode': 'failed'
            }

        left_fit = self.left_line.best_fit[0]
        right_fit = self.right_line.best_fit[0]

        ploty = np.linspace(0, self.img_height - 1, self.img_height)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        left_curverad, right_curverad = self.geometry.calculate_curvature(ploty, left_fitx, right_fitx)
        vehicle_offset = self.geometry.calculate_vehicle_offset(left_fitx, right_fitx, self.img_width)
        lane_width_m, _ = self.geometry.calculate_lane_width(left_fitx, right_fitx)

        lane_overlay = self._create_lane_overlay(img_undist, ploty, left_fitx, right_fitx)

        return {
            'overlay': lane_overlay,
            'left_curvature': left_curverad,
            'right_curvature': right_curverad,
            'vehicle_offset': vehicle_offset,
            'lane_width': lane_width_m,
            'lane_valid': False,
            'search_mode': 'previous_fit'
        }

    def reset(self):
        """Reset lane detection state"""
        self.left_line.reset()
        self.right_line.reset()
    