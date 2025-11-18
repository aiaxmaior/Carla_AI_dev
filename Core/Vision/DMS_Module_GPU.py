"""
GPU-Accelerated Driver Monitoring System
Allocates processing to specific GPU (GPU2)
Author: Arjun Joshi
Date: 10.14.2025
"""
# ============================================================================
# PERF CHECK (file-level):
# ============================================================================
# [X] | Role: GPU-accelerated DMS (alternate version of DMS_Module.py)
# [X] | Hot-path functions: Similar to DMS_Module.py (if used)
# [X] |- Heavy allocs in hot path? Moderate (if used) - numpy arrays, GPU ops
# [ ] |- pandas/pyarrow/json/disk/net in hot path? No
# [X] | Graphics here? YES - OpenCV + MediaPipe with GPU acceleration
# [X] | Data produced (tick schema?): DriverState dataclass (if active)
# [ ] | Storage (Parquet/Arrow/CSV/none): None
# [X] | Queue/buffer used?: Likely (if follows DMS_Module.py pattern)
# [ ] | Session-aware? No
# [ ] | Debug-only heavy features?: None visible
# Top 3 perf risks:
# 1. [PERF_HOT] MediaPipe + GPU operations (if active)
# 2. [PERF_OK] GPU allocation at import time (similar to DMS_Module.py)
# 3. [PERF_SPLIT] NOTE: Appears to be alternate version - check which is used
# ============================================================================

import os
import cv2 as cv
import numpy as np
import mediapipe as mp
import time
import logging
import subprocess
import torch

from dataclasses import dataclass
from typing import Optional, Tuple

# ============================
# GPU ALLOCATION 
# ============================

# Method 1: Set CUDA device for all CUDA operations
gpu_id = 0
def prompt_gpu_id() :
    gpu_choice=0
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True
        )
        
        gpu_list = []
        for line in result.stdout.strip().split('\n'):
            if line:
                idx, name = line.split(', ', 1)
                gpu_list.append((idx.strip(), name.strip()))
                print(f"  {idx.strip()}) {name.strip()}")

    # Establish default gpu as 1
        if len(gpu_list)>1 and gpu_list[1]:
            gpu_choice = 1
            gpu_choice = input("\nSelect GPU ID for local models (default 1): ").strip()
        else:
            gpu_choice = 0
        return gpu_choice
    except Exception as e:
        logging.warning(f"Could not query GPUs {e}")
        print('GPU set to primary')

try:
    gpu_id = prompt_gpu_id()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Use PCI bus ordering
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
except Exception as e:
    logging.error(f"Error assigning cuda device, defaulting to [cpu]")


# For TensorFlow backend (if MediaPipe uses it)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

class GPUAcceleratedDMS:
    """DMS with GPU acceleration on specified device"""
    
    def __init__(self, gpu_device=0, use_gpu=True):
        """
        Initialize DMS with GPU acceleration
        Args:
            gpu_device: GPU index to use (2 for GPU2)
            use_gpu: Enable GPU acceleration
        """
        self.gpu_device = gpu_device
        self.use_gpu = use_gpu
        
        # Configure GPU before initializing MediaPipe
        if self.use_gpu:
            self._configure_gpu()
        
        # Initialize MediaPipe with GPU support
        self._init_mediapipe()
        
        # Initialize OpenCV with CUDA if available
        self._init_opencv_cuda()
        
        logging.info(f"DMS initialized on {'GPU' + str(gpu_device) if use_gpu else 'CPU'}")
    
    def _configure_gpu(self):
        """Configure GPU settings for optimal performance"""
        try:
            # Check if CUDA is available
            if torch.cuda.is_available():
                torch.cuda.set_device(self.gpu_device)
                self.torch_device = torch.device(f'cuda:{self.gpu_device}')
                logging.info(f"PyTorch using GPU{self.gpu_device}: {torch.cuda.get_device_name(self.gpu_device)}")
            else:
                self.torch_device = torch.device('cpu')
                logging.warning("PyTorch CUDA not available, falling back to CPU")
        except ImportError:
            logging.info("PyTorch not installed, skipping torch GPU setup")
        
        # Check TensorFlow GPU (if using TF backend)
        try:
            import tensorflow as tf
            
            # List available GPUs
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    # Use only specified GPU
                    tf.config.set_visible_devices(gpus[self.gpu_device], 'GPU')
                    
                    # Enable memory growth to avoid allocating all GPU memory
                    tf.config.experimental.set_memory_growth(gpus[self.gpu_device], True)
                    
                    logging.info(f"TensorFlow using GPU{self.gpu_device}: {gpus[self.gpu_device].name}")
                except RuntimeError as e:
                    logging.error(f"TensorFlow GPU configuration failed: {e}")
        except ImportError:
            logging.info("TensorFlow not installed, skipping TF GPU setup")
    
    def _init_mediapipe(self):
        """Initialize MediaPipe with GPU acceleration"""
        self.mp_face_mesh = mp.solutions.face_mesh
        
        if self.use_gpu:
            # MediaPipe GPU delegate configuration
            # Note: Full GPU support requires building from source or using TFLite GPU delegate
            
            # Option 1: Standard initialization (uses GPU when available)
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                static_image_mode=False  # Enable video mode for better GPU utilization
            )
            
            # Option 2: If you have MediaPipe GPU build (requires special installation)
            # self.face_mesh = self._create_gpu_face_mesh()
        else:
            # CPU-only mode
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
    
    def _create_gpu_face_mesh(self):
        """
        Create GPU-accelerated FaceMesh (requires MediaPipe GPU build)
        This is for advanced users who build MediaPipe from source
        """
        try:
            import mediapipe as mp
            from mediapipe.python.solutions import face_mesh as fm
            from mediapipe.calculators.tensor import inference_calculator_pb2
            
            # Create custom graph config for GPU
            graph_config = fm.FACE_MESH_GRAPH_CONFIG
            
            # Modify calculator options for GPU
            for calculator in graph_config.node:
                if calculator.name.endswith('InferenceCalculator'):
                    options = calculator.options.Extensions[
                        inference_calculator_pb2.InferenceCalculatorOptions.ext
                    ]
                    
                    # Set GPU delegate
                    options.delegate.gpu.use_advanced_gpu_api = True
                    options.delegate.gpu.use_gpu_plugin_for_input_conversion = True
            
            # Create face mesh with GPU config
            return mp.FaceMesh(graph_config=graph_config)
            
        except Exception as e:
            logging.warning(f"GPU FaceMesh creation failed, using default: {e}")
            return self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
    
    def _init_opencv_cuda(self):
        """Initialize OpenCV with CUDA support if available"""
        self.cuda_available = False
        
        try:
            # Check if OpenCV was built with CUDA
            if cv.cuda.getCudaEnabledDeviceCount() > 0:
                cv.cuda.setDevice(self.gpu_device)
                self.cuda_available = True
                
                # Create CUDA accelerated objects
                self.cuda_stream = cv.cuda_Stream()
                
                # Print GPU info
                gpu_info = cv.cuda.DeviceInfo(self.gpu_device)
                logging.info(f"OpenCV CUDA enabled on GPU{self.gpu_device}: {gpu_info.name()}")
                logging.info(f"  Compute capability: {gpu_info.majorVersion()}.{gpu_info.minorVersion()}")
                logging.info(f"  Total memory: {gpu_info.totalMemory() / 1024**3:.1f} GB")
                
            else:
                logging.info("OpenCV not built with CUDA support, using CPU")
        except Exception as e:
            logging.info(f"OpenCV CUDA not available: {e}")
    
    def process_frame_gpu(self, frame):
        """Process frame using GPU acceleration"""
        
        if self.cuda_available:
            # Upload frame to GPU
            gpu_frame = cv.cuda_GpuMat()
            gpu_frame.upload(frame, stream=self.cuda_stream)
            
            # GPU-accelerated preprocessing
            gpu_frame_rgb = cv.cuda.cvtColor(gpu_frame, cv.COLOR_BGR2RGB, stream=self.cuda_stream)
            
            # Download for MediaPipe (unless using GPU MediaPipe build)
            rgb_frame = gpu_frame_rgb.download(stream=self.cuda_stream)
            self.cuda_stream.waitForCompletion()
        else:
            # CPU fallback
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
        # Process with MediaPipe (uses GPU if available)
        results = self.face_mesh.process(rgb_frame)
        
        return results
    
    def benchmark_performance(self):
        """Benchmark CPU vs GPU performance"""
        import time
        
        # Create test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Warmup
        for _ in range(10):
            _ = self.process_frame_gpu(test_frame)
        
        # Benchmark
        iterations = 100
        start = time.perf_counter()
        
        for _ in range(iterations):
            _ = self.process_frame_gpu(test_frame)
        
        elapsed = time.perf_counter() - start
        fps = iterations / elapsed
        ms_per_frame = (elapsed / iterations) * 1000
        
        device = f"GPU{self.gpu_device}" if self.use_gpu else "CPU"
        print(f"\nPerformance on {device}:")
        print(f"  Average FPS: {fps:.1f}")
        print(f"  MS per frame: {ms_per_frame:.2f}")
        
        return fps, ms_per_frame

# ============================
# PYTORCH GPU ACCELERATION
# ============================

class TorchAcceleratedDMS:
    """Alternative: Using PyTorch for GPU acceleration"""
    
    def __init__(self, gpu_device=2):
        import torch
        import torchvision
        
        self.device = torch.device(f'cuda:{gpu_device}')
        
        # Load pre-trained face detection model on GPU
        self.face_detector = torchvision.models.detection.retinanet_resnet50_fpn(
            pretrained=True
        ).to(self.device).eval()
        
        # Custom eye detection model (example)
        self.eye_model = self._load_custom_model().to(self.device)
        
        logging.info(f"Torch models loaded on GPU{gpu_device}")
    
    def _load_custom_model(self):
        """Load custom trained eye tracking model"""
        import torch.nn as nn
        
        # Example lightweight CNN for eye state classification
        class EyeStateNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1)
                )
                self.classifier = nn.Sequential(
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(64, 2)  # Open/Closed
                )
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
        
        return EyeStateNet()
    
    def process_frame_torch(self, frame):
        """Process frame using PyTorch on GPU"""
        import torch
        import torchvision.transforms as T
        
        # Convert to tensor and move to GPU
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Frame to tensor
        frame_tensor = transform(frame).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Run inference on GPU
            predictions = self.face_detector(frame_tensor)
        
        return predictions

# ============================
# USAGE EXAMPLE
# ============================

def check_gpu_availability():
    """Check what GPU acceleration is available"""
    print("\n" + "="*50)
    print("GPU AVAILABILITY CHECK")
    print("="*50)
    
    # Check CUDA devices
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n✓ PyTorch CUDA available")
            print(f"  GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU{i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
                print(f"    Compute: {props.major}.{props.minor}")
        else:
            print("\n✗ PyTorch CUDA not available")
    except ImportError:
        print("\n✗ PyTorch not installed")
    
    # Check TensorFlow GPU
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"\n✓ TensorFlow GPU available")
            for gpu in gpus:
                print(f"  {gpu.name}")
        else:
            print("\n✗ TensorFlow GPU not available")
    except ImportError:
        print("\n✗ TensorFlow not installed")
    
    # Check OpenCV CUDA
    try:
        if cv.cuda.getCudaEnabledDeviceCount() > 0:
            print(f"\n✓ OpenCV CUDA available")
            print(f"  Device count: {cv.cuda.getCudaEnabledDeviceCount()}")
        else:
            print("\n✗ OpenCV CUDA not available")
    except:
        print("\n✗ OpenCV CUDA not available")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    # Check GPU availability
    check_gpu_availability()
    
    # Initialize DMS on GPU2
    print("\nInitializing DMS on GPU2...")
    dms = GPUAcceleratedDMS(gpu_device=gpu_id, use_gpu=True)
    
    # Benchmark
    print("\nRunning benchmark...")
    fps_gpu, ms_gpu = dms.benchmark_performance()
    
    # Compare with CPU
    print("\nInitializing DMS on CPU for comparison...")
    dms_cpu = GPUAcceleratedDMS(gpu_device=0, use_gpu=False)
    fps_cpu, ms_cpu = dms_cpu.benchmark_performance()
    
    # Results
    print("\n" + "="*50)
    print("PERFORMANCE COMPARISON")
    print("="*50)
    print(f"GPU2 Performance: {fps_gpu:.1f} FPS ({ms_gpu:.2f} ms/frame)")
    print(f"CPU Performance:  {fps_cpu:.1f} FPS ({ms_cpu:.2f} ms/frame)")
    print(f"Speedup: {fps_gpu/fps_cpu:.2f}x")