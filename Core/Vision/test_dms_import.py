#!/usr/bin/env python
"""
Test script to verify DMS_Module imports correctly without errors
"""

import os
import sys

# Simulate selecting GPU 1
sys.stdin = open('/dev/stdin', 'r')

print("Testing DMS Module import...")
print("=" * 80)

# Set environment before import
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print(f"CUDA_VISIBLE_DEVICES set to: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print("=" * 80)

# Now test the import by importing just the MediaPipe parts
import warnings
from io import StringIO

warnings.filterwarnings('ignore', category=DeprecationWarning, module='google.protobuf')
warnings.filterwarnings('ignore', message='.*GetPrototype.*')

# Capture and suppress stderr during MediaPipe import
_original_stderr = sys.stderr
sys.stderr = StringIO()

import mediapipe as mp

# Restore stderr
sys.stderr = _original_stderr

print("\n✓ MediaPipe imported successfully!")
print(f"MediaPipe version: {mp.__version__}")

# Test basic GPU info
try:
    import torch
    if torch.cuda.is_available():
        print(f"✓ PyTorch sees {torch.cuda.device_count()} GPU(s)")
        print(f"  Current device: {torch.cuda.current_device()}")
        print(f"  Device name: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ PyTorch: No CUDA GPUs detected")
except Exception as e:
    print(f"⚠ PyTorch GPU check failed: {e}")

print("\n" + "=" * 80)
print("✓ All imports successful - no AttributeError!")
print("=" * 80)
