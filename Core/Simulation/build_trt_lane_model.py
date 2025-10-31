#!/usr/bin/env python3
"""
Build TensorRT Lane Segmentation Model
========================================

This script converts a trained lane segmentation model (ONNX or TensorFlow)
to TensorRT format optimized for Jetson Orin or desktop GPU.

Usage:
    python build_trt_lane_model.py --input model.onnx --output lane_seg.trt --fp16

Supported input formats:
    - ONNX (.onnx)
    - TensorFlow SavedModel (directory)
    - Keras (.h5)
"""

import argparse
import os
import sys
import tensorrt as trt
import numpy as np

# TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def build_engine_from_onnx(onnx_path, engine_path, fp16=False, int8=False, max_batch_size=1):
    """
    Build TensorRT engine from ONNX model.

    Args:
        onnx_path: Path to ONNX model
        engine_path: Output path for TensorRT engine
        fp16: Use FP16 precision (recommended for Jetson Orin)
        int8: Use INT8 precision (requires calibration)
        max_batch_size: Maximum batch size
    """
    print(f"Building TensorRT engine from ONNX: {onnx_path}")
    print(f"FP16: {fp16}, INT8: {int8}, Batch size: {max_batch_size}")

    # Create builder and network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX model
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse ONNX file")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False

    print(f"✓ Parsed ONNX model: {network.num_layers} layers")

    # Create builder config
    config = builder.create_builder_config()

    # Set memory pool limit (in bytes)
    # Allocate 4GB workspace for optimization
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    # Enable FP16 mode if requested
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("✓ FP16 mode enabled")

    # Enable INT8 mode if requested
    if int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        # Note: INT8 requires calibration data - implement calibrator if needed
        print("✓ INT8 mode enabled (requires calibration)")

    # Build engine
    print("Building TensorRT engine (this may take several minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("ERROR: Failed to build engine")
        return False

    # Save engine to file
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    print(f"✓ TensorRT engine saved to: {engine_path}")
    print(f"  File size: {os.path.getsize(engine_path) / (1024*1024):.2f} MB")

    return True


def build_engine_from_tf(saved_model_path, engine_path, fp16=False, input_shape=(720, 1280, 3)):
    """
    Build TensorRT engine from TensorFlow SavedModel.

    Args:
        saved_model_path: Path to TensorFlow SavedModel directory
        engine_path: Output path for TensorRT engine
        fp16: Use FP16 precision
        input_shape: Model input shape (H, W, C)
    """
    try:
        import tensorflow as tf
        from tensorflow.python.compiler.tensorrt import trt_convert as trt_tf
    except ImportError:
        print("ERROR: TensorFlow not available. Install with: pip install tensorflow")
        return False

    print(f"Building TensorRT engine from TensorFlow SavedModel: {saved_model_path}")

    # Configure precision
    if fp16:
        precision_mode = trt_tf.TrtPrecisionMode.FP16
        print("Using FP16 precision")
    else:
        precision_mode = trt_tf.TrtPrecisionMode.FP32
        print("Using FP32 precision")

    # Create TRT converter
    conversion_params = trt_tf.DEFAULT_TRT_CONVERSION_PARAMS._replace(
        precision_mode=precision_mode,
        max_workspace_size_bytes=4 << 30,  # 4GB
        maximum_cached_engines=1
    )

    converter = trt_tf.TrtGraphConverterV2(
        input_saved_model_dir=saved_model_path,
        conversion_params=conversion_params
    )

    # Convert model
    print("Converting TensorFlow model to TensorRT...")
    converter.convert()

    # Build engine with sample input
    def input_fn():
        # Create dummy input for engine building
        batch_size = 1
        inp = np.random.normal(size=(batch_size,) + input_shape).astype(np.float32)
        yield [inp]

    print("Building optimized engine...")
    converter.build(input_fn=input_fn)

    # Save
    converter.save(engine_path)
    print(f"✓ TensorRT engine saved to: {engine_path}")

    return True


def create_simple_lane_model_onnx(output_path="simple_lane_model.onnx", input_shape=(720, 1280, 3)):
    """
    Create a simple dummy ONNX lane segmentation model for testing.
    This is just for demonstration - replace with your actual trained model.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.onnx
    except ImportError:
        print("ERROR: PyTorch not available for model creation")
        return False

    print("Creating simple dummy lane segmentation model...")

    class SimpleLaneNet(nn.Module):
        def __init__(self):
            super(SimpleLaneNet, self).__init__()
            # Simple encoder-decoder architecture
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 16, 2, stride=2),
                nn.ReLU(),
                nn.Conv2d(16, 1, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    model = SimpleLaneNet()
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1])

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"✓ Created dummy ONNX model: {output_path}")
    return True


def benchmark_engine(engine_path, input_shape=(720, 1280, 3), iterations=100):
    """
    Benchmark TensorRT engine performance.
    """
    import pycuda.driver as cuda
    import pycuda.autoinit
    import time

    print(f"\nBenchmarking TensorRT engine: {engine_path}")
    print(f"Iterations: {iterations}")

    # Load engine
    with open(engine_path, 'rb') as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # Allocate buffers
    input_size = trt.volume(input_shape)
    output_size = input_shape[0] * input_shape[1]  # Binary mask

    h_input = cuda.pagelocked_empty(input_size, dtype=np.float32)
    h_output = cuda.pagelocked_empty(output_size, dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    stream = cuda.Stream()
    bindings = [int(d_input), int(d_output)]

    # Warmup
    for _ in range(10):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.time()
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
        times.append((time.time() - start) * 1000)

    avg_time = np.mean(times)
    fps = 1000 / avg_time

    print("=" * 60)
    print(f"Average inference time: {avg_time:.2f} ms")
    print(f"Throughput: {fps:.1f} FPS")
    print(f"Min: {np.min(times):.2f} ms, Max: {np.max(times):.2f} ms")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Build TensorRT Lane Segmentation Model")
    parser.add_argument("--input", type=str, help="Input model path (ONNX or TF SavedModel)")
    parser.add_argument("--output", type=str, default="lane_seg.trt", help="Output TensorRT engine path")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 precision (recommended for Jetson)")
    parser.add_argument("--int8", action="store_true", help="Enable INT8 precision (requires calibration)")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark the built engine")
    parser.add_argument("--create-dummy", action="store_true", help="Create a dummy ONNX model for testing")
    parser.add_argument("--width", type=int, default=1280, help="Input width")
    parser.add_argument("--height", type=int, default=720, help="Input height")

    args = parser.parse_args()

    if args.create_dummy:
        dummy_path = "simple_lane_model.onnx"
        if create_simple_lane_model_onnx(dummy_path, (args.height, args.width, 3)):
            print(f"\nNow build TensorRT engine with:")
            print(f"  python {sys.argv[0]} --input {dummy_path} --output lane_seg.trt --fp16")
        return

    if not args.input:
        print("ERROR: --input required (or use --create-dummy to create test model)")
        parser.print_help()
        return

    if not os.path.exists(args.input):
        print(f"ERROR: Input model not found: {args.input}")
        return

    # Determine input format and build engine
    if args.input.endswith('.onnx'):
        success = build_engine_from_onnx(args.input, args.output, args.fp16, args.int8)
    elif os.path.isdir(args.input):
        success = build_engine_from_tf(args.input, args.output, args.fp16, (args.height, args.width, 3))
    else:
        print(f"ERROR: Unsupported input format: {args.input}")
        print("Supported formats: .onnx, TensorFlow SavedModel directory")
        return

    if not success:
        print("Failed to build TensorRT engine")
        return

    # Benchmark if requested
    if args.benchmark and os.path.exists(args.output):
        try:
            benchmark_engine(args.output, (args.height, args.width, 3))
        except Exception as e:
            print(f"Benchmark failed: {e}")


if __name__ == "__main__":
    main()
