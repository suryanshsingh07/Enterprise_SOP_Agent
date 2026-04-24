"""
Quantisation & Export — Chest X-ray Benchmark
================================================
Post-training INT8 quantisation, ONNX export, and edge inference benchmarking.

Functions:
    quantise_to_int8 — static INT8 quantisation with calibration
    export_to_onnx — export to ONNX with dynamic batch
    benchmark_edge_inference — simulate Raspberry Pi CPU inference
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# =============================================================================
# INT8 Quantisation
# =============================================================================
def quantise_to_int8(
    model: nn.Module,
    calibration_loader: DataLoader,
    save_path: str | Path,
    n_calibration: int = 100,
) -> nn.Module:
    """Post-training static INT8 quantisation.

    Uses PyTorch's quantisation framework:
    1. Prepare model for static quantisation
    2. Calibrate on sample images
    3. Convert to quantised model
    4. Save and report size comparison

    Parameters
    ----------
    model : nn.Module
        Trained model (must be on CPU for quantisation).
    calibration_loader : DataLoader
        DataLoader for calibration (use validation or train subset).
    save_path : str | Path
        Path to save the quantised model.
    n_calibration : int
        Number of calibration images. Default 100.

    Returns
    -------
    nn.Module
        Quantised model.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.cpu()
    model.eval()

    # Save original model size
    original_path = save_path.with_suffix(".original.pth")
    torch.save(model.state_dict(), original_path)
    original_size_mb = os.path.getsize(original_path) / (1024 * 1024)

    # Prepare model for quantisation
    model.qconfig = torch.quantization.get_default_qconfig("x86")
    model_prepared = torch.quantization.prepare(model, inplace=False)

    # Calibration
    print(f"\n  Calibrating with {n_calibration} images...")
    count = 0
    with torch.no_grad():
        for images, _ in calibration_loader:
            images = images.cpu()
            model_prepared(images)
            count += images.shape[0]
            if count >= n_calibration:
                break

    # Convert to quantised
    model_quantised = torch.quantization.convert(model_prepared, inplace=False)

    # Save quantised model
    torch.save(model_quantised.state_dict(), save_path)
    quantised_size_mb = os.path.getsize(save_path) / (1024 * 1024)

    # Report
    compression = original_size_mb / max(quantised_size_mb, 1e-6)
    print(f"\n  Quantisation Results:")
    print(f"    Original size:  {original_size_mb:.1f} MB")
    print(f"    Quantised size: {quantised_size_mb:.1f} MB")
    print(f"    Compression:    {compression:.1f}×")

    # Speed comparison
    dummy = torch.randn(1, 3, 224, 224)

    # Original speed
    with torch.no_grad():
        t0 = time.time()
        for _ in range(50):
            model(dummy)
        original_ms = (time.time() - t0) / 50 * 1000

    # Quantised speed
    with torch.no_grad():
        t0 = time.time()
        for _ in range(50):
            model_quantised(dummy)
        quantised_ms = (time.time() - t0) / 50 * 1000

    speedup = original_ms / max(quantised_ms, 1e-6)
    print(f"\n  Speed Comparison:")
    print(f"    Original:  {original_ms:.1f} ms/image")
    print(f"    Quantised: {quantised_ms:.1f} ms/image")
    print(f"    Speedup:   {speedup:.1f}×")

    # Cleanup temporary file
    if original_path.exists():
        original_path.unlink()

    return model_quantised


# =============================================================================
# ONNX Export
# =============================================================================
def export_to_onnx(
    model: nn.Module,
    save_path: str | Path,
    img_size: int = 224,
    opset_version: int = 17,
) -> None:
    """Export model to ONNX format with dynamic batch axis.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    save_path : str | Path
        Output ONNX file path.
    img_size : int
        Input image size. Default 224.
    opset_version : int
        ONNX opset version. Default 17.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.cpu()
    model.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size)

    print(f"\n  Exporting to ONNX: {save_path}")

    torch.onnx.export(
        model,
        dummy_input,
        str(save_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    # Verify export
    import onnx
    onnx_model = onnx.load(str(save_path))
    onnx.checker.check_model(onnx_model)

    # Print info
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    graph = onnx_model.graph

    print(f"  ✓ ONNX export verified!")
    print(f"    File size: {file_size_mb:.1f} MB")
    print(f"    Input:  {[inp.name for inp in graph.input]}")
    print(f"    Output: {[out.name for out in graph.output]}")

    # Print shapes
    for inp in graph.input:
        shape = [d.dim_value if d.dim_value else "dynamic"
                 for d in inp.type.tensor_type.shape.dim]
        print(f"    Input shape:  {shape}")
    for out in graph.output:
        shape = [d.dim_value if d.dim_value else "dynamic"
                 for d in out.type.tensor_type.shape.dim]
        print(f"    Output shape: {shape}")


# =============================================================================
# Edge Inference Benchmarking
# =============================================================================
def benchmark_edge_inference(
    onnx_path: str | Path,
    n_runs: int = 100,
    img_size: int = 224,
) -> dict:
    """Simulate edge (Raspberry Pi) inference using ONNX Runtime on CPU.

    Parameters
    ----------
    onnx_path : str | Path
        Path to ONNX model file.
    n_runs : int
        Number of inference runs. Default 100.
    img_size : int
        Input image size. Default 224.

    Returns
    -------
    dict
        Benchmark results: mean, std, min, max latency (ms), throughput (img/s).
    """
    import onnxruntime as ort

    onnx_path = str(onnx_path)

    # Force CPU-only execution (simulate edge device)
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1  # Single-threaded like RPi
    sess_options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        onnx_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(1, 3, img_size, img_size).astype(np.float32)

    # Warmup
    for _ in range(5):
        session.run(None, {input_name: dummy})

    # Benchmark
    latencies = []
    for _ in range(n_runs):
        t0 = time.time()
        session.run(None, {input_name: dummy})
        latencies.append((time.time() - t0) * 1000)  # ms

    latencies = np.array(latencies)
    throughput = 1000.0 / latencies.mean()  # images per second

    results = {
        "mean_ms": round(float(latencies.mean()), 2),
        "std_ms": round(float(latencies.std()), 2),
        "min_ms": round(float(latencies.min()), 2),
        "max_ms": round(float(latencies.max()), 2),
        "throughput_ips": round(throughput, 1),
        "n_runs": n_runs,
    }

    print(f"\n  Edge Inference Benchmark (CPU, single-thread):")
    print(f"    Model:      {onnx_path}")
    print(f"    Runs:       {n_runs}")
    print(f"    Mean:       {results['mean_ms']:.2f} ms")
    print(f"    Std:        {results['std_ms']:.2f} ms")
    print(f"    Min:        {results['min_ms']:.2f} ms")
    print(f"    Max:        {results['max_ms']:.2f} ms")
    print(f"    Throughput: {results['throughput_ips']:.1f} images/sec")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("  Quantise & Export — Demo")
    print("=" * 60)

    from src.models.cnn_models import EfficientNetChest

    model = EfficientNetChest(n_classes=6, freeze_backbone=False)
    dummy = torch.randn(1, 3, 224, 224)

    # ONNX export
    export_to_onnx(model, "outputs/onnx/efficientnet_b3.onnx")

    # Edge benchmark
    benchmark_edge_inference("outputs/onnx/efficientnet_b3.onnx", n_runs=20)

    print("\n  ✓ Quantise & export demo complete.")
