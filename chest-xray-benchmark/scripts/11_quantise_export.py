#!/usr/bin/env python3
"""
11 — Quantise & Export
=======================
Quantise models to INT8, export to ONNX, and benchmark edge inference.

Usage:
    python scripts/11_quantise_export.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.deployment.quantise import (
    benchmark_edge_inference,
    export_to_onnx,
)
from src.training.utils import set_seed, get_device


def main() -> None:
    """Quantise and export all models."""
    print("=" * 60)
    print("  CHEST X-RAY BENCHMARK — QUANTISE & EXPORT")
    print("=" * 60)

    set_seed(42)

    checkpoint_dir = Path("outputs/checkpoints")
    onnx_dir = Path("outputs/onnx")
    onnx_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Models to export
    model_configs = {
        "EfficientNet-B3": (
            "efficientnet_b3_best.pth",
            "src.models.cnn_models",
            "EfficientNetChest",
        ),
        "ResNet-50": (
            "resnet50_best.pth",
            "src.models.cnn_models",
            "ResNet50Chest",
        ),
        "Swin-Tiny": (
            "swin_tiny_best.pth",
            "src.models.transformer_models",
            "SwinChest",
        ),
    }

    for model_name, (ckpt_file, module_path, class_name) in model_configs.items():
        ckpt_path = checkpoint_dir / ckpt_file
        if not ckpt_path.exists():
            print(f"\n  ⚠ Skipping {model_name}: checkpoint not found")
            continue

        print(f"\n{'='*60}")
        print(f"  Processing: {model_name}")
        print(f"{'='*60}")

        # Load model
        import importlib
        mod = importlib.import_module(module_path)
        model_cls = getattr(mod, class_name)
        model = model_cls(n_classes=6, freeze_backbone=False)

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Export to ONNX
        onnx_path = onnx_dir / f"{model_name.lower().replace('/', '_').replace('-', '_')}.onnx"
        try:
            export_to_onnx(model, onnx_path, img_size=224)

            # Benchmark edge inference
            bench = benchmark_edge_inference(onnx_path, n_runs=50)
            results[model_name] = {
                "onnx_path": str(onnx_path),
                **bench,
            }
        except Exception as e:
            print(f"  ✗ Export failed for {model_name}: {e}")
            results[model_name] = {"error": str(e)}

    # Save results
    if results:
        experiments_dir = Path("experiments")
        experiments_dir.mkdir(parents=True, exist_ok=True)
        with open(experiments_dir / "export_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print("\n" + "=" * 60)
        print("  EXPORT & BENCHMARK RESULTS")
        print("=" * 60)
        for name, res in results.items():
            if "error" in res:
                print(f"  {name}: FAILED — {res['error']}")
            else:
                print(f"  {name}: {res['mean_ms']:.1f}ms, {res['throughput_ips']:.1f} img/s")

    print("\n" + "=" * 60)
    print("  ✓ Quantise & export complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
