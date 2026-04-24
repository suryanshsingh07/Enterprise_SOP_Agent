#!/usr/bin/env python3
"""
10 — Generate Explanations
============================
Generate Grad-CAM for CNNs and attention maps for Swin-T.

Usage:
    python scripts/10_generate_explanations.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_builder import CLASS_NAMES, NUM_CLASSES
from src.data.preprocessing import ChestXrayDataset
from src.explainability.gradcam import GradCAM, batch_gradcam, get_target_layer
from src.explainability.attention_maps import (
    SwinAttentionExtractor,
    compare_explanations,
)
from src.explainability.xai_utils import compute_attention_coverage
from src.training.utils import set_seed, get_device


def main() -> None:
    """Generate explainability visualisations."""
    print("=" * 60)
    print("  CHEST X-RAY BENCHMARK — EXPLAINABILITY")
    print("=" * 60)

    set_seed(42)
    device = get_device()

    checkpoint_dir = Path("outputs/checkpoints")
    output_dir = Path("outputs/explainability")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests_dir = Path("data/manifests")
    if not (manifests_dir / "test.csv").exists():
        print("\n  ⚠ Run 01_build_dataset.py first.")
        return

    test_df = pd.read_csv(manifests_dir / "test.csv")
    test_ds = ChestXrayDataset(test_df, split="test", img_size=224)

    # ── CNN Grad-CAM ──────────────────────────────────────────────────────────
    cnn_ckpt = checkpoint_dir / "efficientnet_b3_best.pth"
    if cnn_ckpt.exists():
        print("\n  Generating Grad-CAM for EfficientNet-B3...")
        from src.models.cnn_models import EfficientNetChest

        cnn_model = EfficientNetChest(n_classes=NUM_CLASSES, freeze_backbone=False)
        ckpt = torch.load(cnn_ckpt, map_location=device, weights_only=False)
        cnn_model.load_state_dict(ckpt["model_state_dict"])
        cnn_model.to(device)
        cnn_model.eval()

        target_layer = get_target_layer(cnn_model, "efficientnet")
        results = batch_gradcam(
            cnn_model, test_ds, CLASS_NAMES,
            save_dir=output_dir,
            target_layer_name=target_layer,
            n_samples=10,
        )
        print(f"  Generated {len(results)} Grad-CAM images.")
    else:
        print("  ⚠ EfficientNet checkpoint not found, skipping Grad-CAM.")

    # ── Swin Attention Maps ───────────────────────────────────────────────────
    swin_ckpt = checkpoint_dir / "swin_tiny_best.pth"
    if swin_ckpt.exists():
        print("\n  Generating attention maps for Swin-Tiny...")
        from src.models.transformer_models import SwinChest

        swin_model = SwinChest(n_classes=NUM_CLASSES, freeze_backbone=False)
        ckpt = torch.load(swin_ckpt, map_location=device, weights_only=False)
        swin_model.load_state_dict(ckpt["model_state_dict"])
        swin_model.to(device)
        swin_model.eval()

        extractor = SwinAttentionExtractor(swin_model)

        # Generate for sample images
        coverages = {name: [] for name in CLASS_NAMES}

        for class_id, class_name in enumerate(CLASS_NAMES):
            class_indices = test_df[test_df["label_id"] == class_id].index.tolist()
            selected = class_indices[:10]

            for idx in selected:
                img_tensor, _ = test_ds[idx]
                attn_maps = extractor.extract(img_tensor)

                if attn_maps:
                    coverage = compute_attention_coverage(attn_maps[-1])
                    coverages[class_name].append(coverage)

                    img_path = test_df.iloc[idx]["image_path"]
                    save_path = (
                        output_dir / "attention" / class_name /
                        f"attn_{Path(img_path).stem}.png"
                    )
                    extractor.visualise_stage(
                        img_path, attn_maps[-1],
                        stage_idx=len(attn_maps) - 1,
                        save_path=save_path,
                    )

        extractor.remove_hooks()

        # Print coverage summary
        print("\n  Mean Attention Coverage per Class:")
        for name, covs in coverages.items():
            if covs:
                print(f"    {name:15s}: {np.mean(covs):.3f} ± {np.std(covs):.3f}")
    else:
        print("  ⚠ Swin-Tiny checkpoint not found, skipping attention maps.")

    # ── Comparison Figures ────────────────────────────────────────────────────
    if cnn_ckpt.exists() and swin_ckpt.exists():
        print("\n  Generating comparison figures...")
        sample_indices = test_df.sample(min(10, len(test_df)), random_state=42).index

        for i, idx in enumerate(sample_indices):
            img_path = test_df.iloc[idx]["image_path"]
            compare_explanations(
                img_path=img_path,
                cnn_model=cnn_model,
                vit_model=swin_model,
                class_names=CLASS_NAMES,
                cnn_target_layer=get_target_layer(cnn_model, "efficientnet"),
                save_path=output_dir / f"comparison_{i}.png",
            )

    print("\n" + "=" * 60)
    print("  ✓ Explainability generation complete!")
    print(f"  Saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
