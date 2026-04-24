"""
Training Utilities — Chest X-ray Benchmark
============================================
Seed setting, device detection, parameter counting, and DataLoader creation.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for full reproducibility.

    Sets seeds for Python, NumPy, PyTorch (CPU and CUDA).

    Parameters
    ----------
    seed : int
        Random seed value. Default 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"  Random seed set to {seed} (Python, NumPy, PyTorch, CUDA)")


def get_device() -> torch.device:
    """Detect and return the best available compute device.

    Priority: CUDA → MPS → CPU.

    Returns
    -------
    torch.device
        Best available device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"  Device: CUDA ({gpu_name}, {gpu_mem:.1f} GB)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"  Device: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print(f"  Device: CPU")

    return device


def count_parameters(model: torch.nn.Module) -> dict:
    """Count model parameters.

    Parameters
    ----------
    model : nn.Module
        PyTorch model.

    Returns
    -------
    dict
        Dictionary with total, trainable, and frozen parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    info = {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
    }

    print(f"\n  Parameter Count:")
    print(f"    {'Total':15s}: {total:>12,}")
    print(f"    {'Trainable':15s}: {trainable:>12,}")
    print(f"    {'Frozen':15s}: {frozen:>12,}")

    return info


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, test DataLoaders from DataFrames.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training DataFrame.
    val_df : pd.DataFrame
        Validation DataFrame.
    test_df : pd.DataFrame
        Test DataFrame.
    batch_size : int
        Batch size. Default 32.
    num_workers : int
        DataLoader workers. Default 4.
    img_size : int
        Image size. Default 224.

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader]
        (train_loader, val_loader, test_loader)
    """
    from src.data.preprocessing import ChestXrayDataset

    train_ds = ChestXrayDataset(train_df, split="train", img_size=img_size)
    val_ds = ChestXrayDataset(val_df, split="val", img_size=img_size)
    test_ds = ChestXrayDataset(test_df, split="test", img_size=img_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"\n  DataLoaders created:")
    print(f"    Train: {len(train_ds):,} samples, {len(train_loader)} batches")
    print(f"    Val:   {len(val_ds):,} samples, {len(val_loader)} batches")
    print(f"    Test:  {len(test_ds):,} samples, {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


def compute_class_weights_from_loader(
    loader: DataLoader,
    n_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute inverse-frequency class weights from a DataLoader.

    Parameters
    ----------
    loader : DataLoader
        Data loader to iterate.
    n_classes : int
        Number of classes.
    device : torch.device
        Target device.

    Returns
    -------
    torch.Tensor
        Class weights, shape (n_classes,).
    """
    counts = torch.zeros(n_classes, dtype=torch.float64)

    for _, labels in loader:
        for label in labels:
            counts[label.item()] += 1

    total = counts.sum()
    weights = total / (n_classes * (counts + 1e-6))
    weights = weights.float().to(device)

    print(f"\n  Class weights computed from loader:")
    for i, (c, w) in enumerate(zip(counts.numpy(), weights.cpu().numpy())):
        print(f"    Class {i}: count={int(c):>7,}  weight={w:.3f}")

    return weights
