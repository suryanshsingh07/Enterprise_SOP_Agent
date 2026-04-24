"""
Training Configuration — Chest X-ray Benchmark
================================================
Centralised dataclass holding all training hyperparameters.
Every field is documented with its default, valid range, and effect.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class TrainingConfig:
    """Master training configuration for all deep-learning experiments.

    Attributes are grouped by category and documented with:
    - Default value
    - Valid range
    - Effect on training
    """

    # ── General ───────────────────────────────────────────────────────────────
    seed: int = 42
    """Random seed for full reproducibility. Range: any int. Effect: deterministic runs."""

    device: str = "cuda"
    """Compute device. Options: 'cuda', 'mps', 'cpu'. Effect: GPU accelerates training ~10×."""

    num_workers: int = 4
    """DataLoader worker processes. Range: [0, 16]. Effect: higher = faster data loading."""

    # ── Image ─────────────────────────────────────────────────────────────────
    img_size: int = 224
    """Input image resolution (square). Range: [128, 512]. Effect: larger = more detail, slower."""

    batch_size: int = 32
    """Training batch size. Range: [8, 128]. Effect: larger = stabler gradients, more VRAM."""

    # ── Training Phases ───────────────────────────────────────────────────────
    epochs_phase1: int = 10
    """Phase 1 epochs (frozen backbone). Range: [5, 20]. Effect: trains classification head."""

    epochs_phase2: int = 40
    """Phase 2 epochs (unfrozen top blocks). Range: [20, 100]. Effect: fine-tunes features."""

    lr_phase1: float = 1e-3
    """Phase 1 learning rate. Range: [1e-4, 1e-2]. Effect: head convergence speed."""

    lr_phase2: float = 1e-4
    """Phase 2 learning rate. Range: [1e-5, 1e-3]. Effect: fine-tuning granularity."""

    weight_decay: float = 1e-4
    """AdamW weight decay (L2 reg). Range: [1e-5, 1e-2]. Effect: prevents overfitting."""

    # ── Loss ──────────────────────────────────────────────────────────────────
    focal_gamma: float = 2.0
    """Focal loss gamma (focusing parameter). Range: [0.5, 5.0]. Effect: harder example focus."""

    use_focal_loss: bool = True
    """Whether to use Focal Loss. If False, uses CrossEntropy. Effect: class imbalance handling."""

    # ── Scheduler ─────────────────────────────────────────────────────────────
    scheduler: str = "onecycle"
    """LR scheduler type. Options: 'onecycle', 'cosine', 'step'. Effect: learning rate trajectory."""

    pct_start: float = 0.1
    """Fraction of training for warmup (OneCycleLR). Range: [0.05, 0.3]. Effect: warm start."""

    # ── Early Stopping ────────────────────────────────────────────────────────
    early_stopping_patience: int = 10
    """Epochs without improvement before stopping. Range: [5, 20]. Effect: prevents waste."""

    early_stopping_min_delta: float = 0.001
    """Minimum metric improvement to count. Range: [0, 0.01]. Effect: sensitivity threshold."""

    # ── Augmentation ──────────────────────────────────────────────────────────
    flip_p: float = 0.5
    """Horizontal flip probability. Range: [0, 1]. Effect: spatial invariance."""

    rotation_deg: int = 10
    """Max rotation degrees (±). Range: [0, 30]. Effect: rotational invariance."""

    translate: float = 0.05
    """Max affine translation fraction. Range: [0, 0.2]. Effect: positional robustness."""

    erase_p: float = 0.2
    """Random erasing probability. Range: [0, 0.5]. Effect: occlusion robustness."""

    color_jitter_brightness: float = 0.2
    """Brightness jitter factor. Range: [0, 0.5]. Effect: lighting invariance."""

    color_jitter_contrast: float = 0.2
    """Contrast jitter factor. Range: [0, 0.5]. Effect: contrast invariance."""

    # ── Model-Specific ────────────────────────────────────────────────────────
    dropout: float = 0.4
    """Dropout rate in classification head. Range: [0.1, 0.6]. Effect: regularisation."""

    n_classes: int = 6
    """Number of output classes. Fixed at 6 for this benchmark."""

    unfreeze_blocks: int = 3
    """Number of top blocks to unfreeze in Phase 2. Range: [1, 5]. Effect: fine-tuning depth."""

    # ── Paths ─────────────────────────────────────────────────────────────────
    checkpoint_dir: str = "outputs/checkpoints"
    """Directory to save model checkpoints."""

    log_dir: str = "outputs/logs"
    """Directory to save training logs."""

    experiment_dir: str = "experiments"
    """Directory to save experiment results."""

    # ── Mixed Precision ───────────────────────────────────────────────────────
    use_amp: bool = True
    """Use automatic mixed precision (FP16). Effect: ~2× speedup, lower VRAM."""

    # ── CLAHE Preprocessing ───────────────────────────────────────────────────
    clahe_clip: float = 2.0
    """CLAHE clip limit. Range: [1.0, 4.0]. Effect: local contrast enhancement strength."""

    clahe_tile: Tuple[int, int] = (8, 8)
    """CLAHE tile grid size. Options: (4,4), (8,8), (16,16). Effect: adaptation granularity."""

    # ── ImageNet Normalization ────────────────────────────────────────────────
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    """ImageNet channel means for normalization."""

    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    """ImageNet channel stds for normalization."""


# ── Convenience Instance ──────────────────────────────────────────────────────
DEFAULT_CONFIG = TrainingConfig()


if __name__ == "__main__":
    cfg = TrainingConfig()
    print("=== Training Configuration ===")
    for fld in cfg.__dataclass_fields__:
        val = getattr(cfg, fld)
        print(f"  {fld:35s} = {val}")
