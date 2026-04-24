"""
Transformer Models — Chest X-ray Benchmark
=============================================
Swin-Tiny, ViT-B/16, and Convolutional Autoencoder for chest X-ray
classification and anomaly detection.

Classes:
    SwinChest — Swin Transformer Tiny with attention map extraction
    ViTChest — Vision Transformer Base/16
    ChestAutoencoder — Convolutional autoencoder for anomaly detection
Functions:
    train_autoencoder — train autoencoder on Normal class only
    compute_anomaly_threshold — determine reconstruction error threshold
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# =============================================================================
# Swin Transformer Tiny
# =============================================================================
class SwinChest(nn.Module):
    """Swin Transformer Tiny for 6-class chest X-ray classification.

    Uses timm to load a pretrained Swin-Tiny model with custom head.
    Supports attention map extraction via forward hooks.

    Parameters
    ----------
    model_name : str
        timm model identifier. Default 'swin_tiny_patch4_window7_224'.
    n_classes : int
        Number of output classes. Default 6.
    freeze_backbone : bool
        Freeze backbone on init. Default True.
    """

    def __init__(
        self,
        model_name: str = "swin_tiny_patch4_window7_224",
        n_classes: int = 6,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        import timm

        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=n_classes,
        )
        self.n_classes = n_classes

        # Freeze backbone if requested
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if "head" not in name:
                    param.requires_grad = False

        # Storage for attention hooks
        self._attention_maps: List[torch.Tensor] = []
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input images, shape (B, 3, 224, 224).

        Returns
        -------
        torch.Tensor
            Logits, shape (B, n_classes).
        """
        return self.model(x)

    def unfreeze_last_n_stages(self, n: int = 2) -> None:
        """Unfreeze the last n of 4 Swin stages plus the head.

        Swin-Tiny has 4 stages (layers.0 through layers.3).

        Parameters
        ----------
        n : int
            Number of stages to unfreeze from the end. Default 2.
        """
        # Always unfreeze head
        for param in self.model.head.parameters():
            param.requires_grad = True

        # Unfreeze last n stages
        if hasattr(self.model, "layers"):
            total_stages = len(self.model.layers)
            for i in range(max(0, total_stages - n), total_stages):
                for param in self.model.layers[i].parameters():
                    param.requires_grad = True
            logger.info(f"  Unfroze last {n} Swin stages + head")
        else:
            # timm may use different attribute names
            logger.warning("  Could not find 'layers' attribute, unfreezing all")
            for param in self.model.parameters():
                param.requires_grad = True

    def get_attention_maps(self, x: torch.Tensor) -> List[np.ndarray]:
        """Extract attention maps from all window attention layers.

        Registers temporary forward hooks, runs a forward pass,
        collects attention weights, removes hooks.

        Parameters
        ----------
        x : torch.Tensor
            Input images, shape (B, 3, 224, 224).

        Returns
        -------
        list[np.ndarray]
            List of attention maps, each averaged across heads,
            normalised to [0, 1].
        """
        self._attention_maps = []
        self._hooks = []

        # Register hooks on all WindowAttention modules
        for name, module in self.model.named_modules():
            if "attn" in name.lower() and hasattr(module, "softmax"):
                hook = module.register_forward_hook(self._attn_hook)
                self._hooks.append(hook)

        # Forward pass
        self.model.eval()
        with torch.no_grad():
            _ = self.model(x)

        # Remove hooks
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

        # Process attention maps
        processed = []
        for attn in self._attention_maps:
            # attn shape: (B*num_windows, num_heads, window_size^2, window_size^2)
            # Average across heads
            attn_avg = attn.mean(dim=1)  # (B*nW, ws^2, ws^2)
            # Average across the source tokens
            attn_map = attn_avg.mean(dim=-1)  # (B*nW, ws^2)
            # Normalise to [0, 1]
            attn_np = attn_map.cpu().numpy()
            attn_np = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-8)
            processed.append(attn_np)

        return processed

    def _attn_hook(
        self,
        module: nn.Module,
        input: Tuple,
        output: torch.Tensor,
    ) -> None:
        """Forward hook to capture attention weights."""
        # For Swin, the attention is computed inside the forward method
        # We capture the output attention if available
        if hasattr(module, "attn_drop"):
            # Try to get the attention weights from the module's state
            if hasattr(module, "_attn_weights"):
                self._attention_maps.append(module._attn_weights.detach())


# =============================================================================
# Vision Transformer (ViT-B/16)
# =============================================================================
class ViTChest(nn.Module):
    """Vision Transformer Base/16 for 6-class chest X-ray classification.

    Parameters
    ----------
    model_name : str
        timm model identifier. Default 'vit_base_patch16_224'.
    n_classes : int
        Number of output classes. Default 6.
    freeze_backbone : bool
        Freeze backbone on init. Default True.
    drop_rate : float
        Dropout rate. Default 0.1.
    drop_path_rate : float
        Stochastic depth rate. Default 0.1.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        n_classes: int = 6,
        freeze_backbone: bool = True,
        drop_rate: float = 0.1,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        import timm

        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=n_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        self.n_classes = n_classes

        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if "head" not in name:
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input images, shape (B, 3, 224, 224).

        Returns
        -------
        torch.Tensor
            Logits, shape (B, n_classes).
        """
        return self.model(x)

    def unfreeze_last_n_blocks(self, n: int = 4) -> None:
        """Unfreeze the last n transformer blocks plus the head.

        ViT-B/16 has 12 blocks.

        Parameters
        ----------
        n : int
            Number of blocks to unfreeze. Default 4.
        """
        # Unfreeze head
        for param in self.model.head.parameters():
            param.requires_grad = True

        # Unfreeze last n blocks
        if hasattr(self.model, "blocks"):
            total_blocks = len(self.model.blocks)
            for i in range(max(0, total_blocks - n), total_blocks):
                for param in self.model.blocks[i].parameters():
                    param.requires_grad = True
            logger.info(f"  Unfroze last {n} ViT blocks + head")
        else:
            logger.warning("  Could not find 'blocks' attribute.")


# =============================================================================
# Convolutional Autoencoder
# =============================================================================
class ChestAutoencoder(nn.Module):
    """Convolutional Autoencoder for anomaly detection on chest X-rays.

    Architecture:
    - Input: (B, 1, 224, 224) grayscale
    - Encoder: Conv2d stack → 256-d latent
      224→112 (32ch) → 56 (64ch) → 28 (128ch) → 14 (256ch) → Flatten → Linear → 256-d
    - Decoder: Linear → Unflatten → ConvTranspose2d stack → (1, 224, 224)

    Training: Only on Normal class (label=0).
    Inference: High reconstruction error = anomaly.

    Parameters
    ----------
    latent_dim : int
        Latent vector dimension. Default 256.
    """

    def __init__(self, latent_dim: int = 256) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: (1, 224, 224) → (256, 14, 14) → 256-d
        self.encoder = nn.Sequential(
            # 224 → 112
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 112 → 56
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 56 → 28
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 28 → 14
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # Flatten and project to latent
        self.fc_encode = nn.Linear(256 * 14 * 14, latent_dim)

        # Decoder: 256-d → (256, 14, 14) → (1, 224, 224)
        self.fc_decode = nn.Linear(latent_dim, 256 * 14 * 14)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 14, 14)),
            # 14 → 28
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 28 → 56
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 56 → 112
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 112 → 224
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent vector.

        Parameters
        ----------
        x : torch.Tensor
            Input images, shape (B, 1, 224, 224).

        Returns
        -------
        torch.Tensor
            Latent vectors, shape (B, latent_dim).
        """
        h = self.encoder(x)
        h = h.flatten(1)
        z = self.fc_encode(h)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to image.

        Parameters
        ----------
        z : torch.Tensor
            Latent vectors, shape (B, latent_dim).

        Returns
        -------
        torch.Tensor
            Reconstructed images, shape (B, 1, 224, 224).
        """
        h = self.fc_decode(z)
        x_hat = self.decoder(h)
        return x_hat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode + decode.

        Parameters
        ----------
        x : torch.Tensor
            Input images, shape (B, 1, 224, 224).

        Returns
        -------
        torch.Tensor
            Reconstructed images, shape (B, 1, 224, 224).
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-image mean squared reconstruction error.

        Parameters
        ----------
        x : torch.Tensor
            Input images, shape (B, 1, 224, 224).

        Returns
        -------
        torch.Tensor
            Per-image MSE, shape (B,).
        """
        x_hat = self.forward(x)
        mse = F.mse_loss(x_hat, x, reduction="none")
        return mse.mean(dim=[1, 2, 3])  # Average over C, H, W


# =============================================================================
# Autoencoder Training
# =============================================================================

def train_autoencoder(
    model: ChestAutoencoder,
    normal_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str | torch.device = "cuda",
) -> Tuple[ChestAutoencoder, List[float]]:
    """Train the autoencoder on Normal (label=0) images only.

    Parameters
    ----------
    model : ChestAutoencoder
        Autoencoder model.
    normal_loader : DataLoader
        DataLoader containing only Normal class images.
    epochs : int
        Number of training epochs. Default 50.
    lr : float
        Learning rate. Default 1e-3.
    device : str | torch.device
        Compute device.

    Returns
    -------
    tuple[ChestAutoencoder, list[float]]
        (trained_model, per_epoch_loss_list)
    """
    device = torch.device(device) if isinstance(device, str) else device
    model = model.to(device)
    model.train()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epoch_losses: List[float] = []

    print(f"\n  Training Autoencoder: {epochs} epochs, device={device}")
    print(f"  {'Epoch':>5s} │ {'Loss':>10s}")
    print(f"  {'─' * 20}")

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        n_batches = 0

        for images, _ in normal_loader:
            # Convert to grayscale if needed (take first channel)
            if images.shape[1] == 3:
                images = images.mean(dim=1, keepdim=True)
            images = images.to(device)

            optimizer.zero_grad()
            reconstructed = model(images)
            loss = criterion(reconstructed, images)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / max(n_batches, 1)
        epoch_losses.append(avg_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  {epoch:5d} │ {avg_loss:10.6f}")

    print(f"\n  Training complete. Final loss: {epoch_losses[-1]:.6f}")
    return model, epoch_losses


def compute_anomaly_threshold(
    model: ChestAutoencoder,
    normal_val_loader: DataLoader,
    percentile: float = 95,
    device: str | torch.device = "cuda",
) -> float:
    """Compute anomaly threshold from Normal validation images.

    Parameters
    ----------
    model : ChestAutoencoder
        Trained autoencoder.
    normal_val_loader : DataLoader
        DataLoader with Normal validation images.
    percentile : float
        Percentile for threshold. Default 95.
    device : str | torch.device
        Compute device.

    Returns
    -------
    float
        Anomaly threshold (reconstruction error above this = anomaly).
    """
    device = torch.device(device) if isinstance(device, str) else device
    model = model.to(device)
    model.eval()

    all_errors: List[float] = []

    with torch.no_grad():
        for images, _ in normal_val_loader:
            if images.shape[1] == 3:
                images = images.mean(dim=1, keepdim=True)
            images = images.to(device)

            errors = model.reconstruction_error(images)
            all_errors.extend(errors.cpu().numpy().tolist())

    errors_array = np.array(all_errors)
    threshold = float(np.percentile(errors_array, percentile))

    print(f"\n  Anomaly Threshold Computation:")
    print(f"    Normal images: {len(all_errors)}")
    print(f"    Mean error:    {errors_array.mean():.6f}")
    print(f"    Std error:     {errors_array.std():.6f}")
    print(f"    {percentile}th percentile: {threshold:.6f} (threshold)")

    return threshold


if __name__ == "__main__":
    print("=" * 60)
    print("  Transformer Models — Demo")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy = torch.randn(2, 3, 224, 224).to(device)

    # Swin-Tiny
    try:
        swin = SwinChest(n_classes=6, freeze_backbone=True).to(device)
        out = swin(dummy)
        print(f"\n  Swin-Tiny:   input={dummy.shape} → output={out.shape}")
        swin.unfreeze_last_n_stages(2)
    except Exception as e:
        print(f"\n  Swin-Tiny: skipped (timm not available: {e})")

    # ViT-B/16
    try:
        vit = ViTChest(n_classes=6, freeze_backbone=True).to(device)
        out = vit(dummy)
        print(f"  ViT-B/16:    input={dummy.shape} → output={out.shape}")
        vit.unfreeze_last_n_blocks(4)
    except Exception as e:
        print(f"  ViT-B/16: skipped (timm not available: {e})")

    # Autoencoder
    ae = ChestAutoencoder(latent_dim=256).to(device)
    dummy_gray = torch.randn(2, 1, 224, 224).to(device)
    out = ae(dummy_gray)
    print(f"  Autoencoder: input={dummy_gray.shape} → output={out.shape}")
    error = ae.reconstruction_error(dummy_gray)
    print(f"  Recon error: {error.cpu().numpy()}")

    print("\n  ✓ Transformer models demo complete.")
