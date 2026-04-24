"""
ANN Model — Chest X-ray Benchmark
====================================
Simple Artificial Neural Network (multi-layer perceptron) for
baseline comparison on extracted features.

Classes:
    ANNChest — configurable MLP classifier
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class ANNChest(nn.Module):
    """Multi-layer perceptron for chest X-ray classification.

    Operates on pre-extracted feature vectors (e.g., HOG+LBP+PCA 512-d).

    Parameters
    ----------
    input_dim : int
        Input feature dimension. Default 512 (PCA output).
    hidden_dims : list[int]
        Hidden layer dimensions. Default [256, 128].
    n_classes : int
        Number of output classes. Default 6.
    dropout : float
        Dropout rate between layers. Default 0.3.
    use_batch_norm : bool
        Whether to use batch normalisation. Default True.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dims: Optional[List[int]] = None,
        n_classes: int = 6,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, n_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Feature vectors, shape (B, input_dim).

        Returns
        -------
        torch.Tensor
            Logits, shape (B, n_classes).
        """
        return self.network(x)


if __name__ == "__main__":
    print("=" * 60)
    print("  ANN Model — Demo")
    print("=" * 60)

    model = ANNChest(input_dim=512, hidden_dims=[256, 128], n_classes=6)
    dummy = torch.randn(4, 512)
    out = model(dummy)
    print(f"  Input:  {dummy.shape}")
    print(f"  Output: {out.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    print("\n  ✓ ANN model demo complete.")
