"""TopK SAE model implementation."""

from __future__ import annotations

import torch
from torch import nn


class TopKSAE(nn.Module):
    """Simple ReLU TopK sparse autoencoder for dense reward-model embeddings."""

    def __init__(self, input_dim: int, dict_size: int, k: int):
        super().__init__()
        if not 0 < k <= dict_size:
            raise ValueError("k must be between 1 and dict_size")
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.k = k
        self.encoder = nn.Linear(input_dim, dict_size)
        self.decoder = nn.Linear(dict_size, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        activations = torch.relu(self.encoder(x))
        values, indices = torch.topk(activations, k=self.k, dim=-1)
        sparse = torch.zeros_like(activations)
        sparse.scatter_(dim=-1, index=indices, src=values)
        return sparse

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
