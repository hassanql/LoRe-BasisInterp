"""TopK SAE model implementation with dead-feature revival support."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TopKSAE(nn.Module):
    """ReLU TopK sparse autoencoder for dense reward-model embeddings.

    Improvements over the original baseline:
    - learnable pre-encoder bias (input centering)
    - unit-norm decoder columns
    - encoder initialized as the decoder transpose
    - auxiliary TopK reconstruction on dead features
    """

    def __init__(
        self,
        input_dim: int,
        dict_size: int,
        k: int,
        *,
        normalize_decoder: bool = True,
        aux_k: int | None = None,
    ):
        super().__init__()
        if not 0 < k <= dict_size:
            raise ValueError("k must be between 1 and dict_size")
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.k = k
        self.normalize_decoder = normalize_decoder
        # Aux path uses up to k dead features by default (capped by live dead count).
        self.aux_k = aux_k if aux_k is not None else k

        self.b_pre = nn.Parameter(torch.zeros(input_dim))
        self.encoder = nn.Linear(input_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, input_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize decoder columns unit-norm; encoder ≈ decoder.T."""
        nn.init.kaiming_uniform_(self.decoder.weight)
        with torch.no_grad():
            self.decoder.bias.zero_()
            self.normalize_decoder_()
            self.encoder.weight.copy_(self.decoder.weight.t())
            self.encoder.bias.zero_()
            self.b_pre.zero_()

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        """Project decoder columns onto the unit sphere."""
        if not self.normalize_decoder:
            return
        norms = torch.linalg.norm(self.decoder.weight, dim=0, keepdim=True).clamp_min(1e-8)
        self.decoder.weight.div_(norms)

    @torch.no_grad()
    def set_pre_bias(self, mean: torch.Tensor) -> None:
        """Initialize pre-encoder bias from a data mean (e.g. train embedding mean)."""
        if mean.shape != (self.input_dim,):
            raise ValueError(f"mean shape {tuple(mean.shape)} != ({self.input_dim},)")
        self.b_pre.copy_(mean.detach().to(self.b_pre.dtype).to(self.b_pre.device))

    def encode_pre_acts(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x - self.b_pre)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre_acts = self.encode_pre_acts(x)
        return self._topk_activate(pre_acts, self.k)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z) + self.b_pre

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def forward_with_loss(
        self,
        x: torch.Tensor,
        *,
        dead_mask: torch.Tensor | None = None,
        aux_k_coef: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """Forward pass returning reconstruction and optional aux dead-feature loss."""
        pre_acts = self.encode_pre_acts(x)
        z = self._topk_activate(pre_acts, self.k)
        x_hat = self.decode(z)
        recon_mse = F.mse_loss(x_hat, x)

        aux_mse = x.new_zeros(())
        if aux_k_coef > 0.0 and dead_mask is not None and bool(dead_mask.any()):
            residual = (x - x_hat).detach()
            aux_latent = self._topk_activate_masked(pre_acts, dead_mask, self.aux_k)
            # Aux path reconstructs residual only (no b_pre), so dead features re-enter.
            aux_recon = self.decoder(aux_latent)
            aux_mse = F.mse_loss(aux_recon, residual)

        total = recon_mse + aux_k_coef * aux_mse
        return {
            "x_hat": x_hat,
            "z": z,
            "pre_acts": pre_acts,
            "recon_mse": recon_mse,
            "aux_mse": aux_mse,
            "loss": total,
        }

    @staticmethod
    def _topk_activate(pre_acts: torch.Tensor, k: int) -> torch.Tensor:
        k = min(k, pre_acts.shape[-1])
        values, indices = torch.topk(pre_acts, k=k, dim=-1)
        values = F.relu(values)
        sparse = torch.zeros_like(pre_acts)
        sparse.scatter_(dim=-1, index=indices, src=values)
        return sparse

    @staticmethod
    def _topk_activate_masked(
        pre_acts: torch.Tensor,
        dead_mask: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """TopK over dead features only; live features are masked out."""
        if dead_mask.ndim != 1 or dead_mask.shape[0] != pre_acts.shape[-1]:
            raise ValueError("dead_mask must have shape [dict_size]")
        n_dead = int(dead_mask.sum().item())
        if n_dead == 0:
            return torch.zeros_like(pre_acts)
        k = min(k, n_dead)
        masked = pre_acts.masked_fill(~dead_mask.unsqueeze(0), -1e9)
        values, indices = torch.topk(masked, k=k, dim=-1)
        values = F.relu(values)
        sparse = torch.zeros_like(pre_acts)
        sparse.scatter_(dim=-1, index=indices, src=values)
        return sparse
