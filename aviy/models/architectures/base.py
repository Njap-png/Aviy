"""Shared language-model architecture contracts."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ArchitectureConfig:
    vocab_size: int = 16384
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 512
    context_length: int = 1024
    num_experts: int = 4
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for name in ("vocab_size", "d_model", "n_layers", "n_heads",
                     "d_ff", "context_length", "num_experts"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("attention head dimension must be even for RoPE")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class LanguageModel(nn.Module):
    """Common causal language-model interface."""

    architecture_name = "base"

    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.output_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def embed(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] > self.config.context_length:
            raise ValueError("input sequence exceeds context_length")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must contain integer token IDs")
        return self.token_embedding(input_ids.long())

    def forward(self, input_ids: Tensor) -> Tensor:
        raise NotImplementedError
