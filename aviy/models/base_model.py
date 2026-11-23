"""Aviy Base decoder-only language model with random initialization."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class BaseModelConfig:
    vocab_size: int = 16384
    context_length: int = 1024
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2688
    dropout: float = 0.0
    tie_embeddings: bool = True
    initialization: str = "random"
    pretrained_weights: bool = False

    def validate(self) -> None:
        for name in (
            "vocab_size", "context_length", "d_model",
            "n_layers", "n_heads", "d_ff",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.initialization != "random" or self.pretrained_weights:
            raise ValueError("Aviy Base must use random initialization only")


def validate_tokenizer_compatibility(
    tokenizer: object,
    config: BaseModelConfig,
) -> None:
    """Ensure every tokenizer ID fits inside the model embedding vocabulary."""
    vocabulary_size = getattr(tokenizer, "vocabulary_size", None)
    if isinstance(vocabulary_size, bool) or not isinstance(vocabulary_size, int):
        raise TypeError("Tokenizer must expose an integer vocabulary_size")
    if vocabulary_size < 1 or vocabulary_size > config.vocab_size:
        raise ValueError(
            f"Tokenizer vocabulary ({vocabulary_size}) does not fit "
            f"model vocabulary ({config.vocab_size})"
        )


class RMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + 1e-6)
        return normalized * self.weight


def apply_rope(value: torch.Tensor) -> torch.Tensor:
    _, _, length, width = value.shape
    positions = torch.arange(length, device=value.device, dtype=torch.float32)
    frequencies = 1.0 / (
        10000 ** (
            torch.arange(0, width, 2, device=value.device, dtype=torch.float32) / width
        )
    )
    angles = torch.outer(positions, frequencies)
    cosine = angles.cos().to(value.dtype)[None, None, :, :]
    sine = angles.sin().to(value.dtype)[None, None, :, :]
    first, second = value[..., 0::2], value[..., 1::2]
    rotated = torch.stack(
        (first * cosine - second * sine, first * sine + second * cosine),
        dim=-1,
    )
    return rotated.flatten(-2)


class SwiGLU(nn.Module):
    def __init__(self, config: BaseModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.value = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.output = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.output(F.silu(self.gate(value)) * self.value(value))


class CausalSelfAttention(nn.Module):
    def __init__(self, config: BaseModelConfig) -> None:
        super().__init__()
        self.heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = config.dropout

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, length, width = value.shape
        qkv = self.qkv(value).view(
            batch, length, 3, self.heads, self.head_dim
        )
        query, key, content = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        query = apply_rope(query)
        key = apply_rope(key)
        result = F.scaled_dot_product_attention(
            query,
            key,
            content,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        result = result.transpose(1, 2).contiguous().view(batch, length, width)
        return self.output(result)


class TransformerBlock(nn.Module):
    def __init__(self, config: BaseModelConfig) -> None:
        super().__init__()
        self.norm_attention = RMSNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.norm_mlp = RMSNorm(config.d_model)
        self.mlp = SwiGLU(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value + self.dropout(self.attention(self.norm_attention(value)))
        return value + self.dropout(self.mlp(self.norm_mlp(value)))


class AviyBaseLM(nn.Module):
    def __init__(self, config: BaseModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or BaseModelConfig()
        self.config.validate()
        self.token_embedding = nn.Embedding(
            self.config.vocab_size, self.config.d_model
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(self.config) for _ in range(self.config.n_layers)]
        )
        self.final_norm = RMSNorm(self.config.d_model)
        self.output = nn.Linear(
            self.config.d_model, self.config.vocab_size, bias=False
        )
        if self.config.tie_embeddings:
            self.output.weight = self.token_embedding.weight
        self.apply(self._initialize)
        self.initialization_verified = True

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be a rank-2 torch.long tensor")
        if input_ids.shape[1] > self.config.context_length:
            raise ValueError("Input exceeds configured context length")
        if input_ids.numel() and (
            input_ids.min() < 0 or input_ids.max() >= self.config.vocab_size
        ):
            raise ValueError("Input token ID outside configured vocabulary")
        value = self.token_embedding(input_ids)
        for block in self.blocks:
            value = block(value)
        return self.output(self.final_norm(value))

    def parameter_count(self, trainable_only: bool = False) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if not trainable_only or parameter.requires_grad
        )

    def configuration_report(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "parameters": self.parameter_count(),
            "trainable_parameters": self.parameter_count(True),
            "initialization": "random",
            "pretrained_weights_loaded": False,
            "tied_embeddings": (
                self.output.weight is self.token_embedding.weight
            ),
        }
