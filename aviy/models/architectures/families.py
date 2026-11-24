"""Initial reference implementations of Aviy neural architecture families."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from aviy.models.architectures.base import ArchitectureConfig, LanguageModel


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return x * scale.to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.gate = nn.Linear(width, hidden)
        self.up = nn.Linear(width, hidden)
        self.down = nn.Linear(hidden, width)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


def apply_rotary_embedding(value: Tensor) -> Tensor:
    """Apply rotary position embeddings to [batch, heads, sequence, head_dim]."""
    head_dim = value.shape[-1]
    length = value.shape[-2]
    frequencies = 1.0 / (
        10000.0
        ** (
            torch.arange(0, head_dim, 2, device=value.device, dtype=torch.float32)
            / head_dim
        )
    )
    positions = torch.arange(length, device=value.device, dtype=torch.float32)
    angles = positions[:, None] * frequencies[None, :]
    cosine = angles.cos().to(dtype=value.dtype)[None, None, :, :]
    sine = angles.sin().to(dtype=value.dtype)[None, None, :, :]

    even = value[..., 0::2]
    odd = value[..., 1::2]
    rotated = torch.empty_like(value)
    rotated[..., 0::2] = even * cosine - odd * sine
    rotated[..., 1::2] = even * sine + odd * cosine
    return rotated


class CausalAttention(nn.Module):
    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        self.heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.output = nn.Linear(config.d_model, config.d_model)
        self.dropout = config.dropout

    def forward(self, x: Tensor) -> Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split(t: Tensor) -> Tensor:
            return t.view(batch, length, self.heads, self.head_dim).transpose(1, 2)

        q, k, v = split(q), split(k), split(v)
        q = apply_rotary_embedding(q)
        k = apply_rotary_embedding(k)
        mask = torch.ones(length, length, device=x.device, dtype=torch.bool).triu(1)
        result = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=~mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        result = result.transpose(1, 2).contiguous().view(batch, length, width)
        return self.output(result)


class TransformerBlock(nn.Module):
    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attention = CausalAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.mlp = SwiGLU(config.d_model, config.d_ff)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.dropout(self.attention(self.norm1(x)))
        return x + self.dropout(self.mlp(self.norm2(x)))


class TransformerLM(LanguageModel):
    architecture_name = "transformer"

    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__(config)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.d_model)

    def hidden(self, input_ids: Tensor) -> Tensor:
        """Return contextual representations before the vocabulary projection."""
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.output_head(self.hidden(input_ids))


class RecurrentLM(LanguageModel):
    architecture_name = "recurrent"

    def __init__(self, config: ArchitectureConfig, cell: str = "gru") -> None:
        super().__init__(config)
        recurrent = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}
        if cell not in recurrent:
            raise ValueError(f"Unknown recurrent cell: {cell}")
        self.cell = cell
        self.recurrent = recurrent[cell](
            config.d_model,
            config.d_model,
            config.n_layers,
            batch_first=True,
            dropout=config.dropout if config.n_layers > 1 else 0.0,
        )
        self.norm = RMSNorm(config.d_model)

    def forward(self, input_ids: Tensor) -> Tensor:
        x, _ = self.recurrent(self.embed(input_ids))
        return self.output_head(self.norm(x))


class SSMBlock(nn.Module):
    """Simple gated recurrent state-space reference block."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(width, width)
        self.state_projection = nn.Linear(width, width)
        self.gate = nn.Linear(width, width)
        self.norm = RMSNorm(width)

    def forward(self, x: Tensor) -> Tensor:
        state = torch.zeros_like(x[:, 0])
        outputs = []
        for index in range(x.shape[1]):
            current = x[:, index]
            state = torch.tanh(
                self.input_projection(current) + self.state_projection(state)
            )
            gate = torch.sigmoid(self.gate(current))
            outputs.append(gate * state + (1.0 - gate) * current)
        return self.norm(torch.stack(outputs, dim=1))


class SSMLM(LanguageModel):
    architecture_name = "ssm"

    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__(config)
        self.blocks = nn.ModuleList(
            [SSMBlock(config.d_model) for _ in range(config.n_layers)]
        )

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = x + block(x)
        return self.output_head(x)


class RWKVTimeMix(nn.Module):
    """Educational time-mixing block; not a full optimized RWKV implementation."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.mix = nn.Parameter(torch.full((1, 1, width), 0.5))
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.receptance = nn.Linear(width, width)
        self.output = nn.Linear(width, width)

    def forward(self, x: Tensor) -> Tensor:
        shifted = torch.cat((torch.zeros_like(x[:, :1]), x[:, :-1]), dim=1)
        mixed = x * self.mix + shifted * (1.0 - self.mix)
        return self.output(
            torch.sigmoid(self.receptance(mixed))
            * self.value(torch.tanh(self.key(mixed)))
        )


class RWKVLM(LanguageModel):
    architecture_name = "rwkv"

    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__(config)
        self.blocks = nn.ModuleList(
            [RWKVTimeMix(config.d_model) for _ in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.d_model)

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = x + block(x)
        return self.output_head(self.norm(x))


class ConvLM(LanguageModel):
    architecture_name = "cnn"

    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__(config)
        self.layers = nn.ModuleList(
            [
                nn.Conv1d(config.d_model, config.d_model, 3, padding=0)
                for _ in range(config.n_layers)
            ]
        )
        self.norm = RMSNorm(config.d_model)

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embed(input_ids).transpose(1, 2)
        for layer in self.layers:
            padded = F.pad(x, (2, 0))
            x = x + F.gelu(layer(padded))
        return self.output_head(self.norm(x.transpose(1, 2)))


class MoELayer(nn.Module):
    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        self.router = nn.Linear(config.d_model, config.num_experts)
        self.experts = nn.ModuleList(
            [SwiGLU(config.d_model, config.d_ff) for _ in range(config.num_experts)]
        )

    def forward(self, x: Tensor) -> Tensor:
        weights = torch.softmax(self.router(x), dim=-1)
        top_values, top_indices = weights.topk(min(2, len(self.experts)), dim=-1)
        top_values = top_values / top_values.sum(dim=-1, keepdim=True)
        output = torch.zeros_like(x)
        for expert_index, expert in enumerate(self.experts):
            contribution = expert(x)
            mask = top_indices.eq(expert_index)
            weight = (top_values * mask).sum(dim=-1, keepdim=True)
            output = output + contribution * weight
        return output


class MoELM(LanguageModel):
    architecture_name = "moe"

    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__(config)
        self.layers = nn.ModuleList([MoELayer(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = x + layer(x)
        return self.output_head(self.norm(x))


class GNNLayer(nn.Module):
    """Sequence-neighborhood graph aggregation reference layer."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.self_projection = nn.Linear(width, width)
        self.neighbor_projection = nn.Linear(width, width)

    def forward(self, x: Tensor) -> Tensor:
        neighbors = F.pad(x[:, :-1], (0, 0, 1, 0))
        return F.gelu(self.self_projection(x) + self.neighbor_projection(neighbors))


class GNNLM(LanguageModel):
    architecture_name = "gnn"

    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__(config)
        self.layers = nn.ModuleList(
            [GNNLayer(config.d_model) for _ in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.d_model)

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = x + layer(x)
        return self.output_head(self.norm(x))


class HybridLM(LanguageModel):
    architecture_name = "hybrid"

    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__(config)
        self.transformer = TransformerLM(config)
        self.recurrent = nn.GRU(config.d_model, config.d_model, batch_first=True)
        self.fusion = nn.Linear(2 * config.d_model, config.d_model)

    def forward(self, input_ids: Tensor) -> Tensor:
        embedded = self.embed(input_ids)
        transformer_hidden = self.transformer.hidden(input_ids)
        recurrent_hidden, _ = self.recurrent(embedded)
        fused = torch.tanh(
            self.fusion(torch.cat((transformer_hidden, recurrent_hidden), dim=-1))
        )
        return self.output_head(fused)
