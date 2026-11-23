"""Numerically safe tensor utilities."""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


def ensure_finite(value: Tensor, name: str = "tensor") -> Tensor:
    """Raise when a tensor contains NaN or infinity."""
    if not isinstance(value, Tensor):
        raise TypeError("value must be a torch.Tensor")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} contains NaN or infinity")
    return value


def stable_softmax(value: Tensor, dim: int = -1) -> Tensor:
    """Compute stable softmax, promoting only low-precision inputs."""
    if value.ndim == 0:
        raise ValueError("softmax requires at least one dimension")
    if not -value.ndim <= dim < value.ndim:
        raise IndexError("softmax dimension is out of range")
    work = (
        value.float()
        if value.dtype in (torch.float16, torch.bfloat16)
        else value
    )
    result = torch.softmax(work, dim=dim)
    return result.to(dtype=value.dtype)


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    """Count model parameters without double-counting shared parameters."""
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


def tensor_summary(value: Tensor) -> dict[str, Any]:
    """Return safe structural and numerical tensor metadata."""
    ensure_finite(value)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "numel": value.numel(),
        "minimum": value.min().item() if value.numel() else None,
        "maximum": value.max().item() if value.numel() else None,
        "mean": value.float().mean().item() if value.numel() else None,
    }


def xavier_initialize(module: nn.Module) -> None:
    """Apply Xavier initialization to linear and convolutional weights."""
    for layer in module.modules():
        if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


def zeros_initialize(module: nn.Module) -> None:
    """Initialize linear and convolutional parameters to zero."""
    for layer in module.modules():
        if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
