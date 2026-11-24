"""Optimizer interfaces and implementations."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import nn


class OptimizerFactory(Protocol):
    def __call__(
        self, parameters: object, **kwargs: object
    ) -> torch.optim.Optimizer: ...


def build_optimizer(
    name: str,
    parameters: object,
    *,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """Create a supported optimizer with validated hyperparameters."""
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay cannot be negative")
    choices = {
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
    }
    try:
        factory = choices[name.lower()]
    except (KeyError, AttributeError) as exc:
        raise ValueError(f"Unsupported optimizer: {name}") from exc
    return factory(parameters, lr=learning_rate, weight_decay=weight_decay)


def clip_gradients(model: nn.Module, max_norm: float = 1.0) -> torch.Tensor:
    """Clip gradients and return the pre-clipping total norm."""
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    parameters = [p for p in model.parameters() if p.grad is not None]
    if not parameters:
        raise ValueError("No gradients are available to clip")
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm)
