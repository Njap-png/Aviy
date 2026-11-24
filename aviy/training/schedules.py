"""Learning-rate scheduling utilities."""

from __future__ import annotations

import math

import torch


def warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
    minimum_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by cosine decay."""
    if total_steps <= 0 or warmup_steps < 0 or warmup_steps >= total_steps:
        raise ValueError("Require total_steps > warmup_steps >= 0")
    if not 0.0 <= minimum_ratio <= 1.0:
        raise ValueError("minimum_ratio must be in [0, 1]")

    def scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-12)
        progress = min(
            max((step - warmup_steps) / (total_steps - warmup_steps), 0.0), 1.0
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
