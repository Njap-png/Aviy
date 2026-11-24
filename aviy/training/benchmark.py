"""Small reproducible optimizer benchmark on synthetic data."""

from __future__ import annotations

import time

import torch
from torch import nn

from aviy.math.reproducibility import set_seed
from aviy.training.optimizers import build_optimizer


def benchmark_optimizers(
    *, seed: int = 42, steps: int = 20, width: int = 32
) -> dict[str, dict[str, float]]:
    """Compare optimizers on the same deterministic regression problem."""
    if steps < 2 or width < 2:
        raise ValueError("steps and width must be at least 2")
    results: dict[str, dict[str, float]] = {}
    set_seed(seed)
    x = torch.randn(64, width)
    target = torch.randn(64, width)
    for name in ("sgd", "adam", "adamw"):
        set_seed(seed)
        model = nn.Linear(width, width)
        optimizer = build_optimizer(name, model.parameters(), learning_rate=0.01)
        started = time.perf_counter()
        first_loss = 0.0
        final_loss = 0.0
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(x), target)
            if step == 0:
                first_loss = loss.item()
            loss.backward()
            optimizer.step()
            final_loss = loss.item()
        results[name] = {
            "initial_loss": first_loss,
            "final_loss": final_loss,
            "elapsed_seconds": time.perf_counter() - started,
        }
    return results
