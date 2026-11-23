"""Reproducible random seeds and compute-device selection."""
from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch random number generators."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_device(requested: str = "auto") -> torch.device:
    """Resolve and validate a requested compute device."""
    if not isinstance(requested, str):
        raise TypeError("requested device must be a string")
    name = requested.strip().lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if name.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        try:
            index = int(name.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid CUDA device: {requested}") from exc
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index is unavailable: {index}")
    if name not in {"cpu", "cuda", "mps"} and not name.startswith("cuda:"):
        raise ValueError(f"Unsupported device: {requested}")
    if name == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(name)
