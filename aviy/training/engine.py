"""Reproducible causal language-model training engine for CPU and CUDA."""
from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch import nn

from aviy.math.reproducibility import get_device, set_seed
from aviy.training.checkpoints import (
    atomic_save_checkpoint,
    load_verified_checkpoint,
)


def resource_report(device: torch.device) -> dict[str, object]:
    report: dict[str, object] = {
        "device": str(device),
        "cpu_threads": torch.get_num_threads(),
    }
    if device.type == "cuda":
        report["cuda_name"] = torch.cuda.get_device_name(device)
        report["cuda_allocated_bytes"] = torch.cuda.memory_allocated(device)
        report["cuda_reserved_bytes"] = torch.cuda.memory_reserved(device)
        report["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    return report


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    seed: int,
    scaler: Any = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    if step < 0:
        raise ValueError("Checkpoint step cannot be negative.")

    payload = {
        "format": "aviy-checkpoint-v1",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else {},
        "step": step,
        "seed": seed,
        "metadata": metadata or {},
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
    }
    atomic_save_checkpoint(path, payload)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    # Keep RNG tensors on CPU; optimizer.load_state_dict transfers optimizer
    # state to the parameter devices as required by PyTorch.
    del map_location
    payload = load_verified_checkpoint(path, map_location="cpu")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])

    if scaler is not None and payload.get("scaler"):
        scaler.load_state_dict(payload["scaler"])

    random.setstate(payload["python_rng"])
    torch.set_rng_state(payload["torch_rng"])

    cuda_rng = payload.get("cuda_rng")
    if cuda_rng is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_rng)

    return payload


def train_steps(
    model: nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    steps: int,
    learning_rate: float = 1e-4,
    device_name: str = "auto",
    seed: int = 42,
    precision: str = "auto",
    checkpoint_path: str | Path | None = None,
    resume_from_checkpoint: str | Path | None = None,
) -> dict[str, object]:
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("learning_rate must be finite and positive")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if precision not in {"auto", "fp32", "fp16", "bf16"}:
        raise ValueError("Unsupported precision")

    device = get_device(device_name)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    use_amp = precision != "fp32" and device.type == "cuda"
    amp_dtype = torch.float16 if precision in {"auto", "fp16"} else torch.bfloat16

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp and amp_dtype == torch.float16,
    )

    batch_cache = list(batches)
    if not batch_cache:
        raise ValueError("Training batches are empty")

    start_step = 0
    batch_position = 0

    if resume_from_checkpoint is None:
        set_seed(seed)
    else:
        restored = load_checkpoint(
            resume_from_checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
        )
        start_step = restored["step"]
        seed = restored["seed"]
        restored_metadata = restored.get("metadata", {})
        if not isinstance(restored_metadata, dict):
            raise ValueError("Checkpoint metadata is invalid.")
        batch_position = restored_metadata.get("batch_position", 0)
        if (
            isinstance(batch_position, bool)
            or not isinstance(batch_position, int)
            or not 0 <= batch_position < len(batch_cache)
        ):
            raise ValueError("Checkpoint batch position is invalid.")

    model.train()
    losses: list[float] = []
    started = time.perf_counter()

    for offset in range(steps):
        batch_index = (batch_position + offset) % len(batch_cache)
        inputs, targets = batch_cache[batch_index]
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            logits = model(inputs)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
            )

        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("Non-finite gradient norm")
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))

    global_step = start_step + steps
    next_batch_position = (batch_position + steps) % len(batch_cache)

    if checkpoint_path is not None:
        save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            step=global_step,
            seed=seed,
            scaler=scaler,
            metadata={
                "precision": precision,
                "batch_position": next_batch_position,
            },
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - started
    return {
        "steps": steps,
        "global_step": global_step,
        "resumed_from_step": start_step,
        "seed": seed,
        "learning_rate": learning_rate,
        "precision": precision,
        "losses": losses,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "elapsed_seconds": elapsed,
        "resources": resource_report(device),
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "next_batch_position": next_batch_position,
    }


def write_training_report(report: dict[str, object], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
