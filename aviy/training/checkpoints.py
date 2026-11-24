"""Atomic checkpoint persistence with SHA-256 integrity verification."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_save_checkpoint(
    path: str | Path,
    payload: dict[str, Any],
) -> str:
    """Save a checkpoint atomically and publish its SHA-256 sidecar."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)

    try:
        torch.save(payload, temporary)
        with temporary.open("r+b") as stream:
            stream.flush()
            os.fsync(stream.fileno())

        digest = _sha256(temporary)
        os.replace(temporary, destination)

        _atomic_text(
            destination.with_suffix(destination.suffix + ".sha256"),
            digest + "\n",
        )
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def verify_checkpoint(path: str | Path) -> str:
    """Reject missing, incomplete or modified checkpoint files."""
    source = Path(path)
    sidecar = source.with_suffix(source.suffix + ".sha256")

    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {source}")
    if not sidecar.is_file():
        raise ValueError("Checkpoint integrity sidecar is missing.")

    expected = sidecar.read_text(encoding="utf-8").strip()
    actual = _sha256(source)

    if len(expected) != 64 or actual != expected:
        raise ValueError("Checkpoint SHA-256 integrity verification failed.")

    return actual


def load_verified_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Verify bytes before deserializing a checkpoint."""
    source = Path(path)
    verify_checkpoint(source)

    payload = torch.load(
        source,
        map_location=map_location,
        weights_only=False,
    )

    if not isinstance(payload, dict):
        raise TypeError("Invalid checkpoint payload.")
    if payload.get("format") != "aviy-checkpoint-v1":
        raise ValueError("Unsupported checkpoint format.")
    if not isinstance(payload.get("model"), dict):
        raise TypeError("Checkpoint model state is missing.")
    if not isinstance(payload.get("optimizer"), dict):
        raise TypeError("Checkpoint optimizer state is missing.")
    step = payload.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("Checkpoint step is invalid.")
    seed = payload.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Checkpoint seed is invalid.")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise TypeError("Checkpoint metadata must be a dictionary.")

    return payload
