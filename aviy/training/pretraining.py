"""Governed causal-language-model pretraining for Aviy Base."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import overload

import torch
from torch import Tensor

from aviy.data.pipeline import PreparedDataset
from aviy.models.base_model import AviyBaseLM, validate_tokenizer_compatibility
from aviy.tokenizer.bpe import ByteBPETokenizer
from aviy.training.authorization import (
    TrainingAuthorization,
    fingerprint_prepared_dataset,
    require_training_authorization,
)
from aviy.training.checkpoints import load_verified_checkpoint
from aviy.training.engine import train_steps


def _is_english(language: str) -> bool:
    normalized = language.strip().lower().replace("_", "-")
    return normalized in {"en", "eng", "english"} or normalized.startswith("en-")


class _PretrainingBatches(Sequence[tuple[Tensor, Tensor]]):
    """Lazy next-token windows that never cross document boundaries.

    Window stride equals context_length. Thus every adjacent token pair is
    trained exactly once, including pairs spanning consecutive windows.
    Only window metadata is retained; token tensors are built on demand.
    """

    def __init__(
        self,
        records: Sequence[object],
        tokenizer: ByteBPETokenizer,
        *,
        batch_size: int,
        context_length: int,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if context_length < 1:
            raise ValueError("context_length must be positive")

        self.records = records
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.context_length = context_length
        self._windows: list[tuple[int, int]] = []

        # Count windows without retaining encoded token sequences.
        for record_index, record in enumerate(records):
            text = getattr(record, "text", None)
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Pretraining record contains empty text")
            token_count = len(tokenizer.encode(text))
            if token_count < 2:
                raise ValueError(
                    "Pretraining record must produce at least two tokens"
                )
            pair_count = token_count - 1
            window_count = math.ceil(pair_count / context_length)
            for window_index in range(window_count):
                self._windows.append(
                    (record_index, window_index * context_length)
                )

    def __len__(self) -> int:
        return math.ceil(len(self._windows) / self.batch_size)

    @overload
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]: ...

    @overload
    def __getitem__(
        self, index: slice
    ) -> Sequence[tuple[Tensor, Tensor]]: ...

    def __getitem__(
        self, index: int | slice
    ) -> tuple[Tensor, Tensor] | Sequence[tuple[Tensor, Tensor]]:
        if isinstance(index, slice):
            return tuple(
                self[item] for item in range(*index.indices(len(self)))
            )
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("Batch index must be an integer")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        start = index * self.batch_size
        selected_windows = self._windows[start : start + self.batch_size]

        # Encode each selected document once for this batch only.
        encoded_records: dict[int, list[int]] = {}
        sequences: list[tuple[list[int], list[int]]] = []
        for record_index, offset in selected_windows:
            if record_index not in encoded_records:
                record_text = getattr(self.records[record_index], "text")
                encoded_records[record_index] = self.tokenizer.encode(
                    record_text
                )
            token_ids = encoded_records[record_index]
            segment = token_ids[
                offset : offset + self.context_length + 1
            ]
            if len(segment) < 2:
                raise ValueError("Pretraining window has fewer than two tokens")
            sequences.append((segment[:-1], segment[1:]))

        max_length = max(len(inputs) for inputs, _ in sequences)
        input_batch = torch.zeros(
            (len(sequences), max_length), dtype=torch.long
        )
        target_batch = torch.full(
            (len(sequences), max_length), -100, dtype=torch.long
        )

        for row, (inputs, targets) in enumerate(sequences):
            length = len(inputs)
            input_batch[row, :length] = torch.tensor(
                inputs, dtype=torch.long
            )
            target_batch[row, :length] = torch.tensor(
                targets, dtype=torch.long
            )

        return input_batch, target_batch


def pretrain_base_model(
    model: AviyBaseLM,
    tokenizer: ByteBPETokenizer,
    dataset: PreparedDataset,
    *,
    steps: int,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    device_name: str = "auto",
    seed: int = 42,
    precision: str = "auto",
    checkpoint_path: str | Path | None = None,
    resume_from_checkpoint: str | Path | None = None,
    scheduler_total_steps: int | None = None,
    scheduler_warmup_steps: int = 0,
    minimum_learning_rate_ratio: float = 0.0,
    authorization: TrainingAuthorization | None = None,
) -> dict[str, object]:
    """Train Aviy Base on an explicitly authorized prepared English dataset."""
    for name, value, minimum in (
        ("steps", steps, 1),
        ("batch_size", batch_size, 1),
        ("seed", seed, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")

    if not dataset.audit.passed:
        raise ValueError("Pretraining dataset audit did not pass")

    records = tuple(dataset.records)
    if not records:
        raise ValueError("Prepared pretraining dataset is empty")

    allowed_splits = {"train", "validation", "test"}
    for record in records:
        if getattr(record, "split", None) not in allowed_splits:
            raise ValueError("Prepared dataset contains an unknown split")
        if getattr(record, "approved_for_training", False) is not True:
            raise PermissionError("Prepared dataset contains a non-approved record")
        language = getattr(record, "language", "")
        if not isinstance(language, str) or not _is_english(language):
            raise ValueError("Base pretraining currently requires explicit English metadata")

    train_records = tuple(
        record for record in records if getattr(record, "split", None) == "train"
    )
    if not train_records:
        raise ValueError("Prepared dataset has no training-split records")

    fingerprint = fingerprint_prepared_dataset(dataset)
    require_training_authorization(
        authorization,
        expected_stage="base_pretraining",
        expected_fingerprint=fingerprint,
    )

    config = model.config
    if getattr(config, "pretrained_weights", None) is not False:
        raise ValueError("Aviy Base must use random initialization")
    if getattr(config, "initialization", None) != "random":
        raise ValueError("Aviy Base initialization must be random")

    validate_tokenizer_compatibility(tokenizer, config)
    context_length = getattr(config, "context_length", None)
    if (
        isinstance(context_length, bool)
        or not isinstance(context_length, int)
        or context_length < 2
    ):
        raise ValueError("Model context_length must be at least two")

    batches = _PretrainingBatches(
        train_records,
        tokenizer,
        batch_size=batch_size,
        context_length=context_length,
    )
    if len(batches) < 1:
        raise ValueError("No usable pretraining batches")

    settings: dict[str, object] = {
        "dataset_fingerprint": fingerprint,
        "tokenizer_vocabulary_size": tokenizer.vocabulary_size,
        "model_config": vars(config).copy(),
        "batch_size": batch_size,
        "context_length": context_length,
        "learning_rate": float(learning_rate),
        "seed": seed,
        "scheduler_total_steps": scheduler_total_steps,
        "scheduler_warmup_steps": scheduler_warmup_steps,
        "minimum_learning_rate_ratio": minimum_learning_rate_ratio,
    }

    if resume_from_checkpoint is not None:
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required when resuming")
        payload = load_verified_checkpoint(resume_from_checkpoint, map_location="cpu")
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("Pretraining checkpoint metadata is invalid")
        workflow = metadata.get("workflow")
        if not isinstance(workflow, dict) or workflow.get("pretraining") != settings:
            raise ValueError("Checkpoint does not match this pretraining dataset or configuration")

    report = train_steps(
        model,
        batches,
        steps=steps,
        learning_rate=learning_rate,
        device_name=device_name,
        seed=seed,
        precision=precision,
        checkpoint_path=checkpoint_path,
        resume_from_checkpoint=resume_from_checkpoint,
        scheduler_total_steps=scheduler_total_steps,
        scheduler_warmup_steps=scheduler_warmup_steps,
        minimum_learning_rate_ratio=minimum_learning_rate_ratio,
        checkpoint_metadata={"pretraining": settings},
    )

    report.update(
        {
            "training_stage": "base_pretraining",
            "dataset_fingerprint": fingerprint,
            "tokenizer_vocabulary_size": tokenizer.vocabulary_size,
            "training_records": len(train_records),
            "available_batches": len(batches),
            "batch_size": batch_size,
            "context_length": context_length,
            "random_initialization": True,
            "pretrained_weights_loaded": False,
            "capability_score_claimed": False,
        }
    )
    return report
