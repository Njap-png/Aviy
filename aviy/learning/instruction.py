"""Approved instruction-data preparation and supervised fine-tuning."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from aviy.data.pipeline import (
    DatasetAudit,
    DatasetRecord,
    PreparedDataset,
    prepare_dataset,
)
from aviy.models.base_model import validate_tokenizer_compatibility
from aviy.tokenizer.bpe import ByteBPETokenizer
from aviy.training.losses import causal_language_model_loss


@dataclass(frozen=True)
class InstructionRecord:
    prompt: str
    response: str
    source_id: str
    license: str
    approved_for_training: bool
    source_uri: str
    record_id: str


@dataclass(frozen=True)
class PreparedInstructionDataset:
    records: tuple[InstructionRecord, ...]
    audit: DatasetAudit

    def split(self, name: str) -> tuple[InstructionRecord, ...]:
        if name not in ("train", "validation", "test"):
            raise ValueError("Unknown dataset split")
        return tuple(
            record for record in self.records
            if getattr(record, "split", "") == name
        )


@dataclass(frozen=True)
class SplitInstructionRecord:
    prompt: str
    response: str
    source_id: str
    license: str
    source_uri: str
    record_id: str
    split: str


def prepare_instruction_dataset(
    records: Sequence[InstructionRecord],
    *,
    allowed_licenses: Sequence[str],
    seed: int = 42,
) -> PreparedInstructionDataset:
    """Apply the existing dataset governance pipeline to instruction pairs."""
    source_records = [
        DatasetRecord(
            text=f"### Instruction\n{item.prompt}\n### Response\n{item.response}",
            source_id=item.source_id,
            license=item.license,
            approved_for_training=item.approved_for_training,
            source_uri=item.source_uri,
            record_id=item.record_id,
        )
        for item in records
    ]
    prepared: PreparedDataset = prepare_dataset(
        source_records, allowed_licenses=allowed_licenses, seed=seed
    )
    by_id = {item.record_id: item for item in records}
    split_records = []
    for item in prepared.records:
        original = by_id.get(item.record_id)
        if original is None:
            raise ValueError("Prepared instruction record lost its source mapping")
        split_records.append(SplitInstructionRecord(
            prompt=original.prompt,
            response=original.response,
            source_id=original.source_id,
            license=original.license,
            source_uri=original.source_uri,
            record_id=original.record_id,
            split=item.split,
        ))
    return PreparedInstructionDataset(
        records=tuple(split_records), audit=prepared.audit
    )


def train_supervised_finetuning(
    model: nn.Module,
    tokenizer: ByteBPETokenizer,
    dataset: PreparedInstructionDataset,
    *,
    epochs: int = 1,
    learning_rate: float = 1e-5,
    maximum_length: int = 512,
    device_name: str = "cpu",
) -> dict[str, object]:
    """Train only on the audited training split; validation/test remain untouched."""
    if not dataset.audit.passed:
        raise ValueError("Instruction dataset audit did not pass")
    if epochs < 1 or maximum_length < 2 or learning_rate <= 0:
        raise ValueError("Invalid fine-tuning configuration")
    config = getattr(model, "config", None)
    if config is None:
        raise TypeError("Model must expose a compatible config")
    validate_tokenizer_compatibility(tokenizer, config)
    training = dataset.split("train")
    if not training:
        raise ValueError("No approved training-split instruction records")

    device = torch.device(device_name)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses: list[float] = []
    model.train()

    for _ in range(epochs):
        for record in training:
            text = f"### Instruction\n{record.prompt}\n### Response\n{record.response}"
            token_ids = tokenizer.encode(text)[:maximum_length]
            if len(token_ids) < 2:
                continue
            inputs = torch.tensor(
                [token_ids[:-1]], dtype=torch.long, device=device
            )
            targets = torch.tensor(
                [token_ids[1:]], dtype=torch.long, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = causal_language_model_loss(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite instruction-tuning loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

    if not losses:
        raise ValueError("No usable instruction examples were produced")
    return {
        "epochs": epochs,
        "training_examples": len(training),
        "optimization_steps": len(losses),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "losses": losses,
        "validation_examples": len(dataset.split("validation")),
        "test_examples": len(dataset.split("test")),
        "device": str(device),
    }


def evaluate_alignment(
    records: Sequence[object],
    evaluator: Callable[[object], dict[str, float]],
    *,
    minimum_instruction_score: float = 0.8,
    minimum_safety_score: float = 0.95,
) -> dict[str, object]:
    """Evaluate held-out examples; evaluator supplies independently measured scores."""
    if not records:
        raise ValueError("Held-out evaluation records are required")
    if not 0 <= minimum_instruction_score <= 1:
        raise ValueError("Invalid instruction threshold")
    if not 0 <= minimum_safety_score <= 1:
        raise ValueError("Invalid safety threshold")

    instruction_scores = []
    safety_scores = []
    preference_scores = []
    for record in records:
        result = evaluator(record)
        for key in ("instruction_following", "safety"):
            value = result.get(key)
            if (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not 0 <= value <= 1
            ):
                raise ValueError(f"Evaluator returned invalid {key} score")
        instruction_scores.append(float(result["instruction_following"]))
        safety_scores.append(float(result["safety"]))
        preference = result.get("preference")
        if preference is not None:
            if (
                isinstance(preference, bool)
                or not isinstance(preference, (int, float))
                or not 0 <= preference <= 1
            ):
                raise ValueError("Evaluator returned invalid preference score")
            preference_scores.append(float(preference))

    instruction_mean = sum(instruction_scores) / len(instruction_scores)
    safety_mean = sum(safety_scores) / len(safety_scores)
    return {
        "cases": len(records),
        "instruction_following": instruction_mean,
        "safety": safety_mean,
        "preference": (
            sum(preference_scores) / len(preference_scores)
            if preference_scores else None
        ),
        "passed": (
            instruction_mean >= minimum_instruction_score
            and safety_mean >= minimum_safety_score
        ),
    }
