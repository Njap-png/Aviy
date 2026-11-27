"""Approved instruction-data preparation and response-only supervised fine-tuning."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from aviy.data.pipeline import (
    DatasetAudit,
    DatasetRecord,
    PreparedDataset,
    prepare_dataset,
)
from aviy.math.reproducibility import set_seed
from aviy.models.base_model import validate_tokenizer_compatibility
from aviy.tokenizer.bpe import ByteBPETokenizer
from aviy.training.authorization import (
    TrainingAuthorization,
    fingerprint_prepared_dataset,
    require_training_authorization,
)
from aviy.training.engine import load_checkpoint, save_checkpoint
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
    dataset_revision: str = ""
    license_uri: str = ""
    language: str = ""
    language_script: str = ""
    retrieved_at: str = ""
    training_stage: str = ""
    content_category: str = ""
    processing_history: tuple[str, ...] = ()
    privacy_screening: str = ""
    duplicate_screening: str = ""


@dataclass(frozen=True)
class PreparedInstructionDataset:
    records: tuple[SplitInstructionRecord, ...]
    audit: DatasetAudit

    def split(self, name: str) -> tuple[SplitInstructionRecord, ...]:
        if name not in ("train", "validation", "test"):
            raise ValueError("Unknown dataset split")
        return tuple(
            record for record in self.records if getattr(record, "split", "") == name
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
    dataset_revision: str = ""
    license_uri: str = ""
    language: str = ""
    language_script: str = ""
    retrieved_at: str = ""
    training_stage: str = ""
    content_category: str = ""
    processing_history: tuple[str, ...] = ()
    privacy_screening: str = ""
    duplicate_screening: str = ""


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
            dataset_revision=item.dataset_revision,
            license_uri=item.license_uri,
            language=item.language,
            language_script=item.language_script,
            retrieved_at=item.retrieved_at,
            training_stage=item.training_stage,
            content_category=item.content_category,
            processing_history=item.processing_history,
            privacy_screening=item.privacy_screening,
            duplicate_screening=item.duplicate_screening,
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
        split_records.append(
            SplitInstructionRecord(
                prompt=original.prompt,
                response=original.response,
                source_id=original.source_id,
                license=original.license,
                source_uri=original.source_uri,
                record_id=original.record_id,
                split=item.split,
                dataset_revision=item.dataset_revision,
                license_uri=item.license_uri,
                language=item.language,
                language_script=item.language_script,
                retrieved_at=item.retrieved_at,
                training_stage=item.training_stage,
                content_category=item.content_category,
                processing_history=item.processing_history,
                privacy_screening=item.privacy_screening,
                duplicate_screening=item.duplicate_screening,
            )
        )
    return PreparedInstructionDataset(
        records=tuple(split_records), audit=prepared.audit
    )


def _encode_instruction_record(
    record: InstructionRecord | SplitInstructionRecord,
    tokenizer: ByteBPETokenizer,
    maximum_length: int,
) -> tuple[list[int], int]:
    """Encode a prompt/response pair while retaining response and framing tokens."""
    if not isinstance(record.prompt, str) or not record.prompt.strip():
        raise ValueError(f"Instruction prompt is empty: {record.record_id}")
    if not isinstance(record.response, str) or not record.response.strip():
        raise ValueError(f"Instruction response is empty: {record.record_id}")

    instruction_header = tokenizer.encode("### Instruction\n")
    prompt_ids = tokenizer.encode(record.prompt)
    response_header = tokenizer.encode("\n### Response\n")
    response_ids = tokenizer.encode(record.response)

    if not response_ids:
        raise ValueError(f"Instruction response produced no tokens: {record.record_id}")

    framing_size = len(instruction_header) + len(response_header)

    if maximum_length <= framing_size:
        raise ValueError(
            "maximum_length must leave at least one token for the response "
            "after instruction framing"
        )

    response_budget = min(
        max(1, maximum_length // 2),
        maximum_length - framing_size,
    )
    prefix_budget = maximum_length - response_budget
    prompt_budget = prefix_budget - framing_size
    prefix_ids = (
        instruction_header
        + prompt_ids[:prompt_budget]
        + response_header
    )
    response_ids = response_ids[:response_budget]

    sequence = prefix_ids + response_ids
    if len(sequence) < 2:
        raise ValueError(f"Instruction example is too short: {record.record_id}")
    if len(sequence) > maximum_length:
        raise RuntimeError("Instruction sequence exceeded its configured token budget")

    return sequence, len(prefix_ids)


def _collate_instruction_records(
    examples: Sequence[tuple[list[int], int]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Right-pad inputs and mask prompt/padding targets with ignore_index."""
    if not examples:
        raise ValueError("Cannot collate an empty instruction batch")

    input_length = max(len(sequence) - 1 for sequence, _ in examples)
    if input_length < 1:
        raise ValueError("Instruction batch contains an empty sequence")

    inputs = torch.zeros(
        (len(examples), input_length), dtype=torch.long, device=device
    )
    targets = torch.full(
        (len(examples), input_length), -100, dtype=torch.long, device=device
    )

    response_tokens = 0
    for row, (sequence, prefix_length) in enumerate(examples):
        input_ids = sequence[:-1]
        target_ids = sequence[1:]
        valid_length = len(input_ids)
        response_start = prefix_length - 1

        if response_start < 0 or response_start >= len(target_ids):
            raise ValueError("Instruction example has no response prediction targets")

        inputs[row, :valid_length] = torch.tensor(
            input_ids, dtype=torch.long, device=device
        )
        targets[row, :valid_length] = torch.tensor(
            target_ids, dtype=torch.long, device=device
        )
        targets[row, :response_start] = -100
        response_tokens += valid_length - response_start

    if response_tokens < 1:
        raise ValueError("Instruction batch contains no response targets")

    return inputs, targets, response_tokens


def _fingerprint(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dataset_fingerprint(dataset: PreparedInstructionDataset) -> str:
    # Share the canonical fingerprint used by tokenizer authorization.
    return fingerprint_prepared_dataset(dataset)


def _tokenizer_fingerprint(tokenizer: ByteBPETokenizer) -> str:
    return _fingerprint(
        {
            "vocabulary_size": tokenizer.vocabulary_size,
            "merges": list(tokenizer.merges),
        }
    )


def _model_config_fingerprint(config: object) -> str:
    return _fingerprint(vars(config))


def _validate_training_integer(name: str, value: object, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")


def _evaluate_instruction_loss(
    model: nn.Module,
    examples: Sequence[tuple[list[int], int]],
    *,
    batch_size: int,
    device: torch.device,
) -> float | None:
    if not examples:
        return None

    total_loss = 0.0
    total_tokens = 0

    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        inputs, targets, token_count = _collate_instruction_records(batch, device)
        logits = model(inputs)

        if (
            not isinstance(logits, torch.Tensor)
            or logits.ndim != 3
            or logits.shape[:2] != targets.shape
            or not logits.is_floating_point()
        ):
            raise ValueError("Model must return floating-point logits shaped [B,T,V]")
        if torch.any(targets[targets != -100] >= logits.shape[-1]):
            raise ValueError("Validation target exceeds model vocabulary size")

        loss_sum = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        if not torch.isfinite(loss_sum):
            raise FloatingPointError("Non-finite instruction validation loss")

        total_loss += float(loss_sum.detach().cpu())
        total_tokens += token_count

    if total_tokens == 0:
        raise ValueError("Validation split contains no response targets")
    return total_loss / total_tokens


def train_supervised_finetuning(
    model: nn.Module,
    tokenizer: ByteBPETokenizer,
    dataset: PreparedInstructionDataset,
    *,
    epochs: int = 1,
    learning_rate: float = 1e-5,
    maximum_length: int = 512,
    device_name: str = "cpu",
    batch_size: int = 4,
    seed: int = 42,
    checkpoint_path: str | Path | None = None,
    resume_from_checkpoint: str | Path | None = None,
    authorization: TrainingAuthorization | None = None,
) -> dict[str, object]:
    """Train with response-only loss after explicit dataset-bound authorization."""
    _validate_training_integer("epochs", epochs, 1)
    _validate_training_integer("maximum_length", maximum_length, 2)
    _validate_training_integer("batch_size", batch_size, 1)
    _validate_training_integer("seed", seed, 0)

    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("learning_rate must be finite and positive")
    if not dataset.audit.passed:
        raise ValueError("Instruction dataset audit did not pass")
    require_training_authorization(
        authorization,
        expected_stage="supervised_finetuning",
        expected_fingerprint=_dataset_fingerprint(dataset),
    )
    if resume_from_checkpoint is not None and checkpoint_path is None:
        raise ValueError("checkpoint_path is required when resuming training")

    config = getattr(model, "config", None)
    if config is None:
        raise TypeError("Model must expose a compatible config")
    validate_tokenizer_compatibility(tokenizer, config)

    context_length = getattr(config, "context_length", None)
    if isinstance(context_length, int) and maximum_length > context_length:
        raise ValueError("maximum_length exceeds the model context length")

    train_records = dataset.split("train")
    validation_records = dataset.split("validation")
    if not train_records:
        raise ValueError("No approved training-split instruction records")

    train_examples = [
        _encode_instruction_record(record, tokenizer, maximum_length)
        for record in train_records
    ]
    validation_examples = [
        _encode_instruction_record(record, tokenizer, maximum_length)
        for record in validation_records
    ]

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Model has no trainable parameters")

    device = torch.device(device_name)
    model.to(device)
    optimizer = torch.optim.AdamW(parameters, lr=float(learning_rate))

    dataset_hash = _dataset_fingerprint(dataset)
    tokenizer_hash = _tokenizer_fingerprint(tokenizer)
    config_hash = _model_config_fingerprint(config)
    training_settings = {
        "learning_rate": float(learning_rate),
        "maximum_length": maximum_length,
        "batch_size": batch_size,
        "seed": seed,
        "dataset_fingerprint": dataset_hash,
        "tokenizer_fingerprint": tokenizer_hash,
        "model_config_fingerprint": config_hash,
    }

    batches_per_epoch = math.ceil(len(train_examples) / batch_size)

    start_epoch = 0
    global_step = 0
    response_tokens_seen = 0

    if resume_from_checkpoint is None:
        set_seed(seed)
    else:
        from aviy.training.checkpoints import load_verified_checkpoint

        inspected = load_verified_checkpoint(resume_from_checkpoint, map_location="cpu")
        metadata = inspected.get("metadata")
        if not isinstance(metadata, dict) or not isinstance(metadata.get("sft"), dict):
            raise ValueError("Checkpoint does not contain SFT resume metadata")

        saved_settings = metadata["sft"].get("training_settings")
        if saved_settings != training_settings:
            raise ValueError("Checkpoint is incompatible with this dataset or training configuration")

        completed = metadata["sft"].get("completed_epochs")
        if (
            isinstance(completed, bool)
            or not isinstance(completed, int)
            or completed < 1
            or completed >= epochs
        ):
            raise ValueError("Checkpoint completed epoch count is incompatible with target epochs")

        saved_step = inspected.get("step")
        expected_step = completed * batches_per_epoch
        if (
            isinstance(saved_step, bool)
            or not isinstance(saved_step, int)
            or saved_step != expected_step
        ):
            raise ValueError("Checkpoint optimization step count is inconsistent with completed epochs")

        if inspected.get("seed") != seed:
            raise ValueError("Checkpoint seed does not match requested seed")

        saved_response_tokens = metadata["sft"].get("response_tokens_seen")
        if (
            isinstance(saved_response_tokens, bool)
            or not isinstance(saved_response_tokens, int)
            or saved_response_tokens <= 0
        ):
            raise ValueError("Checkpoint response-token count is invalid")
        response_tokens_seen = saved_response_tokens

        restored = load_checkpoint(
            resume_from_checkpoint,
            model=model,
            optimizer=optimizer,
            map_location=device,
        )
        start_epoch = completed
        global_step = restored["step"]

    losses: list[float] = []
    validation_losses: list[float] = []
    original_training_mode = model.training

    try:
        for epoch in range(start_epoch, epochs):
            indices = list(range(len(train_examples)))
            random.Random(seed + epoch).shuffle(indices)
            model.train()

            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                batch = [train_examples[index] for index in batch_indices]
                inputs, targets, token_count = _collate_instruction_records(
                    batch, device
                )

                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)

                if (
                    not isinstance(logits, torch.Tensor)
                    or logits.ndim != 3
                    or logits.shape[:2] != targets.shape
                    or not logits.is_floating_point()
                ):
                    raise ValueError("Model must return floating-point logits shaped [B,T,V]")
                if torch.any(targets[targets != -100] >= logits.shape[-1]):
                    raise ValueError("Training target exceeds model vocabulary size")

                loss = causal_language_model_loss(logits, targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite instruction-tuning loss")

                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(parameters, 1.0)
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError("Non-finite instruction gradient norm")
                optimizer.step()

                losses.append(float(loss.detach().cpu()))
                response_tokens_seen += token_count
                global_step += 1

            model.eval()
            with torch.no_grad():
                validation_loss = _evaluate_instruction_loss(
                    model,
                    validation_examples,
                    batch_size=batch_size,
                    device=device,
                )
            if validation_loss is not None:
                validation_losses.append(validation_loss)

            if checkpoint_path is not None:
                save_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    step=global_step,
                    seed=seed,
                    metadata={
                        "sft": {
                            "completed_epochs": epoch + 1,
                            "training_settings": training_settings,
                            "response_tokens_seen": response_tokens_seen,
                        }
                    },
                )
    finally:
        model.train(original_training_mode)

    if not losses and start_epoch == 0:
        raise ValueError("No usable instruction examples were produced")

    return {
        "epochs": epochs,
        "completed_epochs": epochs,
        "resumed_from_epoch": start_epoch,
        "training_examples": len(train_examples),
        "optimization_steps": len(losses),
        "global_step": global_step,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "losses": losses,
        "validation_examples": len(validation_examples),
        "validation_losses": validation_losses,
        "final_validation_loss": (
            validation_losses[-1] if validation_losses else None
        ),
        "response_tokens_seen": response_tokens_seen,
        "response_only_loss": True,
        "batch_size": batch_size,
        "maximum_length": maximum_length,
        "device": str(device),
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
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
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
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
                or not math.isfinite(preference)
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
            if preference_scores
            else None
        ),
        "passed": (
            instruction_mean >= minimum_instruction_score
            and safety_mean >= minimum_safety_score
        ),
    }
