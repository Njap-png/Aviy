from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Literal

TrainingStage = Literal["tokenizer_training", "supervised_finetuning", "base_pretraining"]


@dataclass(frozen=True)
class TrainingAuthorization:
    """Explicit, dataset-bound authorization for a specific training stage."""

    stage: TrainingStage
    dataset_fingerprint: str
    approver: str
    authorized_at: str
    evidence_reference: str

    def __post_init__(self) -> None:
        if self.stage not in ("tokenizer_training", "supervised_finetuning", "base_pretraining"):
            raise ValueError("Unsupported training authorization stage")

        if not re.fullmatch(r"[0-9a-f]{64}", self.dataset_fingerprint):
            raise ValueError("dataset_fingerprint must be a lowercase SHA-256 digest")

        for field_name in ("approver", "evidence_reference"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")

        if not isinstance(self.authorized_at, str) or not self.authorized_at.strip():
            raise ValueError("authorized_at must be a non-empty ISO-8601 timestamp")

        try:
            timestamp = datetime.fromisoformat(self.authorized_at)
        except ValueError as exc:
            raise ValueError("authorized_at must be a valid ISO-8601 timestamp") from exc

        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("authorized_at must include a timezone")


def fingerprint_prepared_dataset(dataset: object) -> str:
    """Fingerprint all prepared dataset records and audit metadata."""
    records = getattr(dataset, "records", None)
    audit = getattr(dataset, "audit", None)
    if records is None or audit is None:
        raise TypeError("Expected a prepared dataset with records and audit metadata")

    try:
        payload = {
            "records": [asdict(record) for record in records],
            "audit": asdict(audit),
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Prepared dataset cannot be fingerprinted") from exc

    return hashlib.sha256(encoded).hexdigest()


def require_training_authorization(
    authorization: object | None,
    *,
    expected_stage: TrainingStage,
    expected_fingerprint: str,
) -> TrainingAuthorization:
    """Reject absent, malformed, wrong-stage, or wrong-dataset authorization."""
    if not isinstance(authorization, TrainingAuthorization):
        raise PermissionError("Explicit TrainingAuthorization is required")

    if authorization.stage != expected_stage:
        raise PermissionError("Training authorization is for a different stage")

    if authorization.dataset_fingerprint != expected_fingerprint:
        raise PermissionError("Training authorization does not match this dataset")

    return authorization
