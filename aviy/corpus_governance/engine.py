"""Independent governance gates for the Aviy Original Corpus."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import unicodedata
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit


class GovernanceError(ValueError):
    """Raised when corpus governance requirements are not satisfied."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("Record text must be a string")
    return " ".join(unicodedata.normalize("NFC", text).split())


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def privacy_issues(text: str) -> tuple[str, ...]:
    """Conservative pattern screening; this is not a complete PII detector."""
    patterns = {
        "email_address": r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        "possible_secret": (
            r"(?i)\b(api[_-]?key|password|passwd|secret|access[_-]?token)"
            r"\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{8,}"
        ),
        "private_key": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        "phone_number": (
            r"(?<!\w)(?:\+\d{1,3}[\s.-]?)?"
            r"(?:\(?\d{2,4}\)?[\s.-])\d{3,4}[\s.-]\d{3,4}(?!\w)"
        ),
    }
    return tuple(
        name for name, pattern in patterns.items()
        if re.search(pattern, text)
    )


@dataclass(frozen=True)
class SourceRegistration:
    source_id: str
    source_name: str
    source_uri: str
    revision: str
    license_id: str
    license_status: str = "pending"
    approval_status: str = "pending"
    training_allowed: bool = False
    attribution_required: bool = True
    attribution_text: str = ""
    reviewer: str = ""
    approved_at: str = ""


@dataclass(frozen=True)
class CorpusRecord:
    record_id: str
    text: str
    language: str
    script: str
    category: str
    source_type: str
    source_id: str
    source_name: str
    source_uri: str
    source_revision: str
    license_id: str
    attribution: str
    group_id: str
    authored_at: str
    content_sha256: str
    review_status: str = "pending_review"
    privacy_status: str = "pending"
    approval_status: str = "pending"
    reviewer: str = ""
    reviewed_at: str = ""
    review_notes: str = ""


class SourceRegistry:
    """Explicit source-level licensing and permission registry."""

    def __init__(self, sources=()):
        self._sources = {}
        for source in sources:
            self.add(source)

    def add(self, source: SourceRegistration) -> None:
        if not isinstance(source, SourceRegistration):
            raise TypeError("source must be SourceRegistration")
        if not source.source_id.strip():
            raise GovernanceError("source_id is required")
        if source.approval_status not in {"pending", "approved", "rejected"}:
            raise GovernanceError("Invalid source approval_status")
        if source.license_status not in {"pending", "approved", "rejected"}:
            raise GovernanceError("Invalid source license_status")
        if source.source_id in self._sources:
            raise GovernanceError("Duplicate source_id")
        self._sources[source.source_id] = source

    def get(self, source_id: str):
        return self._sources.get(source_id)

    def is_training_approved(self, source_id: str) -> bool:
        source = self.get(source_id)
        return bool(
            source
            and source.approval_status == "approved"
            and source.license_status == "approved"
            and source.training_allowed is True
            and source.reviewer.strip()
            and source.approved_at.strip()
        )

    def to_dict(self) -> dict:
        return {
            "schema_version": "1.0",
            "sources": [
                asdict(self._sources[key]) for key in sorted(self._sources)
            ],
        }

    @classmethod
    def from_dict(cls, value: dict) -> "SourceRegistry":
        if not isinstance(value, dict) or value.get("schema_version") != "1.0":
            raise GovernanceError("Unsupported source registry schema")
        entries = value.get("sources")
        if not isinstance(entries, list):
            raise GovernanceError("Source registry sources must be a list")
        try:
            return cls(SourceRegistration(**entry) for entry in entries)
        except (TypeError, ValueError) as exc:
            raise GovernanceError("Invalid source registry entry") from exc

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n"
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8", newline="\n")
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path) -> "SourceRegistry":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(value)


def create_record(
    *,
    record_id: str,
    text: str,
    language: str,
    script: str,
    category: str,
    source_type: str,
    source_id: str,
    source_name: str,
    source_uri: str,
    source_revision: str,
    license_id: str,
    attribution: str,
    group_id: str = "",
    authored_at: str = "",
) -> CorpusRecord:
    return CorpusRecord(
        record_id=record_id,
        text=text,
        language=language,
        script=script,
        category=category,
        source_type=source_type,
        source_id=source_id,
        source_name=source_name,
        source_uri=source_uri,
        source_revision=source_revision,
        license_id=license_id,
        attribution=attribution,
        group_id=group_id or record_id,
        authored_at=authored_at or utc_now(),
        content_sha256=content_hash(text),
    )


def _valid_uri(value: str) -> bool:
    if not isinstance(value, str) or not value or value != value.strip():
        return False
    if any(char.isspace() for char in value):
        return False
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    if not parsed.scheme:
        return False
    if parsed.scheme.lower() in {"http", "https"}:
        return bool(parsed.netloc and parsed.hostname)
    return parsed.scheme.lower() in {"aviy", "dataset", "doi"}


def validate_record(
    record: CorpusRecord,
    registry: SourceRegistry,
    *,
    minimum_characters: int = 20,
) -> tuple[str, ...]:
    """Return governance issues without approving or modifying a record."""
    if not isinstance(record, CorpusRecord):
        raise TypeError("record must be CorpusRecord")
    if not isinstance(registry, SourceRegistry):
        raise TypeError("registry must be SourceRegistry")

    issues = []
    required = {
        "record_id": record.record_id,
        "text": record.text,
        "language": record.language,
        "script": record.script,
        "category": record.category,
        "source_type": record.source_type,
        "source_id": record.source_id,
        "source_name": record.source_name,
        "source_uri": record.source_uri,
        "source_revision": record.source_revision,
        "license_id": record.license_id,
        "group_id": record.group_id,
        "authored_at": record.authored_at,
    }
    for name, value in required.items():
        if not isinstance(value, str) or not value.strip():
            issues.append("missing_" + name)

    if not isinstance(record.text, str) or not record.text.strip():
        issues.append("empty_text")
    elif len(normalize_text(record.text)) < minimum_characters:
        issues.append("text_too_short")
    elif any(unicodedata.category(c) == "Cc" for c in record.text):
        issues.append("control_characters")

    if isinstance(record.text, str):
        if record.content_sha256 != content_hash(record.text):
            issues.append("content_hash_mismatch")
        issues.extend(privacy_issues(record.text))

    if record.source_uri and not _valid_uri(record.source_uri):
        issues.append("invalid_source_uri")

    source = registry.get(record.source_id)
    if source is None:
        issues.append("unregistered_source")
    else:
        if not registry.is_training_approved(record.source_id):
            issues.append("source_not_approved_for_training")
        if record.license_id.strip().casefold() != source.license_id.strip().casefold():
            issues.append("record_source_license_mismatch")
        if record.source_name.strip() != source.source_name.strip():
            issues.append("record_source_name_mismatch")
        if source.attribution_required and not record.attribution.strip():
            issues.append("missing_required_attribution")

    return tuple(dict.fromkeys(issues))


def approve_record(
    record: CorpusRecord,
    registry: SourceRegistry,
    *,
    reviewer: str,
    human_reviewed: bool,
    review_notes: str = "",
) -> CorpusRecord:
    """Approve only after explicit human review and zero governance issues."""
    if human_reviewed is not True:
        raise GovernanceError("Explicit human review is required")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise GovernanceError("A reviewer identity is required")
    if record.approval_status in {"rejected", "quarantined"}:
        raise GovernanceError("Rejected or quarantined records cannot be approved")
    issues = validate_record(record, registry)
    if issues:
        raise GovernanceError("Record failed governance: " + ", ".join(issues))
    return replace(
        record,
        review_status="reviewed",
        privacy_status="clear",
        approval_status="approved",
        reviewer=reviewer.strip(),
        reviewed_at=utc_now(),
        review_notes=review_notes,
    )


def quarantine_record(
    record: CorpusRecord,
    *,
    reason: str,
    reviewer: str = "",
) -> CorpusRecord:
    if not isinstance(reason, str) or not reason.strip():
        raise GovernanceError("A quarantine reason is required")
    return replace(
        record,
        review_status="human_review_required",
        privacy_status=(
            "flagged" if privacy_issues(record.text) else record.privacy_status
        ),
        approval_status="quarantined",
        reviewer=reviewer.strip(),
        reviewed_at=utc_now(),
        review_notes=reason.strip(),
    )


def _split_for(group_id: str, seed: int, ratios: tuple[float, float, float]) -> str:
    digest = hashlib.sha256(
        f"{seed}:{group_id}".encode("utf-8")
    ).hexdigest()
    fraction = int(digest, 16) / (1 << 256)
    train, validation, _test = ratios
    if fraction < train:
        return "train"
    if fraction < train + validation:
        return "validation"
    return "test"


def build_manifest(
    records,
    registry: SourceRegistry,
    *,
    seed: int = 42,
    train_ratio: float = 0.90,
    validation_ratio: float = 0.05,
    test_ratio: float = 0.05,
) -> dict:
    """Build metadata-only split manifest from explicitly approved records."""
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 <= value <= 1
        for value in ratios
    ):
        raise GovernanceError("Split ratios must be between zero and one")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise GovernanceError("Split ratios must sum to 1")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise GovernanceError("seed must be a non-negative integer")

    items = list(records)
    if not items:
        raise GovernanceError("Cannot build a manifest from zero records")

    ids = set()
    hashes = set()
    entries = []
    group_splits = {}

    for record in items:
        if record.record_id in ids:
            raise GovernanceError("Duplicate record_id")
        ids.add(record.record_id)

        if record.approval_status != "approved":
            raise GovernanceError("Manifest blocked: record is not approved")
        if record.review_status != "reviewed" or record.privacy_status != "clear":
            raise GovernanceError("Manifest blocked: review or privacy gate incomplete")
        if not record.reviewer.strip() or not record.reviewed_at.strip():
            raise GovernanceError("Manifest blocked: missing review evidence")

        issues = validate_record(record, registry)
        if issues:
            raise GovernanceError(
                "Manifest blocked: " + ", ".join(issues)
            )

        fingerprint = content_hash(record.text)
        if fingerprint in hashes:
            raise GovernanceError("Duplicate content cannot enter manifest")
        hashes.add(fingerprint)

        split = _split_for(record.group_id, seed, ratios)
        previous = group_splits.setdefault(record.group_id, split)
        if previous != split:
            raise GovernanceError("Related records crossed dataset splits")

        entries.append({
            "record_id": record.record_id,
            "content_sha256": fingerprint,
            "source_id": record.source_id,
            "language": record.language,
            "category": record.category,
            "group_id": record.group_id,
            "split": split,
        })

    entries.sort(key=lambda item: item["record_id"])
    counts = {
        name: sum(item["split"] == name for item in entries)
        for name in ("train", "validation", "test")
    }
    return {
        "schema_version": "1.0",
        "corpus": "aviy_original_corpus",
        "created_at": utc_now(),
        "seed": seed,
        "split_ratios": {
            "train": train_ratio,
            "validation": validation_ratio,
            "test": test_ratio,
        },
        "record_count": len(entries),
        "split_counts": counts,
        "records": entries,
        "training_authorized": True,
        "test_split_training_allowed": False,
        "test_split_model_selection_allowed": False,
    }


def training_records(manifest: dict, records):
    """Return only verified training records; never validation or test data."""
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "1.0":
        raise GovernanceError("Invalid manifest")
    if manifest.get("test_split_training_allowed") is not False:
        raise GovernanceError("Manifest does not protect the test split")

    items = list(records)
    record_map = {record.record_id: record for record in items}
    if len(record_map) != len(items):
        raise GovernanceError("Duplicate record IDs supplied")

    selected = []
    for entry in manifest.get("records", []):
        record = record_map.get(entry.get("record_id"))
        if record is None:
            raise GovernanceError("Manifest record is missing from supplied records")
        if content_hash(record.text) != entry.get("content_sha256"):
            raise GovernanceError("Training record hash mismatch")
        if record.approval_status != "approved":
            raise GovernanceError("Training record is no longer approved")
        if entry.get("split") == "train":
            selected.append(record)
    return tuple(selected)


def write_manifest(manifest: dict, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(destination)
