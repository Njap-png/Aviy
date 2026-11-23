"""Governed dataset ingestion, validation, deduplication and splitting."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit

_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE = re.compile(
    r"(?<!\w)(?:\+\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-])"
    r"\d{3,4}[\s.-]\d{3,4}(?!\w)"
)
_SECRET = re.compile(
    r"(?i)\b(api[_-]?key|password|passwd|secret|access[_-]?token)"
    r"\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{8,}"
)
_PRIVATE_KEY = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")


@dataclass(frozen=True)
class DatasetRecord:
    """A training example with explicit source and approval metadata."""

    text: str
    source_id: str
    license: str
    approved_for_training: bool
    source_uri: str = ""
    record_id: str = ""
    split: str = ""


@dataclass(frozen=True)
class DatasetAudit:
    """Machine-readable summary of a prepared dataset."""

    accepted: int
    rejected: int
    duplicates_removed: int
    split_counts: dict[str, int]
    provenance_complete: bool
    licenses_approved: bool
    privacy_scan_passed: bool
    split_integrity_passed: bool
    seed: int

    @property
    def passed(self) -> bool:
        return all((
            self.accepted > 0,
            self.provenance_complete,
            self.licenses_approved,
            self.privacy_scan_passed,
            self.split_integrity_passed,
        ))

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["passed"] = self.passed
        return result


@dataclass(frozen=True)
class PreparedDataset:
    records: tuple[DatasetRecord, ...]
    audit: DatasetAudit


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace consistently."""
    if not isinstance(text, str):
        raise TypeError("Record text must be a string")
    return " ".join(unicodedata.normalize("NFC", text).split())


def _valid_source_uri(value: str) -> bool:
    """Require a syntactically meaningful URI with a scheme."""
    if not value or value != value.strip() or any(char.isspace() for char in value):
        return False
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    if not parsed.scheme:
        return False
    if parsed.scheme.lower() in {"http", "https"}:
        return bool(parsed.netloc and parsed.hostname)
    return bool(parsed.netloc or parsed.path)


def privacy_issues(text: str) -> tuple[str, ...]:
    """Detect common personal contact details and credential patterns."""
    issues = []
    if _EMAIL.search(text):
        issues.append("email_address")
    if _PHONE.search(text):
        issues.append("phone_number")
    if _SECRET.search(text):
        issues.append("possible_secret")
    if _PRIVATE_KEY.search(text):
        issues.append("private_key")
    return tuple(issues)


def _record_from_mapping(item: dict[str, object]) -> DatasetRecord:
    required = ("text", "source_id", "license", "approved_for_training")
    missing = [name for name in required if name not in item]
    if missing:
        raise ValueError(f"Missing required record fields: {', '.join(missing)}")

    text = item["text"]
    source_id = item["source_id"]
    license_name = item["license"]
    approved = item["approved_for_training"]
    source_uri = item.get("source_uri", "")
    record_id = item.get("record_id", "")

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(source_id, str):
        raise TypeError("source_id must be a string")
    if not isinstance(license_name, str):
        raise TypeError("license must be a string")
    if not isinstance(approved, bool):
        raise TypeError("approved_for_training must be a boolean")
    if not isinstance(source_uri, str) or not isinstance(record_id, str):
        raise TypeError("source_uri and record_id must be strings")

    return DatasetRecord(
        text=text,
        source_id=source_id,
        license=license_name,
        approved_for_training=approved,
        source_uri=source_uri,
        record_id=record_id,
    )


def read_jsonl(path: str | Path) -> list[DatasetRecord]:
    """Read JSON Lines records without silently skipping malformed entries."""
    records = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise TypeError("Each JSONL line must be an object")
                records.append(_record_from_mapping(item))
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                raise ValueError(f"Invalid JSONL record at line {line_number}: {exc}") from exc
    return records


def write_jsonl(records: Iterable[DatasetRecord], path: str | Path) -> None:
    """Write records as UTF-8 JSON Lines."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def prepare_dataset(
    records: Iterable[DatasetRecord | dict[str, object]],
    *,
    allowed_licenses: Iterable[str],
    seed: int = 42,
    train_ratio: float = 0.90,
    validation_ratio: float = 0.05,
    test_ratio: float = 0.05,
    minimum_characters: int = 1,
    maximum_characters: int = 100_000,
) -> PreparedDataset:
    """Validate, deduplicate and deterministically split approved records."""
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(value < 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("Split ratios must be non-negative and sum to 1")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if minimum_characters < 1 or maximum_characters < minimum_characters:
        raise ValueError("Invalid character limits")

    approved = {value.strip().lower() for value in allowed_licenses if value.strip()}
    if not approved:
        raise ValueError("At least one explicitly approved license is required")

    accepted: list[DatasetRecord] = []
    rejected = 0
    duplicate_count = 0
    seen: set[str] = set()
    provenance_complete = True
    licenses_approved = True
    privacy_passed = True

    for raw in records:
        try:
            record = raw if isinstance(raw, DatasetRecord) else _record_from_mapping(raw)
            if not isinstance(record.text, str):
                raise TypeError("text must be a string")
            if not isinstance(record.source_id, str):
                raise TypeError("source_id must be a string")
            if not isinstance(record.license, str):
                raise TypeError("license must be a string")
            if not isinstance(record.source_uri, str):
                raise TypeError("source_uri must be a string")
            if not isinstance(record.approved_for_training, bool):
                raise TypeError("approved_for_training must be a boolean")
            text = normalize_text(record.text)
        except (ValueError, TypeError):
            rejected += 1
            provenance_complete = False
            continue

        if (
            not record.source_id.strip()
            or not record.license.strip()
            or not _valid_source_uri(record.source_uri)
        ):
            provenance_complete = False
            rejected += 1
            continue
        if record.license.strip().lower() not in approved:
            licenses_approved = False
            rejected += 1
            continue
        if record.approved_for_training is not True:
            rejected += 1
            continue
        if not text or not minimum_characters <= len(text) <= maximum_characters:
            rejected += 1
            continue
        if any(unicodedata.category(char) == "Cc" for char in text):
            rejected += 1
            continue
        if privacy_issues(text):
            privacy_passed = False
            rejected += 1
            continue

        fingerprint = hashlib.sha256(text.casefold().encode("utf-8")).hexdigest()
        if fingerprint in seen:
            duplicate_count += 1
            continue
        seen.add(fingerprint)

        accepted.append(DatasetRecord(
            text=text,
            source_id=record.source_id.strip(),
            license=record.license.strip(),
            approved_for_training=True,
            source_uri=record.source_uri.strip(),
            record_id=record.record_id.strip() or fingerprint,
        ))

    def split_for(record: DatasetRecord) -> str:
        fingerprint = hashlib.sha256(record.text.casefold().encode("utf-8")).hexdigest()
        value = int(hashlib.sha256(f"{seed}:{fingerprint}".encode()).hexdigest()[:16], 16)
        fraction = value / 0xFFFFFFFFFFFFFFFF
        if fraction < train_ratio:
            return "train"
        if fraction < train_ratio + validation_ratio:
            return "validation"
        return "test"

    split_records = tuple(
        DatasetRecord(**{**asdict(record), "split": split_for(record)})
        for record in accepted
    )

    split_texts = {
        name: {
            hashlib.sha256(record.text.casefold().encode("utf-8")).hexdigest()
            for record in split_records if record.split == name
        }
        for name in ("train", "validation", "test")
    }
    split_integrity = not (
        split_texts["train"] & split_texts["validation"]
        or split_texts["train"] & split_texts["test"]
        or split_texts["validation"] & split_texts["test"]
    )

    counts = {
        name: sum(record.split == name for record in split_records)
        for name in ("train", "validation", "test")
    }

    audit = DatasetAudit(
        accepted=len(split_records),
        rejected=rejected,
        duplicates_removed=duplicate_count,
        split_counts=counts,
        provenance_complete=provenance_complete,
        licenses_approved=licenses_approved,
        privacy_scan_passed=privacy_passed,
        split_integrity_passed=split_integrity,
        seed=seed,
    )
    return PreparedDataset(records=split_records, audit=audit)
