from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aviy.corpus_governance.dataset_batch_evaluator import run_batch
from aviy.corpus_governance.dataset_evaluator import (
    evaluate_source_metadata,
    sha256_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORAGE = PROJECT_ROOT / "data" / "ingestion"
MAX_FILE_BYTES = 50 * 1024 * 1024
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
SUPPORTED_FORMATS = {"jsonl", "json", "csv", "txt"}
REQUIRED_INTAKE_METADATA = (
    "language",
    "script",
)


def _format_for(filename: str, supplied: str | None) -> str:
    value = (supplied or Path(filename).suffix.lstrip(".")).lower()
    if value not in SUPPORTED_FORMATS:
        raise ValueError(
            "Unsupported format. Choose JSONL, JSON, CSV, or TXT."
        )
    return value


def _validate_metadata(metadata: object) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be a JSON object.")  # noqa: TRY004

    result = dict(metadata)
    issues = evaluate_source_metadata(result)
    for field in REQUIRED_INTAKE_METADATA:
        value = result.get(field)
        if not isinstance(value, str) or not value.strip():
            issues.append(f"Missing or invalid intake metadata field: {field}")

    if issues:
        raise ValueError("Metadata validation failed: " + "; ".join(issues))
    return result


def _parse_records(content: bytes, file_format: str) -> list[dict[str, Any]]:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Input must be valid UTF-8 text.") from exc

    records: list[dict[str, Any]] = []

    if file_format == "jsonl":
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {number}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"JSONL line {number} must be an object.")  # noqa: TRY004
            records.append(item)

    elif file_format == "json":
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc
        if isinstance(value, dict) and isinstance(value.get("records"), list):
            value = value["records"]
        elif isinstance(value, dict):
            value = [value]
        if not isinstance(value, list) or not value:
            raise ValueError("JSON must contain a non-empty records array.")
        if any(not isinstance(item, dict) for item in value):
            raise ValueError("Every JSON record must be an object.")
        records = value

    elif file_format == "csv":
        reader = csv.DictReader(io.StringIO(text, newline=""))
        if not reader.fieldnames or "text" not in reader.fieldnames:
            raise ValueError("CSV must have a header row containing 'text'.")
        for number, item in enumerate(reader, 2):
            if not item or not any(value for value in item.values() if value):
                continue
            records.append(dict(item))
        if not records:
            raise ValueError("CSV contains no data records.")

    elif file_format == "txt":
        records = [
            {"text": line.strip()}
            for line in text.splitlines()
            if line.strip()
        ]
        if not records:
            raise ValueError("TXT contains no non-empty lines.")

    return records


def _normalize_records(
    records: list[dict[str, Any]],
    metadata: dict[str, Any],
    source_digest: str,
) -> list[dict[str, Any]]:
    normalized = []
    for index, original in enumerate(records, 1):
        record = dict(original)
        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Record {index} must contain a non-empty 'text' string."
            )

        # Missing provenance fields may be populated only from the metadata
        # explicitly supplied for this intake. Existing conflicting values
        # are preserved so the evaluator can quarantine them.
        record.setdefault(
            "record_id",
            f"ingest-{source_digest[:16]}-{index:08d}",
        )
        record.setdefault("language", metadata["language"])
        record.setdefault("script", metadata["script"])
        record.setdefault("source_id", metadata["source_id"])
        record.setdefault("source_uri", metadata["source_uri"])
        record.setdefault("source_revision", metadata["revision"])
        record.setdefault("license_id", metadata["license_id"])
        normalized.append(record)

    return normalized


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_history(storage_root: str | Path = DEFAULT_STORAGE) -> list[dict]:
    path = Path(storage_root) / "history.jsonl"
    if not path.is_file():
        return []
    results = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    results.append(item)
    return results


def intake_bytes(
    filename: str,
    content: bytes,
    metadata: dict[str, Any],
    name: str | None = None,
    file_format: str | None = None,
    storage_root: str | Path = DEFAULT_STORAGE,
) -> dict[str, Any]:
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("A filename is required.")
    if Path(filename).name != filename or filename in {".", ".."}:
        raise ValueError("Filename must not contain a directory path.")
    if not isinstance(content, bytes) or not content:
        raise ValueError("Input file is empty.")
    if len(content) > MAX_FILE_BYTES:
        raise ValueError("Input exceeds the 50 MiB file size limit.")

    dataset_name = name or Path(filename).stem
    if not isinstance(dataset_name, str) or not NAME_RE.fullmatch(dataset_name):
        raise ValueError("Dataset name must use 1-64 letters, digits, '_' or '-'.")
    fmt = _format_for(filename, file_format)
    checked_metadata = _validate_metadata(metadata)

    digest = hashlib.sha256(content).hexdigest()
    parsed = _parse_records(content, fmt)
    records = _normalize_records(parsed, checked_metadata, digest)

    storage = Path(storage_root).expanduser().resolve()
    run_id = str(uuid.uuid4())
    run_dir = storage / "runs" / run_id
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=False)

    original_path = input_dir / filename
    original_path.write_bytes(content)
    records_path = input_dir / "normalized_records.jsonl"
    with records_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    metadata_path = input_dir / "source_metadata.json"
    _atomic_json(metadata_path, checked_metadata)

    manifest_path = run_dir / "batch_manifest.json"
    _atomic_json(manifest_path, {
        "schema_version": "1.0",
        "datasets": [{
            "name": dataset_name,
            "records": str(records_path),
            "metadata": str(metadata_path),
        }],
    })

    evaluation_path = run_dir / "evaluation"
    try:
        batch_report = run_batch(manifest_path, evaluation_path)
        status = batch_report["status"]
        error = None
    except Exception as exc:  # noqa: BLE001
        status = "INGESTED_EVALUATION_FAILED"
        error = f"{type(exc).__name__}: {exc}"
        batch_report = None

    result = {
        "ingestion_id": run_id,
        "dataset_name": dataset_name,
        "original_filename": filename,
        "input_format": fmt,
        "input_sha256": digest,
        "normalized_records_sha256": sha256_file(records_path),
        "records_received": len(records),
        "status": status,
        "error": error,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "original_file_preserved": True,
        "evaluation_report": (
            str(evaluation_path / "batch_report.json")
            if batch_report is not None else None
        ),
        "production_source_registry_changed": False,
        "records_approved_for_training": False,
        "tokenizer_training_authorized": False,
        "pretraining_authorized": False,
        "training_started": False,
    }
    _atomic_json(run_dir / "ingestion_report.json", result)

    storage.mkdir(parents=True, exist_ok=True)
    history_path = storage / "history.jsonl"
    with history_path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, ensure_ascii=False) + "\n")

    return result


def intake_file(
    file_path: str | Path,
    metadata: dict[str, Any],
    name: str | None = None,
    file_format: str | None = None,
    storage_root: str | Path = DEFAULT_STORAGE,
) -> dict[str, Any]:
    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Input file does not exist: {path}")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("Input exceeds the 50 MiB file size limit.")
    return intake_bytes(
        path.name,
        path.read_bytes(),
        metadata,
        name=name,
        file_format=file_format,
        storage_root=storage_root,
    )
