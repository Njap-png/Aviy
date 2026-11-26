from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from aviy.corpus_governance.dataset_batch_evaluator import run_batch
from aviy.corpus_governance.dataset_evaluator import (
    evaluate_source_metadata,
    sha256_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORAGE = PROJECT_ROOT / "data" / "ingestion"

MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_FILENAME_LENGTH = 128
MAX_LOCK_WAIT_SECONDS = 30.0

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]{0,127}$")

SUPPORTED_FORMATS = {"jsonl", "json", "csv", "txt"}
REQUIRED_INTAKE_METADATA = ("language", "script")

_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def _thread_lock_for(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.Lock())


@contextmanager
def _storage_lock(storage: Path) -> Iterator[None]:
    """Coordinate storage access across threads and local processes."""
    storage.mkdir(parents=True, exist_ok=True)

    lock_path = storage / ".intake.lock"
    thread_lock = _thread_lock_for(lock_path)

    if not thread_lock.acquire(timeout=MAX_LOCK_WAIT_SECONDS):
        raise TimeoutError("Timed out waiting for the ingestion storage lock.")

    handle = None
    locked = False

    try:
        handle = lock_path.open("a+b")
        handle.seek(0, os.SEEK_END)

        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()

        handle.seek(0)
        deadline = time.monotonic() + MAX_LOCK_WAIT_SECONDS

        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    handle.seek(0)
                    msvcrt.locking(
                        handle.fileno(),
                        msvcrt.LK_NBLCK,
                        1,
                    )
                else:
                    import fcntl
                    fcntl.flock(
                        handle.fileno(),
                        fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )

                locked = True
                break

            except (OSError, BlockingIOError):
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Timed out waiting for the ingestion storage lock."
                    )
                time.sleep(0.05)

        yield

    finally:
        if handle is not None:
            try:
                if locked:
                    if os.name == "nt":
                        import msvcrt
                        handle.seek(0)
                        msvcrt.locking(
                            handle.fileno(),
                            msvcrt.LK_UNLCK,
                            1,
                        )
                    else:
                        import fcntl
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()

        thread_lock.release()


def _atomic_json(path: Path, value: object) -> None:
    """Write JSON through a unique same-directory temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

        os.replace(temporary_name, path)

    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _read_history_unlocked(storage: Path) -> list[dict[str, Any]]:
    path = storage / "history.jsonl"

    if not path.is_file():
        return []

    results = []

    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            if isinstance(item, dict):
                results.append(item)

    return results


def read_history(
    storage_root: str | Path = DEFAULT_STORAGE,
) -> list[dict]:
    storage = Path(storage_root).expanduser().resolve()

    if not storage.exists():
        return []

    with _storage_lock(storage):
        return _read_history_unlocked(storage)


def _format_for(filename: str, supplied: str | None) -> str:
    value = (supplied or Path(filename).suffix.lstrip(".")).lower()

    if value not in SUPPORTED_FORMATS:
        raise ValueError(
            "Unsupported format. Choose JSONL, JSON, CSV, or TXT."
        )

    return value


def _validate_filename(filename: object) -> str:
    if not isinstance(filename, str) or not filename:
        raise ValueError("A filename is required.")

    if len(filename) > MAX_FILENAME_LENGTH:
        raise ValueError("Filename exceeds the permitted length.")

    if filename != filename.strip():
        raise ValueError("Filename cannot start or end with whitespace.")

    if any(
        ord(character) < 32 or ord(character) == 127
        for character in filename
    ):
        raise ValueError("Filename contains control characters.")

    if any(separator in filename for separator in ("/", "\\", ":", "\0")):
        raise ValueError("Filename must not contain a directory path or drive.")

    if filename in {".", ".."}:
        raise ValueError("Invalid filename.")

    if PurePosixPath(filename).name != filename:
        raise ValueError("Filename must not contain a directory path.")

    windows_path = PureWindowsPath(filename)

    if windows_path.name != filename or windows_path.drive:
        raise ValueError("Filename must not contain a Windows path or drive.")

    if not FILENAME_RE.fullmatch(filename):
        raise ValueError("Filename contains unsupported characters.")

    if filename.endswith((".", " ")):
        raise ValueError("Filename cannot end with a dot or space.")

    reserved = {"CON", "PRN", "AUX", "NUL"}
    reserved.update(f"COM{i}" for i in range(1, 10))
    reserved.update(f"LPT{i}" for i in range(1, 10))

    stem = filename.split(".", 1)[0].upper()

    if stem in reserved:
        raise ValueError("Filename uses a reserved Windows device name.")

    return filename


def _validate_metadata(metadata: object) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be a JSON object.")  # noqa: TRY004

    result = dict(metadata)
    issues = evaluate_source_metadata(result)

    for field in REQUIRED_INTAKE_METADATA:
        value = result.get(field)

        if not isinstance(value, str) or not value.strip():
            issues.append(
                f"Missing or invalid intake metadata field: {field}"
            )

    if issues:
        raise ValueError(
            "Metadata validation failed: " + "; ".join(issues)
        )

    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(
        f"Non-standard JSON constant is not permitted: {value}"
    )


def _loads_json(value: str | bytes) -> Any:
    return json.loads(
        value,
        parse_constant=_reject_json_constant,
    )


def _parse_records(
    content: bytes,
    file_format: str,
) -> list[dict[str, Any]]:
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
                item = _loads_json(line)
            except (ValueError, RecursionError) as exc:
                raise ValueError(
                    f"Invalid JSONL at line {number}: {exc}"
                ) from exc

            if not isinstance(item, dict):
                raise ValueError(  # noqa: TRY004
                    f"JSONL line {number} must be an object."
                )

            records.append(item)

    elif file_format == "json":
        try:
            value = _loads_json(text)
        except (ValueError, RecursionError) as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc

        if isinstance(value, dict) and isinstance(
            value.get("records"), list
        ):
            value = value["records"]
        elif isinstance(value, dict):
            value = [value]

        if not isinstance(value, list) or not value:
            raise ValueError(
                "JSON must contain a non-empty records array."
            )

        if any(not isinstance(item, dict) for item in value):
            raise ValueError("Every JSON record must be an object.")

        records = value

    elif file_format == "csv":
        try:
            reader = csv.DictReader(
                io.StringIO(text, newline=""),
                strict=True,
            )

            headers = reader.fieldnames

            if not headers or "text" not in headers:
                raise ValueError(
                    "CSV must have a header row containing 'text'."
                )

            if any(
                not isinstance(header, str) or not header.strip()
                for header in headers
            ):
                raise ValueError(
                    "CSV headers must be non-empty strings."
                )

            if len(headers) != len(set(headers)):
                raise ValueError(
                    "CSV contains duplicate column names."
                )

            for number, item in enumerate(reader, 2):
                if None in item:
                    raise ValueError(
                        f"CSV row {number} contains extra columns."
                    )

                if any(value is None for value in item.values()):
                    raise ValueError(
                        f"CSV row {number} has missing column values."
                    )

                if not any(value for value in item.values()):
                    continue

                records.append(dict(item))

        except csv.Error as exc:
            raise ValueError(f"Malformed CSV: {exc}") from exc

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


def _submission_fingerprint(
    filename: str,
    file_format: str,
    dataset_name: str,
    digest: str,
    metadata: dict[str, Any],
) -> str:
    canonical = json.dumps(
        {
            "filename": filename,
            "format": file_format,
            "name": dataset_name,
            "input_sha256": digest,
            "metadata": metadata,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _append_history(
    path: Path,
    result: dict[str, Any],
) -> None:
    encoded = (
        json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")

    with path.open("ab") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def intake_bytes(
    filename: str,
    content: bytes,
    metadata: dict[str, Any],
    name: str | None = None,
    file_format: str | None = None,
    storage_root: str | Path = DEFAULT_STORAGE,
) -> dict[str, Any]:
    filename = _validate_filename(filename)

    if not isinstance(content, bytes) or not content:
        raise ValueError("Input file is empty.")

    if len(content) > MAX_FILE_BYTES:
        raise ValueError("Input exceeds the 50 MiB file size limit.")

    dataset_name = name or Path(filename).stem

    if not isinstance(dataset_name, str):
        raise ValueError("Dataset name must be a string.")  # noqa: TRY004

    if not NAME_RE.fullmatch(dataset_name):
        raise ValueError(
            "Dataset name must use 1-64 letters, digits, '_' or '-'."
        )

    if file_format is not None and not isinstance(file_format, str):
        raise ValueError("Format must be a string.")

    fmt = _format_for(filename, file_format)
    checked_metadata = _validate_metadata(metadata)

    digest = hashlib.sha256(content).hexdigest()

    parsed = _parse_records(content, fmt)
    records = _normalize_records(
        parsed,
        checked_metadata,
        digest,
    )

    fingerprint = _submission_fingerprint(
        filename,
        fmt,
        dataset_name,
        digest,
        checked_metadata,
    )

    storage = Path(storage_root).expanduser().resolve()

    # Serialize duplicate detection, evaluation and history publication.
    with _storage_lock(storage):
        for previous in _read_history_unlocked(storage):
            if previous.get("submission_fingerprint") == fingerprint:
                duplicate = dict(previous)
                duplicate["duplicate_submission"] = True
                duplicate["reused_existing_ingestion"] = True
                return duplicate

        run_id = str(uuid.uuid4())
        run_dir = storage / "runs" / run_id
        input_dir = run_dir / "input"

        input_dir.mkdir(parents=True, exist_ok=False)

        try:
            original_path = input_dir / filename
            original_path.write_bytes(content)

            records_path = input_dir / "normalized_records.jsonl"

            with records_path.open(
                "w",
                encoding="utf-8",
                newline="\n",
            ) as stream:
                for record in records:
                    stream.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )

            metadata_path = input_dir / "source_metadata.json"
            _atomic_json(metadata_path, checked_metadata)

            manifest_path = run_dir / "batch_manifest.json"

            _atomic_json(
                manifest_path,
                {
                    "schema_version": "1.0",
                    "datasets": [
                        {
                            "name": dataset_name,
                            "records": str(records_path),
                            "metadata": str(metadata_path),
                        }
                    ],
                },
            )

            evaluation_path = run_dir / "evaluation"

            try:
                batch_report = run_batch(
                    manifest_path,
                    evaluation_path,
                )
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
                "normalized_records_sha256": sha256_file(
                    records_path
                ),
                "submission_fingerprint": fingerprint,
                "records_received": len(records),
                "status": status,
                "error": error,
                "created_at_utc": datetime.now(UTC).isoformat(),
                "original_file_preserved": True,
                "evaluation_report": (
                    str(evaluation_path / "batch_report.json")
                    if batch_report is not None
                    else None
                ),
                "production_source_registry_changed": False,
                "records_approved_for_training": False,
                "tokenizer_training_authorized": False,
                "pretraining_authorized": False,
                "training_started": False,
                "duplicate_submission": False,
                "reused_existing_ingestion": False,
            }

            _atomic_json(
                run_dir / "ingestion_report.json",
                result,
            )

            _append_history(
                storage / "history.jsonl",
                result,
            )

            return result

        except BaseException:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise


def intake_file(
    file_path: str | Path,
    metadata: dict[str, Any],
    name: str | None = None,
    file_format: str | None = None,
    storage_root: str | Path = DEFAULT_STORAGE,
) -> dict[str, Any]:
    path = Path(file_path).expanduser()

    if not path.is_file():
        raise ValueError("Input file does not exist.")

    # Read at most one byte beyond the limit. Avoid an unbounded read.
    with path.open("rb") as stream:
        content = stream.read(MAX_FILE_BYTES + 1)

    if len(content) > MAX_FILE_BYTES:
        raise ValueError("Input exceeds the 50 MiB file size limit.")

    return intake_bytes(
        path.name,
        content,
        metadata,
        name=name,
        file_format=file_format,
        storage_root=storage_root,
    )
