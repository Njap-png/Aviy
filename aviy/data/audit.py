"""Reproducible dataset preparation and audit report generation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from aviy.data.pipeline import prepare_dataset, read_jsonl, write_jsonl


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_dataset(
    source: str | Path,
    output_directory: str | Path,
    *,
    allowed_licenses: Iterable[str],
    seed: int = 42,
) -> dict[str, object]:
    source_path = Path(source)
    source_hash = _sha256_file(source_path)
    records = read_jsonl(source_path)
    prepared = prepare_dataset(
        records,
        allowed_licenses=allowed_licenses,
        seed=seed,
    )
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)

    prepared_path = destination / "prepared.jsonl"
    write_jsonl(prepared.records, prepared_path)

    report = prepared.audit.to_dict()
    report.update(
        {
            "audit_schema": "aviy-dataset-audit-v2",
            "source_file": str(source_path),
            "source_sha256": source_hash,
            "prepared_file": str(prepared_path),
            "prepared_sha256": _sha256_file(prepared_path),
            "seed": seed,
            "audit_scope": (
                "Rule-based checks; source hashes identify exact file contents "
                "but do not independently authenticate sources or licenses. "
                "Privacy scanning is heuristic and cannot guarantee detection "
                "of all personal or sensitive information."
            ),
        }
    )
    audit_path = destination / "audit.json"
    audit_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report
