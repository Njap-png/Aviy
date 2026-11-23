"""Reproducible dataset preparation and audit report generation."""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from aviy.data.pipeline import prepare_dataset, read_jsonl, write_jsonl


def audit_dataset(
    source: str | Path,
    output_directory: str | Path,
    *,
    allowed_licenses: Iterable[str],
    seed: int = 42,
) -> dict[str, object]:
    records = read_jsonl(source)
    prepared = prepare_dataset(
        records,
        allowed_licenses=allowed_licenses,
        seed=seed,
    )
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    write_jsonl(prepared.records, destination / "prepared.jsonl")
    report = prepared.audit.to_dict()
    report["source_file"] = str(Path(source))
    report["prepared_file"] = str(destination / "prepared.jsonl")
    report["audit_scope"] = (
        "Rule-based checks; privacy scanning is heuristic and does not "
        "guarantee detection of all personal or sensitive information."
    )
    (destination / "audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report
