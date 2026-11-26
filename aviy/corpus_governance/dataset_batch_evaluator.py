"""Batch evaluation for explicitly selected local Aviy datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATOR_MODULE = "aviy.corpus_governance.dataset_evaluator"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def resolve_input(value: Any, manifest_dir: Path, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{field}' must be a non-empty path.")

    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = manifest_dir / candidate

    resolved = candidate.resolve()
    if not resolved.is_file():
        raise ValueError(f"Input '{field}' does not exist: {resolved}")
    return resolved


def load_batch_manifest(manifest_path: str | Path):
    manifest = Path(manifest_path).expanduser().resolve()
    if not manifest.is_file():
        raise ValueError(f"Batch manifest does not exist: {manifest}")

    try:
        data = json.loads(manifest.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid batch manifest: {exc}") from exc

    if not isinstance(data, dict) or data.get("schema_version") != "1.0":
        raise ValueError("Batch manifest schema_version must be '1.0'.")

    entries = data.get("datasets")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Batch manifest must contain a non-empty datasets list.")

    prepared = []
    names = set()

    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Dataset entry {index} must be an object.")

        name = entry.get("name")
        if not isinstance(name, str) or not NAME_PATTERN.fullmatch(name):
            raise ValueError(f"Dataset entry {index} has an invalid name.")

        if name in names:
            raise ValueError(f"Duplicate dataset name: {name}")
        names.add(name)

        records = resolve_input(entry.get("records"), manifest.parent, "records")
        metadata = resolve_input(entry.get("metadata"), manifest.parent, "metadata")

        if records == metadata:
            raise ValueError(f"Dataset '{name}' uses one file for both inputs.")

        prepared.append({
            "name": name,
            "records": records,
            "metadata": metadata,
        })

    return manifest, prepared


def read_report(path: Path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def run_batch(manifest_path: str | Path, output_path: str | Path):
    manifest, datasets = load_batch_manifest(manifest_path)
    output_root = Path(output_path).expanduser().resolve()

    if output_root.exists():
        raise FileExistsError(
            f"Output directory already exists; refusing overwrite: {output_root}"
        )

    if output_root == manifest:
        raise ValueError("Output directory cannot be the manifest file.")

    # Build the complete batch beside the destination so final publication
    # can use a same-filesystem directory rename.
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(
        prefix=f".{output_root.name}.aviy-staging-",
        dir=str(output_root.parent),
    ))
    published = False

    try:
        results = []

        for dataset in datasets:
            name = dataset["name"]
            dataset_output = staging_root / name
            report_path = dataset_output / "evaluation_report.json"

            command = [
                sys.executable,
                "-m",
                EVALUATOR_MODULE,
                "--records",
                str(dataset["records"]),
                "--metadata",
                str(dataset["metadata"]),
                "--output",
                str(dataset_output),
            ]

            try:
                process = subprocess.run(
                    command,
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                return_code = process.returncode
            except OSError:
                return_code = -1

            report = read_report(report_path)
            successful = return_code == 0 and report is not None

            accepted_path = dataset_output / "corrected_accepted.jsonl"
            quarantined_path = dataset_output / "quarantined.jsonl"

            results.append({
                "name": name,
                "status": "COMPLETED" if successful else "FAILED",
                "evaluator_status": report.get("status") if report else None,
                "return_code": return_code,
                "records_input_sha256": sha256_file(dataset["records"]),
                "metadata_input_sha256": sha256_file(dataset["metadata"]),
                "report_sha256": (
                    sha256_file(report_path) if report_path.is_file() else None
                ),
                "records_staged": count_jsonl(accepted_path),
                "records_quarantined": count_jsonl(quarantined_path),
                "output_directory": str(output_root / name),
                "error": None if successful else (
                    "Evaluator failed or did not produce a readable report."
                ),
            })

        failed = sum(item["status"] == "FAILED" for item in results)
        completed = len(results) - failed

        batch_report = {
            "schema_version": "1.0",
            "status": "COMPLETED" if failed == 0 else "COMPLETED_WITH_FAILURES",
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "manifest_sha256": sha256_file(manifest),
            "datasets_total": len(results),
            "datasets_completed": completed,
            "datasets_failed": failed,
            "records_staged": sum(x["records_staged"] for x in results),
            "records_quarantined": sum(x["records_quarantined"] for x in results),
            "datasets": results,
            "production_source_registry_changed": False,
            "tokenizer_training_authorized": False,
            "pretraining_authorized": False,
            "training_started": False,
            "legal_rights_independently_verified": False,
            "note": (
                "Technical screening only. This report does not grant legal "
                "clearance, source approval, record approval, tokenizer "
                "authorization, or training authorization."
            ),
        }

        report_path = staging_root / "batch_report.json"
        temporary_path = staging_root / "batch_report.json.tmp"
        temporary_path.write_text(
            json.dumps(batch_report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(report_path)

        # A same-filesystem rename publishes the entire completed directory.
        # If another process creates the destination first, publication fails
        # without replacing or modifying that destination.
        os.rename(staging_root, output_root)
        published = True

        return batch_report

    finally:
        # Clean up our private staging directory after ordinary failures.
        # A hard process termination or power loss can still leave an orphan;
        # such directories are deliberately not auto-deleted on later runs.
        if not published and staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate selected local Aviy datasets in one batch."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    try:
        report = run_batch(args.manifest, args.output)
    except (ValueError, FileExistsError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(json.dumps({
        "status": report["status"],
        "datasets_total": report["datasets_total"],
        "datasets_completed": report["datasets_completed"],
        "datasets_failed": report["datasets_failed"],
        "records_staged": report["records_staged"],
        "records_quarantined": report["records_quarantined"],
        "report": str(Path(args.output).resolve() / "batch_report.json"),
    }, indent=2))

    return 0 if report["datasets_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())