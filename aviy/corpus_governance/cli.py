from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Annotated, Any

import typer

from aviy.corpus_governance import (
    CorpusRecord,
    GovernanceError,
    SourceRegistry,
    build_manifest,
    validate_record,
)

corpus_app = typer.Typer(
    name="corpus",
    help="Read-only corpus governance inspection commands.",
    no_args_is_help=True,
)

DEFAULT_ROOT = Path.cwd()
DEFAULT_REGISTRY = Path("data/aviy_corpus/manifests/source_registry.json")


def _resolve(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    return candidate.resolve()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Could not read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _load_registry(root: Path, registry_path: Path) -> SourceRegistry:
    data = _read_json(_resolve(root, registry_path))

    if not isinstance(data, dict):
        raise TypeError("Source registry must be a JSON object.")

    try:
        return SourceRegistry.from_dict(data)
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError(f"Invalid source registry: {exc}") from exc


def _load_records(path: Path) -> list[CorpusRecord]:
    if not path.is_file():
        raise ValueError(f"Records file does not exist: {path}")

    records: list[CorpusRecord] = []

    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number}: {exc}"
                    ) from exc

                if not isinstance(item, dict):
                    raise TypeError(
                        f"Line {line_number} must contain a JSON object."
                    )

                try:
                    record = CorpusRecord(**item)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid record on line {line_number}: {exc}"
                    ) from exc

                for field_name, value in vars(record).items():
                    if not isinstance(value, str):
                        raise TypeError(
                            f"Line {line_number}: '{field_name}' must be a string."
                        )

                records.append(record)

    except OSError as exc:
        raise ValueError(f"Could not read records file {path}: {exc}") from exc

    return records


def _run_read(operation):
    try:
        operation()
    except (OSError, TypeError, ValueError, GovernanceError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@corpus_app.command("sources")
def sources(
    root: Annotated[Path, typer.Option("--root")] = DEFAULT_ROOT,
    registry: Annotated[Path, typer.Option("--registry")] = DEFAULT_REGISTRY,
):
    """List registered sources and their training permissions."""

    def run():
        source_registry = _load_registry(root, registry)
        entries = source_registry.to_dict().get("sources", [])

        if not entries:
            typer.echo("No sources are registered.")
            return

        for item in entries:
            typer.echo(
                f"{item.get('source_id', '<unknown>')} | "
                f"{item.get('source_name', '<unnamed>')} | "
                f"approval={item.get('approval_status', 'unknown')} | "
                f"license={item.get('license_status', 'unknown')} | "
                f"training_allowed={item.get('training_allowed', False)}"
            )

    _run_read(run)


@corpus_app.command("validate")
def validate(
    records: Annotated[Path, typer.Option("--records")],
    root: Annotated[Path, typer.Option("--root")] = DEFAULT_ROOT,
    registry: Annotated[Path, typer.Option("--registry")] = DEFAULT_REGISTRY,
):
    """Validate governance records without modifying them."""

    def run():
        source_registry = _load_registry(root, registry)
        loaded = _load_records(_resolve(root, records))
        invalid_count = 0

        for index, record in enumerate(loaded, start=1):
            try:
                issues = validate_record(record, source_registry)
            except (AttributeError, TypeError, ValueError) as exc:
                issues = [f"Validation could not complete: {exc}"]

            if issues:
                invalid_count += 1
                for issue in issues:
                    typer.echo(f"Record {index}: {issue}")
            else:
                typer.echo(f"Record {index}: valid")

        typer.echo(
            f"Validation complete: {len(loaded)} records, "
            f"{invalid_count} invalid."
        )

        if invalid_count:
            raise typer.Exit(code=1)

    _run_read(run)


@corpus_app.command("audit")
def audit(
    records: Annotated[Path, typer.Option("--records")],
    root: Annotated[Path, typer.Option("--root")] = DEFAULT_ROOT,
    registry: Annotated[Path, typer.Option("--registry")] = DEFAULT_REGISTRY,
):
    """Summarize record governance states and validation failures."""

    def run():
        source_registry = _load_registry(root, registry)
        loaded = _load_records(_resolve(root, records))

        approval = Counter(record.approval_status for record in loaded)
        review = Counter(record.review_status for record in loaded)
        privacy = Counter(record.privacy_status for record in loaded)
        source_counts = Counter(record.source_id for record in loaded)

        invalid_count = 0
        for record in loaded:
            try:
                if validate_record(record, source_registry):
                    invalid_count += 1
            except (AttributeError, TypeError, ValueError):
                invalid_count += 1

        typer.echo(f"Total records: {len(loaded)}")
        typer.echo(f"Records with validation issues: {invalid_count}")
        typer.echo(f"Approval states: {dict(approval)}")
        typer.echo(f"Review states: {dict(review)}")
        typer.echo(f"Privacy states: {dict(privacy)}")
        typer.echo(f"Source counts: {dict(source_counts)}")

    _run_read(run)


@corpus_app.command("manifest")
def manifest(
    manifest_path: Annotated[Path, typer.Option("--manifest")],
    root: Annotated[Path, typer.Option("--root")] = DEFAULT_ROOT,
):
    """Inspect an existing manifest without modifying it."""

    def run():
        data = _read_json(_resolve(root, manifest_path))

        if not isinstance(data, dict):
            raise TypeError("Manifest must be a JSON object.")

        fields = (
            "schema_version",
            "corpus",
            "created_at",
            "seed",
            "split_ratios",
            "record_count",
            "split_counts",
            "training_authorized",
            "test_split_training_allowed",
            "test_split_model_selection_allowed",
        )

        for key in fields:
            typer.echo(f"{key}: {data.get(key, '<missing>')}")

    _run_read(run)


@corpus_app.command("verify")
def verify(
    manifest_path: Annotated[Path, typer.Option("--manifest")],
    records: Annotated[Path, typer.Option("--records")],
    root: Annotated[Path, typer.Option("--root")] = DEFAULT_ROOT,
    registry: Annotated[Path, typer.Option("--registry")] = DEFAULT_REGISTRY,
):
    """Verify an existing manifest against records and the source registry."""

    def run():
        source_registry = _load_registry(root, registry)
        supplied = _read_json(_resolve(root, manifest_path))

        if not isinstance(supplied, dict):
            raise TypeError("Manifest must be a JSON object.")

        loaded = _load_records(_resolve(root, records))

        try:
            expected = build_manifest(loaded, source_registry)
        except (GovernanceError, ValueError) as exc:
            raise ValueError(
                f"Records cannot produce a valid manifest: {exc}"
            ) from exc

        keys = (
            "schema_version",
            "corpus",
            "seed",
            "split_ratios",
            "record_count",
            "split_counts",
            "records",
            "training_authorized",
            "test_split_training_allowed",
            "test_split_model_selection_allowed",
        )

        mismatches = [
            key for key in keys
            if supplied.get(key) != expected.get(key)
        ]

        if mismatches:
            raise ValueError(
                "Manifest verification failed. Mismatched fields: "
                + ", ".join(mismatches)
            )

        typer.echo("Manifest verification passed.")
        typer.echo("The existing manifest matches the records and registry.")

    _run_read(run)
