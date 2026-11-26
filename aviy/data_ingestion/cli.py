from __future__ import annotations

import json
from pathlib import Path

import typer

from aviy.data_ingestion.service import intake_file, read_history

FILE_OPTION = typer.Option(..., "--file", exists=True, dir_okay=False)
METADATA_OPTION = typer.Option(..., "--metadata", exists=True, dir_okay=False)
NAME_OPTION = typer.Option(None, "--name")
FORMAT_OPTION = typer.Option(None, "--format")

ingestion_app = typer.Typer(
    name="ingest",
    help="Import local datasets through Aviy's governed evaluator.",
    no_args_is_help=True,
)


@ingestion_app.command("file")
def ingest_file_command(
    file: Path = FILE_OPTION,
    metadata: Path = METADATA_OPTION,
    name: str | None = NAME_OPTION,
    format: str | None = FORMAT_OPTION,
):
    """Ingest one local JSONL, JSON, CSV, or TXT dataset."""
    try:
        details = json.loads(metadata.read_text(encoding="utf-8-sig"))
        result = intake_file(file, details, name=name, file_format=format)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        typer.echo(f"Ingestion failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] == "INGESTED_EVALUATION_FAILED":
        raise typer.Exit(code=1)


@ingestion_app.command("history")
def history_command():
    """Display local ingestion history."""
    typer.echo(json.dumps(read_history(), indent=2, ensure_ascii=False))
