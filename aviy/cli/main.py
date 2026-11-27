import os
from pathlib import Path

import typer
import uvicorn

from aviy.corpus_governance.cli import corpus_app
from aviy.data_ingestion.cli import ingestion_app

app = typer.Typer(name="aviy", help="Aviy local intelligence platform.")
app.add_typer(corpus_app, name="corpus")
app.add_typer(ingestion_app, name="ingest")
DEFAULT_PROJECT_ROOT = Path.cwd()
ROOT_OPTION = typer.Option(DEFAULT_PROJECT_ROOT, "--root")


@app.command()
def version():
    """Display the Aviy version."""
    typer.echo("Aviy 0.1.0")


@app.command()
def validate(root: Path = ROOT_OPTION):
    """Validate the basic project structure."""
    required = ["pyproject.toml", "aviy", "tests"]
    missing = [item for item in required if not (root / item).exists()]
    if missing:
        typer.echo("Missing: " + ", ".join(missing), err=True)
        raise typer.Exit(code=1)
    typer.echo("Project structure: valid")


def _make_validator():
    """Create the real persistent authentication service for API requests."""
    key = os.environ.get("AVIY_TOKEN_SIGNING_KEY")
    if not key or len(key.encode("utf-8")) < 32:
        raise RuntimeError("Set AVIY_TOKEN_SIGNING_KEY to at least 32 bytes.")

    from aviy.security.authentication import AuthenticationService

    database = Path(os.environ.get("AVIY_AUTH_DB", "data/aviy_auth.sqlite3"))
    database.parent.mkdir(parents=True, exist_ok=True)
    service = AuthenticationService(
        database=database,
        signing_key=key.encode("utf-8"),
    )
    return service.verify


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port", min=1, max=65535),
):
    """Start the local API."""
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise typer.BadParameter("Only loopback hosting is currently supported.")
    try:
        from aviy.api.app import create_app

        validator = _make_validator()
    except (ImportError, RuntimeError) as exc:
        typer.echo(f"Authentication setup failed: {exc}", err=True)
        raise typer.Exit(code=1)
    uvicorn.run(create_app(validator), host=host, port=port)




@app.command()
def pretrain(
    dataset: Path = typer.Option(..., "--dataset", exists=True, dir_okay=False),  # noqa: B008
    audit: Path = typer.Option(..., "--audit", exists=True, dir_okay=False),  # noqa: B008
    tokenizer_path: Path = typer.Option(..., "--tokenizer", exists=True, dir_okay=False),  # noqa: B008
    authorization_path: Path = typer.Option(..., "--authorization", exists=True, dir_okay=False),  # noqa: B008
    checkpoint: Path = typer.Option(..., "--checkpoint"),  # noqa: B008
    report_path: Path = typer.Option(Path("reports/base_pretraining_report.json"), "--report"),  # noqa: B008
    steps: int = typer.Option(100, "--steps", min=1),
    batch_size: int = typer.Option(4, "--batch-size", min=1),
    learning_rate: float = typer.Option(1e-4, "--learning-rate", min=0.0),
    device: str = typer.Option("auto", "--device"),
    precision: str = typer.Option("auto", "--precision"),
    seed: int = typer.Option(42, "--seed", min=0),
    model_config_path: Path | None = typer.Option(None, "--model-config", exists=True, dir_okay=False),  # noqa: B008
):
    """Run explicitly authorized base pretraining on a prepared dataset."""
    import json
    from dataclasses import fields

    from aviy.data.pipeline import DatasetAudit, PreparedDataset, read_jsonl
    from aviy.models.base_model import AviyBaseLM, BaseModelConfig
    from aviy.tokenizer.bpe import ByteBPETokenizer
    from aviy.training.authorization import TrainingAuthorization
    from aviy.training.engine import write_training_report
    from aviy.training.pretraining import pretrain_base_model

    try:
        records = read_jsonl(dataset)
        audit_payload = json.loads(audit.read_text(encoding="utf-8"))
        audit_data = audit_payload.get("audit", audit_payload)
        if not isinstance(audit_data, dict):
            raise TypeError("Audit file must contain an audit object")

        audit_fields = {item.name for item in fields(DatasetAudit)}
        missing = audit_fields.difference(audit_data)
        if missing:
            raise ValueError(
                "Audit file is missing required fields: " + ", ".join(sorted(missing))
            )
        dataset_audit = DatasetAudit(
            **{name: audit_data[name] for name in audit_fields}
        )
        prepared = PreparedDataset(records=tuple(records), audit=dataset_audit)

        authorization_data = json.loads(
            authorization_path.read_text(encoding="utf-8")
        )
        authorization = TrainingAuthorization(**authorization_data)

        config_data = {}
        if model_config_path is not None:
            config_data = json.loads(
                model_config_path.read_text(encoding="utf-8")
            )
            if not isinstance(config_data, dict):
                raise ValueError("Model configuration must be a JSON object")
        config = BaseModelConfig(**config_data)
        model = AviyBaseLM(config)
        tokenizer = ByteBPETokenizer.load(tokenizer_path)

        result = pretrain_base_model(
            model,
            tokenizer,
            prepared,
            steps=steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device_name=device,
            precision=precision,
            seed=seed,
            checkpoint_path=checkpoint,
            authorization=authorization,
        )
        write_training_report(result, report_path)
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False))
    except (ValueError, TypeError, PermissionError, OSError, KeyError) as exc:
        typer.echo(f"Pretraining rejected: {exc}", err=True)
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
