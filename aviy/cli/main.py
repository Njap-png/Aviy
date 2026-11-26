import os
from pathlib import Path

import typer
import uvicorn

from aviy.corpus_governance.cli import corpus_app

app = typer.Typer(name="aviy", help="Aviy local intelligence platform.")
app.add_typer(corpus_app, name="corpus")
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


if __name__ == "__main__":
    app()
