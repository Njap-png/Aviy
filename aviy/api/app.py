import base64
import binascii
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse

from aviy.data_ingestion.service import intake_bytes, read_history
from aviy.infrastructure.resources import (
    ResourceMonitor,
    ResourceQuota,
    WorkloadScheduler,
)
from aviy.security.authentication import AuthenticationError


def create_app(
    token_validator: Callable | None = None,
    research_handler: Callable[[dict[str, object]], object] | None = None,
    research_scheduler: WorkloadScheduler | None = None,
) -> FastAPI:
    """Create the local API.

    Research requests are queued, not executed inside the HTTP request.
    A configured worker can execute queued work through
    app.state.research_scheduler.run_next().
    """
    app = FastAPI(
        title="Aviy API",
        version="0.1.0",
        description="Local API for the Aviy intelligence platform.",
    )
    web_root = Path(__file__).resolve().parents[1] / "interfaces" / "web"

    scheduler = research_scheduler
    if scheduler is None and research_handler is not None:
        scheduler = WorkloadScheduler(ResourceQuota(), ResourceMonitor())

    app.state.research_scheduler = scheduler

    def authenticate(authorization: str | None) -> object:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="Bearer token required")
        if token_validator is None:
            raise HTTPException(status_code=503, detail="Authentication unavailable")
        token = authorization.split(" ", 1)[1].strip()
        try:
            identity = token_validator(token)
        except (AuthenticationError, TypeError, ValueError):
            raise HTTPException(status_code=401, detail="Invalid or expired token") from None
        if not identity:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        return identity


    @app.post("/v1/ingestion/intake", status_code=201)
    def ingestion_intake(
        payload: dict[str, object],
        authorization: str | None = Header(default=None),
    ) -> dict[str, object]:
        authenticate(authorization)
        filename = payload.get("filename")
        encoded = payload.get("content_base64")
        metadata = payload.get("metadata")
        name = payload.get("name")
        file_format = payload.get("format")

        if not isinstance(filename, str) or not isinstance(encoded, str):
            raise HTTPException(status_code=400, detail="filename and content_base64 are required")
        if len(encoded) > 70 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Upload exceeds the 50 MiB limit")
        try:
            content = base64.b64decode(encoded, validate=True)
            result = intake_bytes(
                filename,
                content,
                metadata,
                name=name if isinstance(name, str) else None,
                file_format=file_format if isinstance(file_format, str) else None,
            )
        except (ValueError, binascii.Error, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return result

    @app.get("/v1/ingestion/history")
    def ingestion_history(
        authorization: str | None = Header(default=None),
    ) -> dict[str, object]:
        authenticate(authorization)
        return {"items": read_history()}

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "aviy"}

    @app.get("/v1/status")
    def status(authorization: str | None = Header(default=None)) -> dict[str, object]:
        identity = authenticate(authorization)
        return {"status": "ok", "authenticated": True, "identity": str(identity)}

    @app.post("/v1/research/requests", status_code=202)
    def research_request(
        payload: dict[str, object],
        authorization: str | None = Header(default=None),
    ) -> dict[str, object]:
        authenticate(authorization)

        handler = research_handler
        if handler is None or scheduler is None:
            raise HTTPException(
                status_code=503,
                detail="Research execution is not configured",
            )

        job_id = str(uuid4())
        request_payload = dict(payload)

        def execute_research() -> object:
            return handler(request_payload)

        submission = scheduler.submit(job_id, execute_research)
        if not submission.get("accepted", False):
            raise HTTPException(
                status_code=429,
                detail="Research queue rejected the request",
            )

        return {
            "accepted": True,
            "job_id": job_id,
            "status": "queued",
            "queue_size": submission.get("queue_size"),
        }

    @app.get("/")
    def interface() -> FileResponse:
        page = web_root / "index.html"
        if not page.is_file():
            raise HTTPException(status_code=404, detail="Interface unavailable")
        return FileResponse(page)

    return app


app = create_app()
