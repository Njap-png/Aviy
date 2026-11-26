import base64
import binascii
import json
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from aviy.data_ingestion.service import (
    MAX_FILE_BYTES,
    intake_bytes,
    read_history,
)
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
    async def ingestion_intake(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> dict[str, object]:
        authenticate(authorization)

        max_base64_chars = 4 * ((MAX_FILE_BYTES + 2) // 3)
        max_request_bytes = max_base64_chars + 1024 * 1024

        content_length = request.headers.get("content-length")

        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid Content-Length header",
                ) from None

            if declared_length < 0:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid Content-Length header",
                )

            if declared_length > max_request_bytes:
                raise HTTPException(
                    status_code=413,
                    detail="Upload exceeds the request size limit",
                )

        body = bytearray()

        async for chunk in request.stream():
            body.extend(chunk)

            if len(body) > max_request_bytes:
                raise HTTPException(
                    status_code=413,
                    detail="Upload exceeds the request size limit",
                )

        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            raise HTTPException(
                status_code=400,
                detail="Request body must be valid JSON",
            ) from None

        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=400,
                detail="Request body must be a JSON object",
            )

        filename = payload.get("filename")
        encoded = payload.get("content_base64")
        metadata = payload.get("metadata")
        name = payload.get("name")
        file_format = payload.get("format")

        if not isinstance(filename, str) or not isinstance(encoded, str):
            raise HTTPException(
                status_code=400,
                detail="filename and content_base64 are required",
            )

        if len(encoded) > max_base64_chars:
            raise HTTPException(
                status_code=413,
                detail="Upload exceeds the 50 MiB limit",
            )

        if name is not None and not isinstance(name, str):
            raise HTTPException(
                status_code=400,
                detail="name must be a string",
            )

        if file_format is not None and not isinstance(file_format, str):
            raise HTTPException(
                status_code=400,
                detail="format must be a string",
            )

        if not isinstance(metadata, dict):
            raise HTTPException(
                status_code=400,
                detail="metadata must be an object",
            )

        try:
            content = base64.b64decode(
                encoded,
                validate=True,
            )

            if len(content) > MAX_FILE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="Upload exceeds the 50 MiB limit",
                )

            result = await run_in_threadpool(
                intake_bytes,
                filename,
                content,
                metadata,
                name,
                file_format,
            )

        except HTTPException:
            raise

        except (
            ValueError,
            binascii.Error,
            OSError,
            TimeoutError,
        ) as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

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
