"""Local experiment metadata, event history and lineage tracking."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ExperimentTracker:
    """Record reproducible experiment identity and parent relationships."""

    def __init__(
        self,
        directory: str | Path,
        *,
        config: dict[str, Any],
        seed: int,
        parent_run_id: str | None = None,
        description: str = "",
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("Experiment seed must be a non-negative integer.")

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run_id = str(uuid.uuid4())
        self.metadata_path = self.directory / "metadata.json"
        self.events_path = self.directory / "events.jsonl"

        self.metadata = {
            "format": "aviy-experiment-v1",
            "run_id": self.run_id,
            "parent_run_id": parent_run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "seed": seed,
            "description": description,
            "config": config,
        }
        self._write_metadata()
        self.record_event("experiment_created", {"seed": seed})

    def _write_metadata(self) -> None:
        self.metadata_path.write_text(
            json.dumps(self.metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def record_event(
        self,
        event: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        if not event.strip():
            raise ValueError("Event name cannot be empty.")

        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "run_id": self.run_id,
            "event": event,
            "details": details or {},
        }
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def record_checkpoint(self, path: str | Path, step: int) -> None:
        self.record_event(
            "checkpoint_saved",
            {"path": str(path), "step": step},
        )

    def record_result(self, result: dict[str, Any]) -> None:
        self.record_event("experiment_result", result)
