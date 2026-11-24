"""Owner-scoped working, episodic and semantic memory with recovery."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Literal

MemoryKind = Literal["working", "episodic", "semantic"]


@dataclass(frozen=True)
class MemoryRecord:
    record_id: str
    owner_id: str
    kind: MemoryKind
    content: str
    provenance: str
    created_at: float
    expires_at: float | None = None
    version: int = 1
    metadata: dict[str, str] | None = None


class MemoryStore:
    """In-memory store; snapshots provide explicit backup and recovery."""

    def __init__(self, maximum_records: int = 10000) -> None:
        if maximum_records < 1:
            raise ValueError("maximum_records must be positive")
        self.maximum_records = maximum_records
        self._records: dict[str, MemoryRecord] = {}

    def put(
        self,
        *,
        record_id: str,
        owner_id: str,
        kind: MemoryKind,
        content: str,
        provenance: str,
        ttl_seconds: float | None = None,
        metadata: dict[str, str] | None = None,
        conflict: Literal["reject", "replace"] = "reject",
    ) -> MemoryRecord:
        if not all(
            value.strip() for value in (record_id, owner_id, content, provenance)
        ):
            raise ValueError("Memory ID, owner, content and provenance are required")
        if kind not in ("working", "episodic", "semantic"):
            raise ValueError("Unsupported memory kind")
        if conflict not in ("reject", "replace"):
            raise ValueError("Unsupported conflict policy")
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")

        now = time.time()
        existing = self._records.get(record_id)
        if existing is not None:
            if existing.owner_id != owner_id:
                raise PermissionError("Memory belongs to another owner")
            if conflict == "reject":
                raise ValueError("Memory record already exists")
            version = existing.version + 1
            created_at = existing.created_at
        else:
            if len(self._records) >= self.maximum_records:
                raise OverflowError("Memory store capacity reached")
            version = 1
            created_at = now

        record = MemoryRecord(
            record_id=record_id,
            owner_id=owner_id,
            kind=kind,
            content=content,
            provenance=provenance,
            created_at=created_at,
            expires_at=now + ttl_seconds if ttl_seconds is not None else None,
            version=version,
            metadata=dict(metadata or {}),
        )
        self._records[record_id] = record
        return record

    def get(self, record_id: str, owner_id: str) -> MemoryRecord | None:
        record = self._records.get(record_id)
        if record is None:
            return None
        if record.owner_id != owner_id:
            raise PermissionError("Memory belongs to another owner")
        if record.expires_at is not None and record.expires_at <= time.time():
            del self._records[record_id]
            return None
        return record

    def search(
        self,
        query: str,
        owner_id: str,
        *,
        kind: MemoryKind | None = None,
        limit: int = 20,
    ) -> list[MemoryRecord]:
        if limit < 1:
            raise ValueError("limit must be positive")
        needle = query.casefold().strip()
        results = []
        for record_id in list(self._records):
            record = self.get(record_id, owner_id)
            if record is None:
                continue
            if kind is not None and record.kind != kind:
                continue
            if needle in record.content.casefold():
                results.append(record)
        return results[:limit]

    def delete(self, record_id: str, owner_id: str) -> bool:
        record = self._records.get(record_id)
        if record is None:
            return False
        if record.owner_id != owner_id:
            raise PermissionError("Memory belongs to another owner")
        del self._records[record_id]
        return True

    def expire(self, now: float | None = None) -> int:
        moment = time.time() if now is None else now
        expired = [
            key
            for key, record in self._records.items()
            if record.expires_at is not None and record.expires_at <= moment
        ]
        for key in expired:
            del self._records[key]
        return len(expired)

    def snapshot(self) -> dict[str, object]:
        return {
            "format": "aviy-memory-snapshot-v1",
            "records": [asdict(record) for record in self._records.values()],
        }

    def restore(self, snapshot: dict[str, object]) -> None:
        if snapshot.get("format") != "aviy-memory-snapshot-v1":
            raise ValueError("Unsupported memory snapshot format")
        raw_records = snapshot.get("records")
        if not isinstance(raw_records, list):
            raise TypeError("Snapshot records must be a list")
        restored: dict[str, MemoryRecord] = {}
        for raw in raw_records:
            if not isinstance(raw, dict):
                raise TypeError("Invalid memory record in snapshot")
            record = MemoryRecord(**raw)
            if (
                not record.record_id
                or not record.owner_id
                or not record.provenance
                or record.version < 1
                or record.kind not in ("working", "episodic", "semantic")
                or record.record_id in restored
            ):
                raise ValueError("Invalid or duplicate memory snapshot record")
            restored[record.record_id] = record
        if len(restored) > self.maximum_records:
            raise OverflowError("Snapshot exceeds memory store capacity")
        self._records = restored
