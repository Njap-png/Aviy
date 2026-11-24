"""Durable, HMAC-protected governance records and gated recovery."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: dict[str, threading.RLock] = {}


class LedgerIntegrityError(Exception):
    """Raised when the governance ledger is malformed or has been altered."""


class RecoveryVerificationError(Exception):
    """Raised when a recovery action fails its post-recovery verification."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def _path_lock(path: Path) -> threading.RLock:
    key = os.path.normcase(str(path))
    with _LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(key, threading.RLock())


class DurableGovernanceLedger:
    """Append-only JSONL ledger with a keyed hash chain.

    Thread-safe between instances in this Python process. This does not provide
    inter-process locking or protection against a privileged host administrator.
    """

    def __init__(self, path: str | Path, integrity_key: bytes) -> None:
        if not isinstance(integrity_key, bytes) or len(integrity_key) < 32:
            raise ValueError("integrity_key must contain at least 32 bytes")
        supplied = Path(path).expanduser()
        if supplied.is_symlink():
            raise ValueError("Ledger path cannot be a symlink")
        parent = supplied.absolute().parent.resolve()
        self.path = parent / supplied.name
        if self.path.is_symlink():
            raise ValueError("Ledger path cannot be a symlink")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._key = bytes(integrity_key)
        self._lock = _path_lock(self.path)
        self.verify()

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            raw = self.path.read_bytes()
            if raw and not raw.endswith(b"\n"):
                raise LedgerIntegrityError("Ledger has an incomplete final record")
            records = []
            for line in raw.splitlines():
                record = json.loads(
                    line, parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError(f"Invalid JSON constant: {value}")
                    )
                )
                if not isinstance(record, dict):
                    raise LedgerIntegrityError("Invalid ledger record")
                records.append(record)
            return records
        except LedgerIntegrityError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise LedgerIntegrityError("Unable to read governance ledger") from exc

    def _record_hash(self, body: Mapping) -> str:
        return hmac.new(self._key, _canonical(body), hashlib.sha256).hexdigest()

    def verify(self) -> bool:
        with self._lock:
            previous = "0" * 64
            for sequence, record in enumerate(self._read(), start=1):
                body = {key: value for key, value in record.items() if key != "mac"}
                if (
                    body.get("sequence") != sequence
                    or body.get("previous_hash") != previous
                    or not isinstance(record.get("mac"), str)
                    or not hmac.compare_digest(
                        record["mac"], self._record_hash(body)
                    )
                ):
                    raise LedgerIntegrityError(
                        f"Governance ledger integrity failure at record {sequence}"
                    )
                previous = record["mac"]
            return True

    def append(
        self, actor: str, action: str, details: Mapping | None = None
    ) -> dict:
        if not isinstance(actor, str) or not actor.strip():
            raise ValueError("actor is required")
        if not isinstance(action, str) or not action.strip():
            raise ValueError("action is required")
        if details is None:
            details = {}
        if not isinstance(details, Mapping):
            raise TypeError("details must be a mapping")
        safe_details = json.loads(_canonical(dict(details)))
        with self._lock:
            self.verify()
            records = self._read()
            previous = records[-1]["mac"] if records else "0" * 64
            body = {
                "sequence": len(records) + 1,
                "timestamp": int(time.time()),
                "actor": actor.strip(),
                "action": action.strip(),
                "details": safe_details,
                "previous_hash": previous,
            }
            record = dict(body, mac=self._record_hash(body))
            encoded = _canonical(record) + b"\n"
            with self.path.open("ab") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass
            return record


class GovernedRecovery:
    """Require two distinct recovery approvers and verify or roll back changes."""

    REQUIRED_ROLE = "recovery_approver"

    def __init__(self, ledger: DurableGovernanceLedger) -> None:
        if not isinstance(ledger, DurableGovernanceLedger):
            raise TypeError("A DurableGovernanceLedger is required.")
        self.ledger = ledger

    def run(
        self,
        proposal_id: str,
        approvals: Sequence[tuple[str, Sequence[str]]],
        action: Callable[[], object],
        verify: Callable[[], bool],
        rollback: Callable[[], bool],
    ) -> object:
        if not isinstance(proposal_id, str) or not proposal_id.strip():
            raise ValueError("proposal_id is required")
        if not isinstance(approvals, Sequence) or isinstance(approvals, (str, bytes)):
            raise TypeError("approvals must be a sequence of approval records")
        if not all(callable(fn) for fn in (action, verify, rollback)):
            raise TypeError("action, verify, and rollback must be callable")

        qualified = set()
        for approval in approvals:
            if (
                not isinstance(approval, (tuple, list))
                or len(approval) != 2
            ):
                raise ValueError("Each approval must contain a name and roles.")
            name, roles = approval
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Approver names must be non-empty strings.")
            if (
                not isinstance(roles, Sequence)
                or isinstance(roles, (str, bytes))
                or any(not isinstance(role, str) for role in roles)
            ):
                raise ValueError("Approver roles must be a sequence of strings.")
            if self.REQUIRED_ROLE in roles:
                qualified.add(name.strip().casefold())

        if len(qualified) < 2:
            self.ledger.append(
                "governance", "recovery_rejected",
                {"proposal_id": proposal_id, "reason": "two distinct approvers required"},
            )
            raise PermissionError("Two distinct recovery approvers are required")

        self.ledger.append(
            "governance", "recovery_started",
            {"proposal_id": proposal_id, "approvers": sorted(qualified)},
        )
        try:
            result = action()
            if verify() is not True:
                raise RecoveryVerificationError("Recovery verification failed")
        except Exception as exc:
            try:
                rolled_back = rollback()
            except Exception as rollback_error:
                self.ledger.append(
                    "governance", "recovery_rollback_failed",
                    {"proposal_id": proposal_id, "error": type(rollback_error).__name__},
                )
                raise RuntimeError(
                    "Recovery failed and rollback raised an exception"
                ) from rollback_error
            self.ledger.append(
                "governance",
                "recovery_rolled_back" if rolled_back is True else "recovery_rollback_failed",
                {"proposal_id": proposal_id, "cause": type(exc).__name__},
            )
            if rolled_back is not True:
                raise RuntimeError(
                    "Recovery failed and rollback was not verified"
                ) from exc
            raise

        self.ledger.append(
            "governance", "recovery_succeeded", {"proposal_id": proposal_id}
        )
        return result
