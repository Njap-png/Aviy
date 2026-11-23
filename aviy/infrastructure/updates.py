from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class UpdateFile:
    relative_path: str
    content: bytes
    sha256: str


@dataclass(frozen=True)
class UpdateManifest:
    update_id: str
    component: str
    version: str
    files: tuple[UpdateFile, ...]
    signature: bytes
    signed_payload: bytes


@dataclass(frozen=True)
class UpdateResult:
    update_id: str
    applied: tuple[str, ...]
    rolled_back: bool


def canonical_manifest_payload(manifest: UpdateManifest) -> bytes:
    body = {
        "update_id": manifest.update_id,
        "component": manifest.component,
        "version": manifest.version,
        "files": [
            {
                "relative_path": item.relative_path,
                "sha256": hashlib.sha256(item.content).hexdigest(),
            }
            for item in sorted(manifest.files, key=lambda entry: entry.relative_path)
        ],
    }
    return json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")


class UpdateManager:
    def __init__(
        self,
        root: str | Path,
        *,
        trusted_signature_verifier: Callable[[bytes, bytes], bool] | None,
        allowed_components: set[str],
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if trusted_signature_verifier is not None and not callable(
            trusted_signature_verifier
        ):
            raise TypeError("Trusted signature verifier must be callable.")
        if not isinstance(allowed_components, set) or any(
            not isinstance(item, str) or not item.strip()
            for item in allowed_components
        ):
            raise ValueError("Allowed components must be a set of non-empty strings.")
        self._verify_signature = trusted_signature_verifier
        self._allowed_components = frozenset(allowed_components)

    @staticmethod
    def _safe_relative_path(value: str) -> Path:
        if not isinstance(value, str):
            raise TypeError("Update path must be a string.")
        path = PurePosixPath(value)
        if (
            not value or path.is_absolute() or ".." in path.parts
            or "\\" in value or ":" in value or not path.parts
        ):
            raise ValueError("Unsafe update path.")
        return Path(*path.parts)

    def _validate_destination(self, relative: Path) -> Path:
        destination = self.root / relative
        current = self.root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("Update destination contains a symbolic link.")
        resolved = destination.resolve()
        if self.root not in resolved.parents:
            raise ValueError("Update path escapes the installation root.")
        return destination

    def verify(self, manifest: UpdateManifest) -> None:
        if not isinstance(manifest, UpdateManifest):
            raise TypeError("An UpdateManifest is required.")
        if (
            not isinstance(manifest.update_id, str)
            or not manifest.update_id.strip()
            or not isinstance(manifest.version, str)
            or not manifest.version.strip()
            or not isinstance(manifest.component, str)
            or not manifest.component.strip()
        ):
            raise ValueError("Update identity, component and version are required strings.")
        if manifest.component not in self._allowed_components:
            raise PermissionError("Component is not approved for updating.")
        if not isinstance(manifest.files, tuple) or not manifest.files:
            raise ValueError("Update files must be a non-empty tuple.")
        if not isinstance(manifest.signature, bytes) or not manifest.signature:
            raise ValueError("Update signature is required as non-empty bytes.")
        if not isinstance(manifest.signed_payload, bytes):
            raise TypeError("Signed update payload must be bytes.")
        if self._verify_signature is None:
            raise PermissionError("No trusted signature verifier is configured.")

        paths: set[str] = set()
        for item in manifest.files:
            if not isinstance(item, UpdateFile):
                raise TypeError("Update contains an invalid file record.")
            if not isinstance(item.content, bytes):
                raise TypeError("Update file content must be bytes.")
            if not isinstance(item.sha256, str):
                raise TypeError("Update file hash must be a string.")
            relative = self._safe_relative_path(item.relative_path)
            normalized = relative.as_posix().casefold()
            if normalized in paths:
                raise ValueError("Update contains duplicate file paths.")
            paths.add(normalized)
            actual = hashlib.sha256(item.content).hexdigest()
            if actual != item.sha256:
                raise ValueError("Update file hash verification failed.")
            self._validate_destination(relative)

        expected_payload = canonical_manifest_payload(manifest)
        if manifest.signed_payload != expected_payload:
            raise PermissionError(
                "Signed payload does not match the actual update manifest."
            )
        verified = self._verify_signature(expected_payload, manifest.signature)
        if verified is not True:
            raise PermissionError("Update signature verification failed.")

    def apply(
        self,
        manifest: UpdateManifest,
        *,
        compatibility_check: Callable[[UpdateManifest], bool],
        approval: Callable[[UpdateManifest], bool] | None = None,
    ) -> UpdateResult:
        self.verify(manifest)
        if not callable(compatibility_check):
            raise TypeError("Compatibility check must be callable.")
        compatible = compatibility_check(manifest)
        if compatible is not True:
            raise ValueError("Update is incompatible with this installation.")
        if approval is None or not callable(approval):
            raise PermissionError("Explicit update approval is required.")
        approved = approval(manifest)
        if approved is not True:
            raise PermissionError("Explicit update approval is required.")

        stage = Path(tempfile.mkdtemp(prefix=".aviy-stage-", dir=self.root))
        backup = Path(tempfile.mkdtemp(prefix=".aviy-backup-", dir=self.root))
        applied: list[Path] = []
        backed_up: set[Path] = set()
        preserve_backup = False
        try:
            for item in manifest.files:
                relative = self._safe_relative_path(item.relative_path)
                staged = stage / relative
                staged.parent.mkdir(parents=True, exist_ok=True)
                staged.write_bytes(item.content)
                if hashlib.sha256(staged.read_bytes()).hexdigest() != item.sha256:
                    raise OSError("Staged update integrity verification failed.")

            for item in manifest.files:
                relative = self._safe_relative_path(item.relative_path)
                destination = self._validate_destination(relative)
                staged = stage / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    saved = backup / relative
                    saved.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(destination, saved)
                    backed_up.add(relative)
                os.replace(staged, destination)
                applied.append(relative)

            return UpdateResult(
                manifest.update_id,
                tuple(path.as_posix() for path in applied),
                False,
            )
        except Exception:
            rollback_errors: list[Exception] = []
            for relative in reversed(applied):
                destination = self.root / relative
                saved = backup / relative
                try:
                    if relative in backed_up and saved.exists():
                        os.replace(saved, destination)
                    else:
                        destination.unlink(missing_ok=True)
                except Exception as exc:  # noqa: BLE001 - preserve rollback failure details
                    rollback_errors.append(exc)
            if rollback_errors:
                preserve_backup = True
                raise RuntimeError(
                    "Update failed and rollback was incomplete. "
                    f"Recovery backup preserved at: {backup}"
                ) from rollback_errors[0]
            raise
        finally:
            shutil.rmtree(stage, ignore_errors=True)
            if not preserve_backup:
                shutil.rmtree(backup, ignore_errors=True)
