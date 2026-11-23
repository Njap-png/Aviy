from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable


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


class UpdateManager:
    def __init__(
        self,
        root: str | Path,
        *,
        trusted_signature_verifier: Callable[[bytes, bytes], bool],
        allowed_components: set[str],
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._verify_signature = trusted_signature_verifier
        self._allowed_components = frozenset(allowed_components)

    @staticmethod
    def _safe_relative_path(value: str) -> Path:
        path = PurePosixPath(value)
        if (
            not value
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in value
            or ":" in value
        ):
            raise ValueError("Unsafe update path.")
        return Path(*path.parts)

    def verify(self, manifest: UpdateManifest) -> None:
        if not manifest.update_id.strip() or not manifest.version.strip():
            raise ValueError("Update identity and version are required.")
        if manifest.component not in self._allowed_components:
            raise PermissionError("Component is not approved for updating.")
        if not manifest.signature or not manifest.signed_payload:
            raise ValueError("Signed update metadata is required.")
        if not self._verify_signature(manifest.signed_payload, manifest.signature):
            raise PermissionError("Update signature verification failed.")
        if not manifest.files:
            raise ValueError("Update contains no files.")

        paths: set[str] = set()
        for item in manifest.files:
            relative = self._safe_relative_path(item.relative_path)
            normalized = relative.as_posix().casefold()
            if normalized in paths:
                raise ValueError("Update contains duplicate file paths.")
            paths.add(normalized)
            actual = hashlib.sha256(item.content).hexdigest()
            if actual != item.sha256:
                raise ValueError("Update file hash verification failed.")
            destination = (self.root / relative).resolve()
            if self.root not in destination.parents:
                raise ValueError("Update path escapes the installation root.")

    def apply(
        self,
        manifest: UpdateManifest,
        *,
        compatibility_check: Callable[[UpdateManifest], bool],
    ) -> UpdateResult:
        self.verify(manifest)
        if not compatibility_check(manifest):
            raise ValueError("Update is incompatible with this installation.")

        stage = Path(tempfile.mkdtemp(prefix=".aviy-stage-", dir=self.root))
        backup = Path(tempfile.mkdtemp(prefix=".aviy-backup-", dir=self.root))
        applied: list[Path] = []
        backed_up: list[Path] = []
        try:
            for item in manifest.files:
                relative = self._safe_relative_path(item.relative_path)
                staged = stage / relative
                staged.parent.mkdir(parents=True, exist_ok=True)
                staged.write_bytes(item.content)

            for item in manifest.files:
                relative = self._safe_relative_path(item.relative_path)
                destination = self.root / relative
                staged = stage / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    saved = backup / relative
                    saved.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(destination, saved)
                    backed_up.append(relative)
                os.replace(staged, destination)
                applied.append(relative)

            return UpdateResult(
                manifest.update_id,
                tuple(path.as_posix() for path in applied),
                False,
            )
        except Exception:
            for relative in reversed(applied):
                destination = self.root / relative
                saved = backup / relative
                if saved.exists():
                    os.replace(saved, destination)
                else:
                    destination.unlink(missing_ok=True)
            raise
        finally:
            shutil.rmtree(stage, ignore_errors=True)
            shutil.rmtree(backup, ignore_errors=True)
