"""Hash-verified backup, restoration and release archive utilities."""

import hashlib
import json
import shutil
import zipfile
from pathlib import Path


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def create_backup(source, destination):
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if not source.is_dir():
        raise ValueError("Source must be a directory.")
    if destination == source or source in destination.parents:
        raise ValueError("Backup destination cannot be inside source.")
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError("Symlinks are not permitted.")
        if path.is_file():
            name = path.relative_to(source).as_posix()
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            manifest[name] = _digest(target)
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def verify_backup(backup):
    backup = Path(backup).resolve()
    manifest_path = backup / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Backup manifest is missing.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest.items():
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Unsafe backup path.")
        target = backup / rel
        if target.is_symlink() or not target.is_file():
            raise ValueError("Missing or unsafe backup file.")
        if backup not in target.resolve().parents:
            raise ValueError("Backup path escapes its root.")
        if _digest(target) != expected:
            raise ValueError(f"Integrity failure: {name}")
    return True


def restore_backup(backup, destination):
    backup = Path(backup).resolve()
    destination = Path(destination).resolve()
    verify_backup(backup)
    if destination == backup or backup in destination.parents:
        raise ValueError("Restore destination cannot be inside backup.")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Restore destination must be empty.")
    destination.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((backup / "manifest.json").read_text(encoding="utf-8"))
    for name in manifest:
        rel = Path(name)
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backup / rel, target)
    for name, expected in manifest.items():
        if _digest(destination / name) != expected:
            raise ValueError("Restored content failed verification.")
    return True


def create_release_archive(source, archive):
    source = Path(source).resolve()
    archive = Path(archive).resolve()
    if not source.is_dir() or archive == source or source in archive.parents:
        raise ValueError("Invalid release paths.")
    manifest = {}
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in source.rglob("*"):
            if path.is_symlink():
                raise ValueError("Symlinks are not permitted.")
            if path.is_file():
                name = path.relative_to(source).as_posix()
                zf.write(path, name)
                manifest[name] = _digest(path)
        zf.writestr(
            "AVIY_RELEASE_MANIFEST.json", json.dumps(manifest, indent=2, sort_keys=True)
        )
    return manifest


def rollback_release(archive, destination):
    """Restore a release into a new or empty directory."""
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Rollback destination must be empty.")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        marker = "AVIY_RELEASE_MANIFEST.json"
        if marker not in names:
            raise ValueError("Release manifest is missing.")
        manifest = json.loads(zf.read(marker))
        for name, expected in manifest.items():
            rel = Path(name)
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("Unsafe archive path.")
            target = (destination / rel).resolve()
            if destination not in target.parents:
                raise ValueError("Archive path escapes destination.")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            if _digest(target) != expected:
                raise ValueError("Release rollback integrity failure.")
    return True
