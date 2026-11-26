import hashlib
import json
from pathlib import Path

from typer.testing import CliRunner

from aviy.cli.main import app
from aviy.corpus_governance import (
    CorpusRecord,
    SourceRegistration,
    SourceRegistry,
    build_manifest,
)

runner = CliRunner()


def write_registry(path, registry):
    path.write_text(json.dumps(registry.to_dict()), encoding="utf-8")


def write_records(path, records):
    path.write_text(
        "".join(json.dumps(vars(record)) + "\n" for record in records),
        encoding="utf-8",
    )


def approved_fixture():
    registry = SourceRegistry()
    registry.add(SourceRegistration(
        source_id="test-source",
        source_name="Test Source",
        source_uri="https://example.org/data",
        revision="v1",
        license_id="CC-BY-4.0",
        license_status="approved",
        approval_status="approved",
        training_allowed=True,
        attribution_required=False,
        reviewer="test-reviewer",
        approved_at="2026-01-01T00:00:00Z",
    ))

    records = []
    for index in range(20):
        content = f"This is a unique governance test record number {index}."
        records.append(CorpusRecord(
            record_id=f"record-{index}",
            text=content,
            language="en",
            script="Latn",
            category="general",
            source_type="curated",
            source_id="test-source",
            source_name="Test Source",
            source_uri="https://example.org/data",
            source_revision="v1",
            license_id="CC-BY-4.0",
            attribution="",
            group_id=f"group-{index}",
            authored_at="2026-01-01T00:00:00Z",
            content_sha256=hashlib.sha256(content.encode()).hexdigest(),
            review_status="reviewed",
            privacy_status="clear",
            approval_status="approved",
            reviewer="test-reviewer",
            reviewed_at="2026-01-01T00:00:00Z",
            review_notes="Reviewed test fixture.",
        ))
    return registry, records


def test_sources_empty_registry(tmp_path):
    write_registry(tmp_path / "registry.json", SourceRegistry())
    result = runner.invoke(app, [
        "corpus", "sources", "--root", str(tmp_path),
        "--registry", "registry.json",
    ])
    assert result.exit_code == 0, result.output
    assert "No sources are registered." in result.output


def test_sources_registered_source_read_only(tmp_path):
    registry, _ = approved_fixture()
    path = tmp_path / "registry.json"
    write_registry(path, registry)
    original = path.read_bytes()

    result = runner.invoke(app, [
        "corpus", "sources", "--root", str(tmp_path),
        "--registry", "registry.json",
    ])
    assert result.exit_code == 0, result.output
    assert "test-source" in result.output
    assert path.read_bytes() == original


def test_validate_pending_record_read_only(tmp_path):
    write_registry(tmp_path / "registry.json", SourceRegistry())
    content = "This is a pending record for validation testing."
    record = CorpusRecord(
        record_id="pending-1",
        text=content,
        language="en",
        script="Latn",
        category="general",
        source_type="curated",
        source_id="missing-source",
        source_name="Missing Source",
        source_uri="https://example.org/missing",
        source_revision="v1",
        license_id="unknown",
        attribution="",
        group_id="group-1",
        authored_at="",
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
    )
    path = tmp_path / "records.jsonl"
    write_records(path, [record])
    original = path.read_bytes()

    result = runner.invoke(app, [
        "corpus", "validate", "--root", str(tmp_path),
        "--registry", "registry.json", "--records", "records.jsonl",
    ])
    assert result.exit_code == 1
    assert "invalid" in result.output.lower()
    assert path.read_bytes() == original


def test_audit_read_only(tmp_path):
    registry, records = approved_fixture()
    write_registry(tmp_path / "registry.json", registry)
    path = tmp_path / "records.jsonl"
    write_records(path, records)
    original = path.read_bytes()

    result = runner.invoke(app, [
        "corpus", "audit", "--root", str(tmp_path),
        "--registry", "registry.json", "--records", "records.jsonl",
    ])
    assert result.exit_code == 0, result.output
    assert "Total records: 20" in result.output
    assert path.read_bytes() == original


def test_manifest_read_only(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "schema_version": "1.0", "record_count": 0
    }), encoding="utf-8")
    original = path.read_bytes()

    result = runner.invoke(app, [
        "corpus", "manifest", "--root", str(tmp_path),
        "--manifest", "manifest.json",
    ])
    assert result.exit_code == 0, result.output
    assert "schema_version: 1.0" in result.output
    assert path.read_bytes() == original


def test_verify_matching_manifest(tmp_path):
    registry, records = approved_fixture()
    write_registry(tmp_path / "registry.json", registry)
    write_records(tmp_path / "records.jsonl", records)
    manifest = build_manifest(records, registry)
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    result = runner.invoke(app, [
        "corpus", "verify", "--root", str(tmp_path),
        "--registry", "registry.json", "--records", "records.jsonl",
        "--manifest", "manifest.json",
    ])
    assert result.exit_code == 0, result.output
    assert "Manifest verification passed." in result.output


def test_verify_rejects_tampered_manifest(tmp_path):
    registry, records = approved_fixture()
    write_registry(tmp_path / "registry.json", registry)
    write_records(tmp_path / "records.jsonl", records)
    manifest = build_manifest(records, registry)
    manifest["record_count"] += 1
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    result = runner.invoke(app, [
        "corpus", "verify", "--root", str(tmp_path),
        "--registry", "registry.json", "--records", "records.jsonl",
        "--manifest", "manifest.json",
    ])
    assert result.exit_code == 1
    assert "Manifest verification failed" in result.output


def test_verify_rejects_invalid_manifest_shape(tmp_path):
    registry, records = approved_fixture()
    write_registry(tmp_path / "registry.json", registry)
    write_records(tmp_path / "records.jsonl", records)
    (tmp_path / "manifest.json").write_text("[]", encoding="utf-8")

    result = runner.invoke(app, [
        "corpus", "verify", "--root", str(tmp_path),
        "--registry", "registry.json", "--records", "records.jsonl",
        "--manifest", "manifest.json",
    ])
    assert result.exit_code == 1
    assert "Manifest must be a JSON object" in result.output


def test_existing_commands_remain_registered():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("version", "validate", "serve", "corpus"):
        assert command in result.output
