import json

import pytest

from aviy.corpus_governance.dataset_batch_evaluator import (
    load_batch_manifest,
    run_batch,
)


METADATA = {
    "source_id": "aviy_batch_synthetic",
    "source_name": "Synthetic Batch Test",
    "source_uri": "https://example.org/batch-test/",
    "revision": "synthetic-v1",
    "license_id": "CC0-1.0",
    "license_evidence_uri": "https://example.org/batch-test/license",
    "terms_uri": "https://example.org/batch-test/terms",
    "attribution_text": "Synthetic test data only.",
    "rights_scope": "dataset_and_contents",
    "underlying_rights_status": "covered",
}

TEXT = (
    "This synthetic educational passage explains mathematics and programming. "
    "It exists only to test the Aviy dataset batch evaluator."
)


def create_dataset(folder, name, text=TEXT):
    folder.mkdir(parents=True, exist_ok=True)
    metadata_path = folder / f"{name}_metadata.json"
    records_path = folder / f"{name}_records.jsonl"

    metadata_path.write_text(json.dumps(METADATA), encoding="utf-8")
    record = {
        "record_id": f"{name}-001",
        "text": text,
        "language": "en",
        "script": "Latn",
        "source_id": METADATA["source_id"],
        "source_uri": METADATA["source_uri"],
        "source_revision": METADATA["revision"],
        "license_id": METADATA["license_id"],
    }
    records_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    return {
        "name": name,
        "records": records_path.name,
        "metadata": metadata_path.name,
    }


def write_manifest(folder, entries):
    path = folder / "batch_manifest.json"
    path.write_text(
        json.dumps({"schema_version": "1.0", "datasets": entries}),
        encoding="utf-8",
    )
    return path


def test_batch_evaluates_multiple_datasets(tmp_path):
    first = create_dataset(tmp_path, "english")
    second = create_dataset(tmp_path, "swahili")
    manifest = write_manifest(tmp_path, [first, second])

    report = run_batch(manifest, tmp_path / "batch_output")

    assert report["status"] == "COMPLETED"
    assert report["datasets_total"] == 2
    assert report["datasets_completed"] == 2
    assert report["datasets_failed"] == 0
    assert (tmp_path / "batch_output" / "batch_report.json").is_file()
    assert (
        tmp_path / "batch_output" / "english" / "evaluation_report.json"
    ).is_file()
    assert (
        tmp_path / "batch_output" / "swahili" / "evaluation_report.json"
    ).is_file()


def test_batch_aggregates_accepted_and_quarantined_counts(tmp_path):
    clean = create_dataset(tmp_path, "clean")
    flagged = create_dataset(
        tmp_path,
        "flagged",
        "This synthetic privacy test contains test@example.com for screening.",
    )
    manifest = write_manifest(tmp_path, [clean, flagged])

    report = run_batch(manifest, tmp_path / "batch_output")

    assert report["status"] == "COMPLETED"
    assert report["records_staged"] == 1
    assert report["records_quarantined"] == 1


def test_batch_report_has_hashes_and_safety_flags(tmp_path):
    entry = create_dataset(tmp_path, "one")
    manifest = write_manifest(tmp_path, [entry])

    report = run_batch(manifest, tmp_path / "batch_output")

    assert len(report["manifest_sha256"]) == 64
    assert len(report["datasets"][0]["records_input_sha256"]) == 64
    assert len(report["datasets"][0]["metadata_input_sha256"]) == 64
    assert len(report["datasets"][0]["report_sha256"]) == 64
    assert report["production_source_registry_changed"] is False
    assert report["tokenizer_training_authorized"] is False
    assert report["pretraining_authorized"] is False
    assert report["training_started"] is False
    assert report["legal_rights_independently_verified"] is False


def test_refuses_existing_output_directory(tmp_path):
    entry = create_dataset(tmp_path, "one")
    manifest = write_manifest(tmp_path, [entry])
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        run_batch(manifest, output)

    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_rejects_duplicate_dataset_names(tmp_path):
    first = create_dataset(tmp_path, "one")
    second = create_dataset(tmp_path, "two")
    second["name"] = "one"
    manifest = write_manifest(tmp_path, [first, second])

    with pytest.raises(ValueError, match="Duplicate dataset name"):
        load_batch_manifest(manifest)


def test_rejects_path_traversal_dataset_name(tmp_path):
    entry = create_dataset(tmp_path, "one")
    entry["name"] = "../escape"
    manifest = write_manifest(tmp_path, [entry])

    with pytest.raises(ValueError, match="invalid name"):
        load_batch_manifest(manifest)


def test_rejects_missing_input(tmp_path):
    manifest = write_manifest(
        tmp_path,
        [{"name": "missing", "records": "missing.jsonl", "metadata": "missing.json"}],
    )

    with pytest.raises(ValueError, match="does not exist"):
        load_batch_manifest(manifest)


def test_rejects_empty_dataset_list(tmp_path):
    manifest = write_manifest(tmp_path, [])

    with pytest.raises(ValueError, match="non-empty datasets list"):
        load_batch_manifest(manifest)


def test_rejects_unsupported_manifest_version(tmp_path):
    manifest = tmp_path / "batch_manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": "9.0", "datasets": [{"name": "x"}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema_version"):
        load_batch_manifest(manifest)