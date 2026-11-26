import json

import pytest

from aviy.data_ingestion.service import intake_bytes, read_history

METADATA = {
    "source_id": "synthetic_ingestion_test",
    "source_name": "Synthetic ingestion test",
    "source_uri": "https://example.org/ingestion/",
    "revision": "test-v1",
    "license_id": "CC0-1.0",
    "license_evidence_uri": "https://example.org/ingestion/license",
    "terms_uri": "https://example.org/ingestion/terms",
    "attribution_text": "Synthetic test material.",
    "rights_scope": "dataset_and_contents",
    "underlying_rights_status": "covered",
    "language": "en",
    "script": "Latn",
}

PASSAGE = (
    "This synthetic educational passage explains mathematics and software "
    "engineering for automated Aviy ingestion tests."
)


def run(tmp_path, filename, content, fmt=None, metadata=None):
    return intake_bytes(
        filename,
        content,
        metadata or METADATA,
        storage_root=tmp_path / "storage",
        file_format=fmt,
    )


@pytest.mark.parametrize("fmt,filename,content", [
    ("jsonl", "records.jsonl", (json.dumps({"text": PASSAGE}) + "\n").encode()),
    ("json", "records.json", json.dumps([{"text": PASSAGE}]).encode()),
    ("csv", "records.csv", ("text,category\n" + PASSAGE + ",education\n").encode()),
    ("txt", "records.txt", (PASSAGE + "\n").encode()),
])
def test_all_supported_formats_run_existing_evaluator(tmp_path, fmt, filename, content):
    result = run(tmp_path, filename, content, fmt)

    assert result["input_format"] == fmt
    assert result["records_received"] == 1
    assert result["original_file_preserved"] is True
    assert result["production_source_registry_changed"] is False
    assert result["records_approved_for_training"] is False
    assert result["tokenizer_training_authorized"] is False
    assert result["pretraining_authorized"] is False
    assert result["training_started"] is False
    assert result["status"] in {"COMPLETED", "COMPLETED_WITH_FAILURES"}

    run_dir = tmp_path / "storage" / "runs" / result["ingestion_id"]
    original = run_dir / "input" / filename
    normalized = run_dir / "input" / "normalized_records.jsonl"
    assert original.read_bytes() == content
    assert normalized.is_file()
    report = json.loads((run_dir / "ingestion_report.json").read_text())
    assert report["input_sha256"] == result["input_sha256"]


def test_txt_creates_one_record_per_nonempty_line(tmp_path):
    content = (PASSAGE + "\n\n" + PASSAGE.replace("mathematics", "physics") + "\n").encode()
    result = run(tmp_path, "two.txt", content)
    assert result["records_received"] == 2


def test_rejects_incomplete_metadata_before_creating_intake(tmp_path):
    metadata = dict(METADATA)
    metadata.pop("terms_uri")
    with pytest.raises(ValueError, match="Metadata validation failed"):
        run(tmp_path, "records.txt", PASSAGE.encode(), metadata=metadata)
    assert not (tmp_path / "storage" / "runs").exists()


def test_rejects_unsupported_format(tmp_path):
    with pytest.raises(ValueError, match="Unsupported format"):
        run(tmp_path, "records.xml", b"<records/>")


def test_rejects_csv_without_text_column(tmp_path):
    with pytest.raises(ValueError, match="text"):
        run(tmp_path, "records.csv", b"title,body\nA,B\n")


def test_rejects_invalid_jsonl_record(tmp_path):
    with pytest.raises(ValueError, match="Invalid JSONL"):
        run(tmp_path, "records.jsonl", b"{broken\n")


def test_rejects_empty_or_oversized_input(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        run(tmp_path, "empty.txt", b"")
    with pytest.raises(ValueError, match="50 MiB"):
        run(tmp_path, "large.txt", b"x" * (50 * 1024 * 1024 + 1))


def test_rejects_path_in_filename(tmp_path):
    with pytest.raises(ValueError, match="directory path"):
        run(tmp_path, "../records.txt", PASSAGE.encode())


def test_history_records_completed_intake(tmp_path):
    result = intake_bytes(
        "records.txt", PASSAGE.encode(), METADATA,
        storage_root=tmp_path / "storage",
    )
    history = read_history(tmp_path / "storage")
    assert any(item["ingestion_id"] == result["ingestion_id"] for item in history)


def test_existing_source_metadata_claims_are_not_legal_verification(tmp_path):
    result = run(tmp_path, "records.txt", PASSAGE.encode())
    report_path = tmp_path / "storage" / "runs" / result["ingestion_id"] / "evaluation" / "batch_report.json"
    report = json.loads(report_path.read_text())
    assert report["legal_rights_independently_verified"] is False
