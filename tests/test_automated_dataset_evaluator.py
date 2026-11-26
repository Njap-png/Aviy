import json
from pathlib import Path

from aviy.corpus_governance.dataset_evaluator import (
    ALLOWED_LICENSES,
    evaluate_records,
    evaluate_source_metadata,
)


def metadata():
    return {
        "source_id": "pilot_source",
        "source_name": "Pilot dataset",
        "source_uri": "https://example.org/dataset",
        "revision": "revision-1",
        "license_id": "CC-BY-4.0",
        "license_evidence_uri": "https://example.org/license",
        "terms_uri": "https://example.org/terms",
        "attribution_text": "Attribution required",
        "rights_scope": "dataset_and_contents",
        "underlying_rights_status": "covered",
    }


def record(record_id="r1", text=None):
    return {
        "record_id": record_id,
        "text": text or (
            "This is a sufficiently long educational text about "
            "programming concepts and software engineering."
        ),
        "language": "en",
        "script": "Latn",
        "source_id": "pilot_source",
        "source_uri": "https://example.org/dataset/item/1",
        "source_revision": "revision-1",
        "license_id": "CC-BY-4.0",
    }


def write_inputs(tmp_path: Path, rows, meta=None):
    records_path = tmp_path / "records.jsonl"
    metadata_path = tmp_path / "metadata.json"
    records_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(meta or metadata()),
        encoding="utf-8",
    )
    return records_path, metadata_path


def test_license_allowlist_is_explicit():
    assert "CC-BY-4.0" in ALLOWED_LICENSES
    assert "UNKNOWN" not in ALLOWED_LICENSES


def test_valid_metadata_has_no_blockers():
    assert evaluate_source_metadata(metadata()) == []


def test_unknown_license_is_blocked():
    meta = metadata()
    meta["license_id"] = "UNKNOWN"
    assert any("allowlist" in issue for issue in evaluate_source_metadata(meta))


def test_unresolved_underlying_rights_are_blocked():
    meta = metadata()
    meta["underlying_rights_status"] = "unknown"
    assert evaluate_source_metadata(meta)


def test_valid_records_only_enter_staging(tmp_path):
    records, meta = write_inputs(tmp_path, [record()])
    report = evaluate_records(records, json.loads(meta.read_text()), tmp_path / "out")
    assert report["status"] == "ACCEPTED_FOR_CORRECTED_STAGING"
    assert report["records_staged"] == 1
    assert report["pretraining_authorized"] is False
    assert report["tokenizer_training_authorized"] is False


def test_duplicate_records_are_quarantined(tmp_path):
    records, meta = write_inputs(tmp_path, [record("r1"), record("r2")])
    report = evaluate_records(records, json.loads(meta.read_text()), tmp_path / "out", minimum_score=40.0)
    assert report["records_staged"] == 1
    assert report["records_quarantined"] == 1


def test_email_record_is_quarantined(tmp_path):
    rows = [record(text="Contact person@example.org for more educational information.")]
    records, meta = write_inputs(tmp_path, rows)
    report = evaluate_records(records, json.loads(meta.read_text()), tmp_path / "out")
    assert report["records_staged"] == 0
    assert report["records_quarantined"] == 1


def test_source_metadata_failure_blocks_all_staging(tmp_path):
    meta = metadata()
    meta["underlying_rights_status"] = "unresolved"
    records, metadata_path = write_inputs(tmp_path, [record()], meta)
    report = evaluate_records(
        records, json.loads(metadata_path.read_text()), tmp_path / "out"
    )
    assert report["status"] == "HOLD_NOT_ACCEPTED"
    assert report["records_staged"] == 0


def test_wrong_revision_is_quarantined(tmp_path):
    row = record()
    row["source_revision"] = "different-revision"
    records, meta = write_inputs(tmp_path, [row])
    report = evaluate_records(records, json.loads(meta.read_text()), tmp_path / "out")
    assert report["records_staged"] == 0
    assert report["records_quarantined"] == 1


def test_report_does_not_claim_legal_verification(tmp_path):
    records, meta = write_inputs(tmp_path, [record()])
    evaluate_records(records, json.loads(meta.read_text()), tmp_path / "out")
    report = json.loads((tmp_path / "out" / "evaluation_report.json").read_text())
    assert report["legal_rights_independently_verified"] is False
    assert report["production_source_registry_changed"] is False
    assert report["training_started"] is False


def test_record_uri_cannot_escape_source_host_or_path(tmp_path):
    from aviy.corpus_governance.dataset_evaluator import uri_belongs_to_source

    base = "https://example.org/dataset"
    assert uri_belongs_to_source("https://example.org/dataset/item/1", base)
    assert uri_belongs_to_source("https://example.org/dataset", base)
    assert not uri_belongs_to_source("https://example.org.evil.test/dataset/item", base)
    assert not uri_belongs_to_source("https://evil.test/dataset/item", base)
    assert not uri_belongs_to_source("https://example.org/datasets/item", base)
    assert not uri_belongs_to_source("https://user@example.org/dataset/item", base)


def test_existing_output_is_never_overwritten(tmp_path):
    import pytest
    records, meta = write_inputs(tmp_path, [record()])
    output = tmp_path / "out"
    output.mkdir()
    existing = output / "corrected_accepted.jsonl"
    existing.write_text("KEEP THIS FILE", encoding="utf-8")

    with pytest.raises(FileExistsError):
        evaluate_records(records, json.loads(meta.read_text()), output)

    assert existing.read_text(encoding="utf-8") == "KEEP THIS FILE"


def test_attribution_and_audit_hashes_are_preserved(tmp_path):
    records, meta = write_inputs(tmp_path, [record()])
    output = tmp_path / "out"
    report = evaluate_records(records, json.loads(meta.read_text()), output)

    assert (output / "attribution_notice.txt").read_text(encoding="utf-8").strip() == "Attribution required"
    assert report["attribution_text_preserved"] is True
    assert report["input_records_sha256"]
    assert report["source_metadata_sha256"]
    assert report["output_sha256"]["corrected_accepted.jsonl"]
    assert report["output_sha256"]["attribution_notice.txt"]


def test_plain_numeric_identifier_is_not_automatically_phone_flagged():
    from aviy.corpus_governance.dataset_evaluator import privacy_issues

    assert "possible_phone_number" not in privacy_issues(
        "The dataset identifier is 202420252026 and is used for indexing."
    )


def test_invalid_url_forms_are_rejected():
    from aviy.corpus_governance.dataset_evaluator import valid_http_url

    assert not valid_http_url("https:///missing-host")
    assert not valid_http_url("https://user:password@example.org/data")
    assert not valid_http_url("file:///tmp/data")
    assert valid_http_url("https://example.org/data")
