import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_METADATA = {
    "source_id": "aviy_adversarial_test",
    "source_name": "Synthetic Adversarial Test Source",
    "source_uri": "https://example.org/datasets/demo/",
    "revision": "synthetic-v1",
    "license_id": "CC0-1.0",
    "license_evidence_uri": "https://example.org/datasets/demo/license",
    "terms_uri": "https://example.org/datasets/demo/terms",
    "attribution_text": "Synthetic test fixture only; not a real dataset.",
    "rights_scope": "dataset_and_contents",
    "underlying_rights_status": "covered",
}

BASE_RECORD = {
    "record_id": "synthetic-record-001",
    "text": (
        "This synthetic educational passage explains basic mathematics "
        "and programming concepts. It exists only to test Aviy's dataset "
        "evaluation, provenance, normalization, and staging safeguards."
    ),
    "language": "en",
    "script": "Latn",
    "source_id": "aviy_adversarial_test",
    "source_uri": "https://example.org/datasets/demo/",
    "source_revision": "synthetic-v1",
    "license_id": "CC0-1.0",
}


def run_case(tmp_path, metadata=None, records=None, raw_jsonl=None):
    metadata = copy.deepcopy(metadata or BASE_METADATA)
    records = copy.deepcopy(records if records is not None else [BASE_RECORD])

    metadata_path = tmp_path / "source_metadata.json"
    records_path = tmp_path / "records.jsonl"
    output_path = tmp_path / "output"

    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False),
        encoding="utf-8",
    )

    if raw_jsonl is not None:
        records_path.write_text(raw_jsonl, encoding="utf-8")
    else:
        records_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
            encoding="utf-8",
        )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aviy.corpus_governance.dataset_evaluator",
            "--records",
            str(records_path),
            "--metadata",
            str(metadata_path),
            "--output",
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    return result, output_path


def read_report(output_path):
    return json.loads(
        (output_path / "evaluation_report.json").read_text(encoding="utf-8")
    )


def assert_held(tmp_path, metadata=None, records=None):
    result, output = run_case(
        tmp_path,
        metadata=metadata,
        records=records,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr

    report = read_report(output)
    assert report["status"] == "HOLD_NOT_ACCEPTED", report
    assert report["records_staged"] == 0, report
    assert report["records_quarantined"] >= 1, report
    return report, output


def test_rejects_lookalike_hostname(tmp_path):
    record = copy.deepcopy(BASE_RECORD)
    record["source_uri"] = "https://example.org.evil.test/datasets/demo/"
    assert_held(tmp_path, records=[record])


def test_rejects_unrelated_hostname(tmp_path):
    record = copy.deepcopy(BASE_RECORD)
    record["source_uri"] = "https://evil.test/datasets/demo/"
    assert_held(tmp_path, records=[record])


def test_rejects_sibling_path_outside_source_boundary(tmp_path):
    record = copy.deepcopy(BASE_RECORD)
    record["source_uri"] = "https://example.org/datasets/demonstration/"
    assert_held(tmp_path, records=[record])


def test_rejects_scheme_mismatch(tmp_path):
    record = copy.deepcopy(BASE_RECORD)
    record["source_uri"] = "http://example.org/datasets/demo/"
    assert_held(tmp_path, records=[record])


def test_rejects_embedded_url_credentials(tmp_path):
    record = copy.deepcopy(BASE_RECORD)
    record["source_uri"] = (
        "https://username:password@example.org/datasets/demo/"
    )
    assert_held(tmp_path, records=[record])


def test_rejects_missing_attribution(tmp_path):
    metadata = copy.deepcopy(BASE_METADATA)
    metadata["attribution_text"] = ""
    assert_held(tmp_path, metadata=metadata)


def test_rejects_invalid_metadata_url_with_credentials(tmp_path):
    metadata = copy.deepcopy(BASE_METADATA)
    metadata["license_evidence_uri"] = (
        "https://user:secret@example.org/license"
    )
    assert_held(tmp_path, metadata=metadata)


def test_quarantines_empty_text(tmp_path):
    record = copy.deepcopy(BASE_RECORD)
    record["text"] = ""
    assert_held(tmp_path, records=[record])


def test_quarantines_short_text(tmp_path):
    record = copy.deepcopy(BASE_RECORD)
    record["text"] = "Too short."
    assert_held(tmp_path, records=[record])


def test_quarantines_duplicate_record_id(tmp_path):
    first = copy.deepcopy(BASE_RECORD)
    second = copy.deepcopy(BASE_RECORD)
    second["text"] = (
        "A different synthetic educational passage about science, "
        "software engineering, and responsible dataset preparation."
    )
    assert_held(tmp_path, records=[first, second])


def test_quarantines_duplicate_normalized_content(tmp_path):
    first = copy.deepcopy(BASE_RECORD)
    second = copy.deepcopy(BASE_RECORD)
    second["record_id"] = "synthetic-record-002"
    second["text"] = "  " + first["text"].replace("  ", " ") + "  "
    assert_held(tmp_path, records=[first, second])


def test_quarantines_email_address(tmp_path):
    record = copy.deepcopy(BASE_RECORD)
    record["text"] = (
        "This synthetic privacy test contains test@example.com "
        "to verify that personal contact information is quarantined."
    )
    assert_held(tmp_path, records=[record])


def test_quarantines_source_revision_mismatch(tmp_path):
    record = copy.deepcopy(BASE_RECORD)
    record["source_revision"] = "unregistered-revision"
    assert_held(tmp_path, records=[record])


def test_quarantines_source_id_mismatch(tmp_path):
    record = copy.deepcopy(BASE_RECORD)
    record["source_id"] = "different-source"
    assert_held(tmp_path, records=[record])


def test_output_hashes_match_generated_files(tmp_path):
    result, output = run_case(tmp_path)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr

    report = read_report(output)
    for filename, expected_hash in report["output_sha256"].items():
        actual_hash = hashlib.sha256(
            (output / filename).read_bytes()
        ).hexdigest()
        assert actual_hash == expected_hash, filename

    assert report["input_records_sha256"]
    assert report["source_metadata_sha256"]
    assert report["attribution_text_preserved"] is True


def test_refuses_to_overwrite_existing_outputs(tmp_path):
    first_result, output = run_case(tmp_path)
    assert first_result.returncode == 0, (
        first_result.stdout + "\n" + first_result.stderr
    )

    report_path = output / "evaluation_report.json"
    original_hash = hashlib.sha256(report_path.read_bytes()).hexdigest()

    second_result, _ = run_case(tmp_path)
    assert second_result.returncode != 0

    final_hash = hashlib.sha256(report_path.read_bytes()).hexdigest()
    assert final_hash == original_hash


def test_malformed_jsonl_is_not_silently_accepted(tmp_path):
    raw = (
        '{"record_id":"valid"}\n'
        '{"record_id": this-is-not-json }\n'
    )
    result, output = run_case(tmp_path, raw_jsonl=raw)

    if result.returncode == 0:
        report = read_report(output)
        assert report["status"] == "HOLD_NOT_ACCEPTED"
        assert report["records_staged"] == 0
    else:
        combined = result.stdout + "\n" + result.stderr
        assert "JSON" in combined or "json" in combined or "Expecting" in combined
