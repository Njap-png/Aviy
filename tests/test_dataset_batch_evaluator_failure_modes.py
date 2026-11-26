import json
import subprocess
from pathlib import Path

import aviy.corpus_governance.dataset_batch_evaluator as batch_module


METADATA = {
    "source_id": "aviy_batch_failure_test",
    "source_name": "Synthetic Failure Test",
    "source_uri": "https://example.org/failure-test/",
    "revision": "synthetic-v1",
    "license_id": "CC0-1.0",
    "license_evidence_uri": "https://example.org/failure-test/license",
    "terms_uri": "https://example.org/failure-test/terms",
    "attribution_text": "Synthetic test data only.",
    "rights_scope": "dataset_and_contents",
    "underlying_rights_status": "covered",
}

TEXT = (
    "This synthetic educational passage explains mathematics and programming. "
    "It exists only to test safe batch evaluator failure handling."
)


def create_dataset(folder, name):
    folder.mkdir(parents=True, exist_ok=True)

    metadata_path = folder / f"{name}_metadata.json"
    records_path = folder / f"{name}_records.jsonl"

    metadata_path.write_text(json.dumps(METADATA), encoding="utf-8")

    record = {
        "record_id": f"{name}-001",
        "text": TEXT,
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


def test_failed_dataset_does_not_stop_remaining_datasets(tmp_path, monkeypatch):
    first = create_dataset(tmp_path, "forced_failure")
    second = create_dataset(tmp_path, "later_success")
    manifest = write_manifest(tmp_path, [first, second])

    real_run = subprocess.run
    calls = []

    def controlled_run(command, **kwargs):
        records_path = command[command.index("--records") + 1]
        calls.append(records_path)

        if "forced_failure_records.jsonl" in records_path:
            return subprocess.CompletedProcess(
                command, 9, stdout="", stderr="synthetic evaluator failure"
            )

        return real_run(command, **kwargs)

    monkeypatch.setattr(batch_module.subprocess, "run", controlled_run)

    output = tmp_path / "batch_output"
    report = batch_module.run_batch(manifest, output)

    assert len(calls) == 2
    assert report["status"] == "COMPLETED_WITH_FAILURES"
    assert report["datasets_total"] == 2
    assert report["datasets_failed"] == 1
    assert report["datasets_completed"] == 1

    assert report["datasets"][0]["status"] == "FAILED"
    assert report["datasets"][0]["return_code"] == 9
    assert report["datasets"][0]["report_sha256"] is None

    assert report["datasets"][1]["status"] == "COMPLETED"
    assert report["datasets"][1]["return_code"] == 0

    saved_report = json.loads(
        (output / "batch_report.json").read_text(encoding="utf-8")
    )
    assert saved_report["status"] == "COMPLETED_WITH_FAILURES"
    assert saved_report["datasets_failed"] == 1


def test_operating_system_error_is_recorded_and_batch_continues(
    tmp_path, monkeypatch
):
    first = create_dataset(tmp_path, "os_error")
    second = create_dataset(tmp_path, "after_os_error")
    manifest = write_manifest(tmp_path, [first, second])

    real_run = subprocess.run
    calls = []

    def controlled_run(command, **kwargs):
        records_path = command[command.index("--records") + 1]
        calls.append(records_path)

        if Path(records_path).name == "os_error_records.jsonl":
            raise OSError("synthetic process launch failure")

        return real_run(command, **kwargs)

    monkeypatch.setattr(batch_module.subprocess, "run", controlled_run)

    report = batch_module.run_batch(manifest, tmp_path / "batch_output")

    assert len(calls) == 2
    assert report["status"] == "COMPLETED_WITH_FAILURES"
    assert report["datasets_failed"] == 1
    assert report["datasets_completed"] == 1

    assert report["datasets"][0]["status"] == "FAILED"
    assert report["datasets"][0]["return_code"] == -1
    assert report["datasets"][0]["report_sha256"] is None

    assert report["datasets"][1]["status"] == "COMPLETED"


def test_zero_exit_without_readable_report_is_failure(
    tmp_path, monkeypatch
):
    first = create_dataset(tmp_path, "missing_report")
    second = create_dataset(tmp_path, "valid_report")
    manifest = write_manifest(tmp_path, [first, second])

    real_run = subprocess.run
    calls = []

    def controlled_run(command, **kwargs):
        records_path = command[command.index("--records") + 1]
        calls.append(records_path)

        if "missing_report_records.jsonl" in records_path:
            return subprocess.CompletedProcess(
                command, 0, stdout="", stderr=""
            )

        return real_run(command, **kwargs)

    monkeypatch.setattr(batch_module.subprocess, "run", controlled_run)

    report = batch_module.run_batch(manifest, tmp_path / "batch_output")

    assert len(calls) == 2
    assert report["status"] == "COMPLETED_WITH_FAILURES"
    assert report["datasets_failed"] == 1
    assert report["datasets_completed"] == 1

    assert report["datasets"][0]["status"] == "FAILED"
    assert report["datasets"][0]["return_code"] == 0
    assert report["datasets"][0]["evaluator_status"] is None
    assert report["datasets"][0]["report_sha256"] is None

    assert report["datasets"][1]["status"] == "COMPLETED"
