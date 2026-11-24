import json
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from aviy.api.app import create_app
from aviy.infrastructure.backup import (
    create_backup,
    create_release_archive,
    restore_backup,
    rollback_release,
    verify_backup,
)
from aviy.research.programs import compare_benchmarks


def test_api_health_and_authentication():
    client = TestClient(
        create_app(lambda token: {"user": "tester"} if token == "valid" else None)
    )
    assert client.get("/health").status_code == 200
    assert client.get("/v1/status").status_code == 401
    assert (
        client.get("/v1/status", headers={"Authorization": "Bearer valid"}).status_code
        == 200
    )


def test_api_openapi_and_interface():
    client = TestClient(create_app(lambda token: True))
    assert client.get("/openapi.json").status_code == 200
    page = client.get("/")
    assert page.status_code == 200
    assert 'lang="en"' in page.text
    assert 'aria-live="polite"' in page.text


class _BatchResearchScheduler:
    def __init__(self, accepted=True):
        self.accepted = accepted
        self.jobs = []

    def submit(self, job_id, task):
        if not self.accepted:
            return {"accepted": False, "queue_size": len(self.jobs)}
        self.jobs.append((job_id, task))
        return {"accepted": True, "queue_size": len(self.jobs)}

    def run_next(self):
        job_id, task = self.jobs.pop(0)
        return {"job_id": job_id, "result": task()}


def test_research_endpoint_requires_auth():
    client = TestClient(create_app(lambda token: token == "ok"))
    assert (
        client.post("/v1/research/requests", json={"topic": "test"}).status_code
        == 401
    )
    response = client.post(
        "/v1/research/requests",
        json={"topic": "test"},
        headers={"Authorization": "Bearer ok"},
    )
    assert response.status_code == 503


def test_research_endpoint_queues_job_without_running_it():
    calls = []
    scheduler = _BatchResearchScheduler()
    app = create_app(
        lambda token: token == "ok",
        research_handler=lambda payload: calls.append(payload),
        research_scheduler=cast(Any, scheduler),
    )
    response = TestClient(app).post(
        "/v1/research/requests",
        json={"topic": "test topic"},
        headers={"Authorization": "Bearer ok"},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["accepted"] is True
    assert body["status"] == "queued"
    assert body["job_id"]
    assert calls == []
    assert len(scheduler.jobs) == 1

    result = scheduler.run_next()
    assert result["job_id"] == body["job_id"]
    assert calls == [{"topic": "test topic"}]


def test_research_endpoint_reports_queue_rejection():
    scheduler = _BatchResearchScheduler(accepted=False)
    app = create_app(
        lambda token: token == "ok",
        research_handler=lambda payload: payload,
        research_scheduler=cast(Any, scheduler),
    )
    response = TestClient(app).post(
        "/v1/research/requests",
        json={"topic": "test"},
        headers={"Authorization": "Bearer ok"},
    )

    assert response.status_code == 429
    assert scheduler.jobs == []


def test_backup_verify_restore(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "sample.txt").write_text("Aviy", encoding="utf-8")
    backup = tmp_path / "backup"
    create_backup(source, backup)
    assert verify_backup(backup)
    restored = tmp_path / "restored"
    assert restore_backup(backup, restored)
    assert (restored / "sample.txt").read_text(encoding="utf-8") == "Aviy"


def test_backup_detects_tampering(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "data.txt").write_text("original")
    backup = tmp_path / "backup"
    create_backup(source, backup)
    (backup / "data.txt").write_text("tampered")
    with pytest.raises(ValueError):
        verify_backup(backup)


def test_backup_rejects_traversal_manifest(tmp_path):
    backup = tmp_path / "backup"
    backup.mkdir()
    (backup / "manifest.json").write_text(json.dumps({"../outside": "abc"}))
    with pytest.raises(ValueError):
        verify_backup(backup)


def test_release_archive_and_rollback(tmp_path):
    source = tmp_path / "release"
    source.mkdir()
    (source / "app.txt").write_text("release-one")
    archive = tmp_path / "release.zip"
    create_release_archive(source, archive)
    target = tmp_path / "rolled-back"
    assert rollback_release(archive, target)
    assert (target / "app.txt").read_text() == "release-one"


def test_benchmark_comparison_records_provenance(tmp_path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "report.json"
    baseline.write_text(json.dumps({"metrics": {"accuracy": 0.7}}))
    candidate.write_text(json.dumps({"metrics": {"accuracy": 0.8}}))
    report = compare_benchmarks(baseline, candidate, output)
    assert report["comparisons"]["accuracy"]["difference"] == pytest.approx(0.1)
    assert len(report["baseline_sha256"]) == 64
    assert output.is_file()
