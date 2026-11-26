import base64
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import pytest
from fastapi.testclient import TestClient

from aviy.api.app import create_app
from aviy.data_ingestion import service
from aviy.data_ingestion.service import intake_bytes

METADATA = {
    "source_id": "synthetic_security_test",
    "source_name": "Synthetic security test",
    "source_uri": "https://example.org/security/",
    "revision": "security-v1",
    "license_id": "CC0-1.0",
    "license_evidence_uri": "https://example.org/security/license",
    "terms_uri": "https://example.org/security/terms",
    "attribution_text": "Synthetic test material.",
    "rights_scope": "dataset_and_contents",
    "underlying_rights_status": "covered",
    "language": "en",
    "script": "Latn",
}

TEXT = (
    "Synthetic material used only to test Aviy ingestion validation. "
    "This test does not approve production training."
)


def fake_batch(*args, **kwargs):
    return {"status": "COMPLETED"}


@pytest.mark.parametrize(
    "filename",
    [
        "../records.txt",
        "..\\records.txt",
        "C:\\records.txt",
        "/tmp/records.txt",
        "CON.txt",
        "NUL",
        "folder:records.txt",
        "bad:name.txt",
        "records.txt ",
        "records.txt.",
        "bad\nname.txt",
        "a" * 129 + ".txt",
    ],
)
def test_rejects_unsafe_filenames(tmp_path, filename, monkeypatch):
    monkeypatch.setattr(service, "run_batch", fake_batch)

    with pytest.raises(ValueError):
        intake_bytes(
            filename,
            TEXT.encode(),
            METADATA,
            storage_root=tmp_path / "storage",
        )

    assert not (tmp_path / "storage" / "runs").exists()


def test_rejects_invalid_utf8(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "run_batch", fake_batch)

    with pytest.raises(ValueError, match="UTF-8"):
        intake_bytes(
            "records.txt",
            b"\xff\xfe\xfa",
            METADATA,
            storage_root=tmp_path / "storage",
        )

    assert not (tmp_path / "storage" / "runs").exists()


@pytest.mark.parametrize(
    "content,fmt",
    [
        (b'{"text":NaN}', "json"),
        (b'{"text":Infinity}', "jsonl"),
        (b'{"text":-Infinity}', "jsonl"),
        (b'{"records": [', "json"),
    ],
)
def test_rejects_nonstandard_or_malformed_json(
    tmp_path,
    content,
    fmt,
    monkeypatch,
):
    monkeypatch.setattr(service, "run_batch", fake_batch)

    with pytest.raises(ValueError):
        intake_bytes(
            f"records.{fmt}",
            content,
            METADATA,
            storage_root=tmp_path / "storage",
        )

    assert not (tmp_path / "storage" / "runs").exists()


@pytest.mark.parametrize(
    "content",
    [
        b"text,text\nfirst,second\n",
        b"text,category\nfirst,one,extra\n",
        b"text,category\nfirst\n",
        b"text,category\n",
    ],
)
def test_rejects_malformed_csv(tmp_path, content, monkeypatch):
    monkeypatch.setattr(service, "run_batch", fake_batch)

    with pytest.raises(ValueError):
        intake_bytes(
            "records.csv",
            content,
            METADATA,
            storage_root=tmp_path / "storage",
        )

    assert not (tmp_path / "storage" / "runs").exists()


def test_rejects_non_bytes_and_oversized_payloads(tmp_path):
    with pytest.raises(ValueError):
        intake_bytes(
            "records.txt",
            "not bytes",
            METADATA,
            storage_root=tmp_path,
        )

    with pytest.raises(ValueError, match="50 MiB"):
        intake_bytes(
            "records.txt",
            b"x" * (service.MAX_FILE_BYTES + 1),
            METADATA,
            storage_root=tmp_path,
        )


def test_duplicate_submission_reuses_existing_run(
    tmp_path,
    monkeypatch,
):
    calls = []

    def counted_batch(*args, **kwargs):
        calls.append(1)
        return fake_batch()

    monkeypatch.setattr(service, "run_batch", counted_batch)

    storage = tmp_path / "storage"

    first = intake_bytes(
        "records.txt",
        TEXT.encode(),
        METADATA,
        storage_root=storage,
    )

    second = intake_bytes(
        "records.txt",
        TEXT.encode(),
        METADATA,
        storage_root=storage,
    )

    assert first["ingestion_id"] == second["ingestion_id"]
    assert second["duplicate_submission"] is True
    assert second["reused_existing_ingestion"] is True
    assert len(calls) == 1
    assert len(service.read_history(storage)) == 1


def test_concurrent_duplicate_submissions_create_one_run(
    tmp_path,
    monkeypatch,
):
    calls = []
    calls_lock = Lock()
    storage = tmp_path / "storage"

    def counted_batch(*args, **kwargs):
        with calls_lock:
            calls.append(1)
        return fake_batch()

    monkeypatch.setattr(service, "run_batch", counted_batch)

    def submit(_):
        return intake_bytes(
            "records.txt",
            TEXT.encode(),
            METADATA,
            storage_root=storage,
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(submit, range(4)))

    assert len({item["ingestion_id"] for item in results}) == 1
    assert sum(
        item["duplicate_submission"] for item in results
    ) == 3

    assert len(calls) == 1
    assert len(service.read_history(storage)) == 1


def test_evaluation_exception_is_recorded_without_authorization(
    tmp_path,
    monkeypatch,
):
    def fail_batch(*args, **kwargs):
        raise RuntimeError("synthetic evaluator failure")

    monkeypatch.setattr(service, "run_batch", fail_batch)

    result = intake_bytes(
        "records.txt",
        TEXT.encode(),
        METADATA,
        storage_root=tmp_path / "storage",
    )

    assert result["status"] == "INGESTED_EVALUATION_FAILED"
    assert result["training_started"] is False
    assert result["records_approved_for_training"] is False
    assert result["tokenizer_training_authorized"] is False
    assert result["pretraining_authorized"] is False

    assert len(
        service.read_history(tmp_path / "storage")
    ) == 1


def make_client():
    return TestClient(
        create_app(
            lambda token: "test-user" if token == "valid" else None
        )
    )


def test_api_rejects_malformed_json_body():
    response = make_client().post(
        "/v1/ingestion/intake",
        headers={
            "Authorization": "Bearer valid",
            "Content-Type": "application/json",
        },
        content=b'{"filename":',
    )

    assert response.status_code == 400


def test_api_rejects_non_object_json_body():
    response = make_client().post(
        "/v1/ingestion/intake",
        headers={"Authorization": "Bearer valid"},
        json=["not", "an", "object"],
    )

    assert response.status_code == 400


def test_api_rejects_invalid_base64():
    response = make_client().post(
        "/v1/ingestion/intake",
        headers={"Authorization": "Bearer valid"},
        json={
            "filename": "records.txt",
            "content_base64": "%%%not-base64%%%",
            "metadata": METADATA,
        },
    )

    assert response.status_code == 400


def test_api_rejects_invalid_optional_field_types():
    for field, value in (
        ("name", 123),
        ("format", ["txt"]),
    ):
        response = make_client().post(
            "/v1/ingestion/intake",
            headers={"Authorization": "Bearer valid"},
            json={
                "filename": "records.txt",
                "content_base64": base64.b64encode(
                    TEXT.encode()
                ).decode(),
                "metadata": METADATA,
                field: value,
            },
        )

        assert response.status_code == 400


def test_api_rejects_oversized_declared_request():
    response = make_client().post(
        "/v1/ingestion/intake",
        headers={
            "Authorization": "Bearer valid",
            "Content-Length": str(100 * 1024 * 1024),
            "Content-Type": "application/json",
        },
        content=b"{}",
    )

    assert response.status_code == 413


def test_api_rejects_missing_or_invalid_authentication():
    client = make_client()

    assert client.get(
        "/v1/ingestion/history"
    ).status_code == 401

    assert client.get(
        "/v1/ingestion/history",
        headers={"Authorization": "Basic invalid"},
    ).status_code == 401

    assert client.get(
        "/v1/ingestion/history",
        headers={"Authorization": "Bearer invalid"},
    ).status_code == 401


def test_api_history_still_requires_valid_authentication():
    response = make_client().get(
        "/v1/ingestion/history",
        headers={"Authorization": "Bearer valid"},
    )

    assert response.status_code == 200
    assert isinstance(response.json()["items"], list)
