import base64

from fastapi.testclient import TestClient

from aviy.api.app import create_app


def test_ingestion_api_requires_bearer_authentication():
    client = TestClient(create_app(lambda token: "test-user" if token == "valid" else None))
    response = client.get("/v1/ingestion/history")
    assert response.status_code == 401


def test_ingestion_api_rejects_invalid_token():
    client = TestClient(create_app(lambda token: None))
    response = client.get(
        "/v1/ingestion/history",
        headers={"Authorization": "Bearer invalid"},
    )
    assert response.status_code == 401


def test_ingestion_api_validates_payload_before_intake():
    client = TestClient(create_app(lambda token: "test-user" if token == "valid" else None))
    response = client.post(
        "/v1/ingestion/intake",
        headers={"Authorization": "Bearer valid"},
        json={
            "filename": "test.txt",
            "content_base64": base64.b64encode(b"sample").decode(),
            "metadata": {},
        },
    )
    assert response.status_code == 400
