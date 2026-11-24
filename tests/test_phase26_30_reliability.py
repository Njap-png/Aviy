from fastapi.testclient import TestClient

from aviy.api.app import create_app


def test_health_endpoint_repeated_requests_remain_stable():
    app = create_app(token_validator=lambda token: True)
    with TestClient(app) as client:
        for _ in range(500):
            response = client.get("/health")
            assert response.status_code == 200
