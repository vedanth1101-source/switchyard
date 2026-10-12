from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_ok():
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_openapi_and_docs_served_under_api_prefix():
    openapi = client.get("/api/openapi.json")
    docs = client.get("/api/docs")

    assert openapi.status_code == 200
    assert "/api/health" in openapi.json()["paths"]
    assert docs.status_code == 200
    assert "text/html" in docs.headers["content-type"]
