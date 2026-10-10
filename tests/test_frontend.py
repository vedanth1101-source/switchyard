"""Frontend integration through FastAPI's app.frontend(), using a temporary build directory."""

import pytest
from fastapi.testclient import TestClient

import app.main
from app.main import create_app
from app.routes import health

HTML = {"accept": "text/html"}
SPA_MARKER = '<div id="root">switchyard-test-shell</div>'
HEALTHY = {"status": "ok", "db": "ok", "redis": "ok"}


def with_healthy_services(test_app):
    # These tests cover frontend routing; DB/Redis behavior is covered in test_health.py.
    test_app.dependency_overrides[health.database_status] = lambda: "ok"
    test_app.dependency_overrides[health.redis_status] = lambda: "ok"
    return test_app


@pytest.fixture
def fake_dist(tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(f"<!doctype html><html><body>{SPA_MARKER}</body></html>", encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.info('bundle');", encoding="utf-8")
    return dist


@pytest.fixture
def client(fake_dist):
    return TestClient(with_healthy_services(create_app(frontend_dir=fake_dist)))


def test_default_frontend_dir_is_vite_build_output():
    assert app.main.FRONTEND_DIR == "web/dist"


def test_root_serves_frontend_index(client):
    response = client.get("/", headers=HTML)

    assert response.status_code == 200
    assert SPA_MARKER in response.text


def test_spa_deep_link_navigation_falls_back_to_index(client):
    response = client.get("/dashboard/settings", headers=HTML)

    assert response.status_code == 200
    assert SPA_MARKER in response.text


def test_built_asset_is_served(client):
    response = client.get("/assets/app.js")

    assert response.status_code == 200
    assert "bundle" in response.text


def test_missing_asset_is_404_not_spa_shell(client):
    response = client.get("/assets/missing.js", headers={"accept": "*/*"})

    assert response.status_code == 404
    assert SPA_MARKER not in response.text


def test_api_health_is_not_swallowed_by_frontend(client):
    response = client.get("/api/health", headers=HTML)

    assert response.status_code == 200
    assert response.json() == HEALTHY


def test_api_docs_and_openapi_are_not_swallowed_by_frontend(client):
    docs = client.get("/api/docs", headers=HTML)
    openapi = client.get("/api/openapi.json")

    assert docs.status_code == 200
    assert "swagger" in docs.text.lower()
    assert SPA_MARKER not in docs.text
    assert openapi.status_code == 200
    assert "/api/health" in openapi.json()["paths"]


@pytest.mark.parametrize("accept", ["application/json", "text/html", "*/*"])
@pytest.mark.parametrize("path", ["/api/does-not-exist", "/api/v1/nested/unknown", "/api"])
def test_unknown_api_path_is_json_404_for_any_accept_header(client, path, accept):
    response = client.get(path, headers={"accept": accept})

    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}
    assert SPA_MARKER not in response.text


def test_api_not_found_route_is_hidden_from_openapi(client):
    paths = client.get("/api/openapi.json").json()["paths"]

    assert "/api/{path}" not in paths
    assert "/api" not in paths


def test_wrong_method_on_real_api_route_still_405(client):
    assert client.post("/api/health").status_code == 405
    assert client.head("/api/health").status_code == 405


def test_api_not_found_route_is_registered_after_all_api_routes():
    # Starlette uses the first full match, so an API route added after the catch-all
    # would be shadowed by it.
    # Included routers carry no `path`, so check the tail of the whole route table.
    routes = create_app().router.routes

    assert {getattr(route, "path", None) for route in routes[-2:]} == {"/api", "/api/{path:path}"}


def test_missing_frontend_build_fails_at_startup_outside_development(monkeypatch, tmp_path):
    monkeypatch.delenv("FASTAPI_ENV", raising=False)

    with pytest.raises(RuntimeError, match="does not exist"):
        create_app(frontend_dir=tmp_path / "not-built")


def test_missing_index_html_fails_at_startup_outside_development(monkeypatch, tmp_path):
    monkeypatch.delenv("FASTAPI_ENV", raising=False)
    (tmp_path / "dist").mkdir()

    with pytest.raises(RuntimeError, match="index.html"):
        create_app(frontend_dir=tmp_path / "dist")


def test_missing_frontend_build_is_tolerated_with_warning_in_development(monkeypatch, tmp_path):
    monkeypatch.setenv("FASTAPI_ENV", "development")

    with pytest.warns(UserWarning, match="does not exist"):
        dev_app = create_app(frontend_dir=tmp_path / "not-built")

    assert TestClient(with_healthy_services(dev_app)).get("/api/health").json() == HEALTHY
