"""Request-ID middleware and app-wide error envelopes (Task S2.1). No network access."""

import asyncio
import logging
import sys
import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import RequestIdMiddleware, create_app

SECRET = "SECRET-MARKER-c0ffee"


def assert_uuid4_header(response) -> str:
    request_id = response.headers["X-Request-ID"]
    assert uuid.UUID(request_id).version == 4
    return request_id


@pytest.mark.parametrize(
    ("method", "path", "body", "status"),
    [
        ("GET", "/api/does-not-exist", None, 404),
        ("GET", "/api", None, 404),
        ("POST", "/api/health", None, 405),
        ("POST", "/api/route", {}, 422),
    ],
    ids=["unknown-path-404", "api-root-404", "wrong-method-405", "invalid-body-422"],
)
def test_every_api_error_response_carries_a_uuid4_request_id(method, path, body, status):
    response = TestClient(create_app()).request(method, path, json=body)

    assert response.status_code == status
    assert_uuid4_header(response)


def test_health_response_carries_a_request_id():
    response = TestClient(create_app()).get("/api/health")

    assert_uuid4_header(response)


def test_each_request_gets_a_new_id():
    client = TestClient(create_app())

    ids = {client.get("/api/does-not-exist").headers["X-Request-ID"] for _ in range(3)}

    assert len(ids) == 3


def test_inbound_request_id_header_is_ignored():
    response = TestClient(create_app()).get("/api/does-not-exist", headers={"X-Request-ID": "client-chosen-id"})

    assert response.headers["X-Request-ID"] != "client-chosen-id"
    assert_uuid4_header(response)


def test_frontend_paths_are_not_given_a_request_id(tmp_path):
    (tmp_path / "index.html").write_text("<!doctype html><title>spa</title>", encoding="utf-8")

    response = TestClient(create_app(frontend_dir=tmp_path)).get("/")

    assert response.status_code == 200
    assert "X-Request-ID" not in response.headers


# --- 422 ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"prompt": ""},
        {"prompt": "   \n\t"},
        {"prompt": SECRET + "x" * 8000},
        {"prompt": 42},
        {"prompt": SECRET, "dry_run": "maybe-" + SECRET},
    ],
    ids=["missing", "empty", "blank", "too-long", "not-a-string", "bad-dry-run"],
)
def test_422_body_is_request_id_error_code_and_sanitized_detail(body):
    response = TestClient(create_app()).post("/api/route", json=body)

    assert response.status_code == 422
    payload = response.json()
    assert set(payload) == {"request_id", "error", "detail"}
    assert payload["request_id"] == assert_uuid4_header(response)
    assert payload["error"] == "invalid_request"
    assert payload["detail"]
    for item in payload["detail"]:
        assert set(item) == {"loc", "msg", "type"}
    assert SECRET not in response.text


def test_422_detail_keeps_location_message_and_type():
    response = TestClient(create_app()).post("/api/route", json={"prompt": ""})

    assert response.json()["detail"] == [
        {"loc": ["body", "prompt"], "msg": "String should have at least 1 character", "type": "string_too_short"}
    ]


def test_422_for_malformed_json_does_not_echo_the_body_or_parser_context():
    response = TestClient(create_app()).post(
        "/api/route", content=b'{"prompt": "' + SECRET.encode() + b'"', headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 422
    assert response.json()["error"] == "invalid_request"
    assert [item["type"] for item in response.json()["detail"]] == ["json_invalid"]
    assert SECRET not in response.text


def test_openapi_documents_the_422_envelope():
    spec = create_app().openapi()
    schema_ref = spec["paths"]["/api/route"]["post"]["responses"]["422"]["content"]["application/json"]["schema"]

    assert schema_ref["$ref"].endswith("/ValidationErrorResponse")
    assert set(spec["components"]["schemas"]["ValidationErrorResponse"]["properties"]) == {
        "request_id",
        "error",
        "detail",
    }


# --- Unhandled exceptions (raw ASGI, so the middleware is tested on its own) -------------------


def run_asgi(app, path="/api/x"):
    scope = {"type": "http", "method": "GET", "path": path, "headers": [], "query_string": b""}
    sent: list[dict] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(RequestIdMiddleware(app)(scope, receive, send))
    return scope, sent


def failing_app(*, start_first: bool):
    async def app(scope, receive, send):
        if start_first:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"partial", "more_body": True})
        try:
            raise ValueError(f"inner {SECRET}")
        except ValueError as inner:
            raise RuntimeError(f"outer {SECRET} at C:/secret/path/file.py") from inner

    return app


def header(message: dict, name: bytes) -> str:
    return dict(message["headers"])[name].decode()


def assert_sanitized_error_log(caplog, request_id: str) -> None:
    records = [record for record in caplog.records if record.name == "app.main"]
    assert len(records) == 1
    record = records[0]
    assert record.levelno == logging.ERROR
    assert record.request_id == request_id
    assert record.error_type == "RuntimeError"
    # Outermost first: the middleware, then the failing app function (file:line:function only).
    assert record.frames.startswith("main.py:")
    assert record.frames.endswith(":app")
    assert "test_request_id.py:" in record.frames
    assert record.exc_info is None
    assert SECRET not in caplog.text
    assert "secret/path" not in caplog.text
    assert "Traceback" not in caplog.text
    assert "raise RuntimeError" not in caplog.text  # no source lines


def test_unhandled_error_before_the_response_is_a_generic_500(caplog):
    with caplog.at_level(logging.INFO):
        scope, sent = run_asgi(failing_app(start_first=False))

    start, body = sent
    request_id = scope["state"]["request_id"]
    assert start["status"] == 500
    assert header(start, b"x-request-id") == request_id
    assert header(start, b"content-type") == "application/json"
    assert body["body"] == b'{"request_id":"' + request_id.encode() + b'","error":"internal_error"}'
    assert_sanitized_error_log(caplog, request_id)


def test_unhandled_error_after_the_response_started_is_logged_not_reraised(caplog):
    with caplog.at_level(logging.INFO):
        scope, sent = run_asgi(failing_app(start_first=True))

    # No second response can be sent; the partial one keeps its request ID.
    assert [message["type"] for message in sent] == ["http.response.start", "http.response.body"]
    assert header(sent[0], b"x-request-id") == scope["state"]["request_id"]
    assert_sanitized_error_log(caplog, scope["state"]["request_id"])


def test_cancellation_is_not_caught():
    async def cancelled(scope, receive, send):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        run_asgi(cancelled)


# --- The generic 500 itself cannot be sent (e.g. the client disconnected) --------------------

SEND_SECRET = "SEND-FAILURE-MARKER-beef"


def run_asgi_with_send(app, send):
    scope = {"type": "http", "method": "GET", "path": "/api/x", "headers": [], "query_string": b""}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    asyncio.run(RequestIdMiddleware(app)(scope, receive, send))
    return scope


def test_failure_to_send_the_500_is_logged_sanitized_and_does_not_escape(caplog):
    sends: list[dict] = []
    active_exception_during_send: list[BaseException | None] = []

    async def failing_send(message):
        sends.append(message)
        active_exception_during_send.append(sys.exc_info()[1])
        raise OSError(f"connection reset {SEND_SECRET}")

    with caplog.at_level(logging.INFO):
        scope = run_asgi_with_send(failing_app(start_first=False), failing_send)  # must not raise

    request_id = scope["state"]["request_id"]
    # One attempt at the 500, made after the original exception stopped being handled: an exception
    # raised by send() cannot be implicitly chained to it (and so cannot carry its message).
    assert [(message["type"], message["status"]) for message in sends] == [("http.response.start", 500)]
    assert active_exception_during_send == [None]
    records = [record for record in caplog.records if record.name == "app.main"]
    assert [(record.levelno, record.request_id, record.error_type) for record in records] == [
        (logging.ERROR, request_id, "RuntimeError"),
        (logging.ERROR, request_id, "OSError"),
    ]
    assert records[1].getMessage().startswith("error response failed ")
    assert all(record.exc_info is None for record in records)
    for leaked in (SECRET, SEND_SECRET, "secret/path", "Traceback", "raise "):
        assert leaked not in caplog.text


def test_cancellation_while_sending_the_500_is_not_caught():
    async def cancelled_send(message):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        run_asgi_with_send(failing_app(start_first=False), cancelled_send)


def test_non_api_paths_pass_through_untouched():
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    scope, sent = run_asgi(app, path="/assets/app.js")

    assert "state" not in scope
    assert sent[0]["headers"] == []


def test_non_api_errors_are_not_handled_by_the_middleware():
    with pytest.raises(RuntimeError):
        run_asgi(failing_app(start_first=False), path="/index.html")
