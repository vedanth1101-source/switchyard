"""Structured JSON logging (S2.10): format, allowlist, request-ID propagation and leak checks.

Output is read from the Switchyard root handler, which writes to the current sys.stderr (captured by
capsys). Everything is offline: fake routing dependencies and httpx mock transports only.
"""

import asyncio
import gc
import json
import logging
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import Request
from fastapi.testclient import TestClient

from app.infra.budget import create_strong_budget
from app.infra.logging import (
    ALLOWED_FIELDS,
    REQUEST_ID,
    THIRD_PARTY_EXCEPTION_MESSAGE,
    JsonFormatter,
    SwitchyardJsonHandler,
    configure_logging,
)
from app.main import RequestIdMiddleware, create_app
from app.routes import route as route_api
from tests.test_health import client_with_urls
from tests.test_route_api import CHEAP, EASY_JEV, REAL_LIMITER, ApiHarness, ok

FORMATTER_FIELDS = {"timestamp", "level", "logger", "message", "exc_type"}
PROMPT_CANARY = "CANARY-S210-PROMPT-must-never-be-logged"
URL_CANARY = "canary-s210-url-host"
FIELD_CANARY = "CANARY-S210-unknown-field"
EXC_CANARY = "CANARY-S210-exception-message"


def json_lines(err: str) -> list[dict]:
    """Every line the Switchyard handler wrote; each must be exactly one JSON object."""
    records = []
    for line in err.splitlines():
        if line.startswith("{"):
            record = json.loads(line)
            assert isinstance(record, dict)
            records.append(record)
    return records


def make_record(msg="hello %s", args=("world",), level=logging.WARNING, exc_info=None, **extra) -> logging.LogRecord:
    record = logging.LogRecord("app.test", level, __file__, 1, msg, args, exc_info)
    for key, value in extra.items():
        setattr(record, key, value)  # As logging does for `extra=`
    return record


# --- Formatter --------------------------------------------------------------------------------


def test_formatter_emits_one_compact_json_object_with_the_owned_fields():
    output = JsonFormatter().format(make_record(status="success", attempts=2, latency_ms=15))

    assert "\n" not in output
    entry = json.loads(output)
    assert entry["level"] == "WARNING"
    assert entry["logger"] == "app.test"
    assert entry["message"] == "hello world"
    assert entry["timestamp"].endswith("+00:00")
    assert (entry["status"], entry["attempts"], entry["latency_ms"]) == ("success", 2, 15)


def test_formatter_keeps_allowlisted_fields_and_drops_everything_else():
    record = make_record(
        request_id="req-1",
        error_type="ValueError",
        prompt=PROMPT_CANARY,
        api_key=FIELD_CANARY,
        url=f"https://{URL_CANARY}/x",
        canary_field=FIELD_CANARY,
    )

    output = JsonFormatter().format(record)

    entry = json.loads(output)
    assert set(entry) <= FORMATTER_FIELDS | set(ALLOWED_FIELDS)
    assert (entry["request_id"], entry["error_type"]) == ("req-1", "ValueError")
    for leaked in (PROMPT_CANARY, FIELD_CANARY, URL_CANARY, "prompt", "api_key", "canary_field", "pathname", "args"):
        assert leaked not in output


def test_allowlist_is_exactly_the_existing_structured_fields():
    assert set(ALLOWED_FIELDS) == {
        "request_id",
        "status",
        "tier",
        "reason",
        "task_type",
        "final_model",
        "attempts",
        "latency_ms",
        "failure_reason",
        "jev_error",
        "jev_error_cause",
        "error",
        "error_type",
        "component",
        "operation",
        "frames",
    }


def test_non_json_values_are_rendered_by_type_name_only():
    class Secretive:
        def __repr__(self):
            return FIELD_CANARY

        __str__ = __repr__

    entry = json.loads(JsonFormatter().format(make_record(status=Secretive(), latency_ms=float("nan"))))

    assert entry["status"] == "<Secretive>"
    assert entry["latency_ms"] is None
    assert FIELD_CANARY not in json.dumps(entry)


def test_exceptions_render_the_type_only():
    try:
        raise RuntimeError(f"boom {EXC_CANARY}")
    except RuntimeError as exc:
        record = make_record(msg="failed", args=(), level=logging.ERROR, exc_info=(type(exc), exc, exc.__traceback__))

    output = JsonFormatter().format(record)

    assert json.loads(output)["exc_type"] == "RuntimeError"
    for leaked in (EXC_CANARY, "Traceback", "boom", __file__):
        assert leaked not in output
    assert record.exc_text is None  # Never formatted (and so never cached for other handlers)


def test_formatting_does_not_change_record_attributes():
    record = make_record(request_id="req-1", reason="redis_not_configured", prompt="kept on the record")
    before = dict(vars(record))

    JsonFormatter().format(record)

    assert vars(record) == before  # caplog tests keep reading the same attributes


# --- Configuration ----------------------------------------------------------------------------


def switchyard_handlers() -> list[logging.Handler]:
    return [h for h in logging.getLogger().handlers if isinstance(h, SwitchyardJsonHandler)]


def test_configuration_is_idempotent_and_keeps_other_handlers():
    root = logging.getLogger()
    others = [h for h in root.handlers if not isinstance(h, SwitchyardJsonHandler)]
    for handler in switchyard_handlers():  # Only ours: create_app() alone must install it
        root.removeHandler(handler)

    create_app()
    assert len(switchyard_handlers()) == 1
    configure_logging()
    configure_logging()
    create_app()

    assert len(switchyard_handlers()) == 1
    assert [h for h in root.handlers if not isinstance(h, SwitchyardJsonHandler)] == others


def test_app_info_logs_are_enabled_and_url_bearing_http_client_info_is_not():
    configure_logging()

    assert logging.getLogger("app.core.router").isEnabledFor(logging.INFO)
    for name in ("httpx", "httpcore"):
        assert not logging.getLogger(name).isEnabledFor(logging.INFO)
        assert logging.getLogger(name).isEnabledFor(logging.WARNING)


def test_httpx_request_urls_never_reach_the_output(capsys):
    configure_logging()
    transport = httpx.MockTransport(lambda request: httpx.Response(200))

    with httpx.Client(transport=transport) as client:
        client.get(f"https://{URL_CANARY}.example/secret-path?token={FIELD_CANARY}")
    logging.getLogger("app.test").warning("after the http call")

    err = capsys.readouterr().err
    assert [entry["message"] for entry in json_lines(err)] == ["after the http call"]
    assert URL_CANARY not in err
    assert FIELD_CANARY not in err


def test_logs_outside_a_request_have_no_request_id(capsys):
    configure_logging()

    logging.getLogger("app.test").warning("outside any request")

    (entry,) = json_lines(capsys.readouterr().err)
    assert "request_id" not in entry


def test_logged_exceptions_never_print_their_message_or_traceback(capsys):
    configure_logging()

    try:
        raise RuntimeError(f"boom {EXC_CANARY}")
    except RuntimeError:
        logging.getLogger("app.test").exception("handled failure")

    err = capsys.readouterr().err
    (entry,) = json_lines(err)
    assert (entry["message"], entry["exc_type"]) == ("handled failure", "RuntimeError")
    assert EXC_CANARY not in err
    assert "Traceback" not in err


# --- Request-ID propagation -------------------------------------------------------------------


def test_route_budget_and_rate_limit_logs_carry_the_response_request_id(capsys):
    harness = ApiHarness(EASY_JEV, [ok(CHEAP[0])], limiter=REAL_LIMITER)  # No Redis: limiter logs a warning

    @asynccontextmanager
    async def factory_with_real_budget(settings, *, dry_run):
        create_strong_budget(settings)  # No REDIS_URL in unit tests: logs "strong budget unavailable"
        yield harness.deps

    harness.client.app.dependency_overrides[route_api.get_deps_factory] = lambda: factory_with_real_budget

    response = harness.post({"prompt": PROMPT_CANARY})

    assert response.status_code == 200
    request_id = response.headers["X-Request-ID"]
    assert response.json()["request_id"] == request_id
    entries = json_lines(capsys.readouterr().err)
    by_logger = {entry["logger"]: entry for entry in entries}
    for name in ("app.infra.ratelimit", "app.infra.budget", "app.core.router", "app.routes.route"):
        assert by_logger[name]["request_id"] == request_id, name
    assert by_logger["app.infra.budget"]["reason"] == "redis_not_configured"
    assert by_logger["app.core.router"]["status"] == "success"


def test_prompt_never_reaches_the_output(capsys):
    harness = ApiHarness(EASY_JEV, [ok(CHEAP[0])], limiter=REAL_LIMITER)

    assert harness.post({"prompt": PROMPT_CANARY}).status_code == 200

    err = capsys.readouterr().err
    assert json_lines(err)  # Application logs were emitted
    assert PROMPT_CANARY not in err


def test_health_logs_from_the_worker_thread_carry_the_request_id(capsys):
    response = client_with_urls("notadialect://db.example/x", None).get("/api/health")

    assert response.status_code == 503
    (entry,) = [e for e in json_lines(capsys.readouterr().err) if e["logger"] == "app.routes.health"]
    assert entry["component"] == "db"
    assert entry["request_id"] == response.headers["X-Request-ID"]


def test_unhandled_error_log_keeps_its_own_request_id(capsys):
    def broken():
        raise RuntimeError(f"bug {EXC_CANARY}")

    app = create_app()
    app.dependency_overrides[route_api.get_deps_factory] = broken

    response = TestClient(app).post("/api/route", json={"prompt": "hi"})

    assert response.status_code == 500
    err = capsys.readouterr().err
    (entry,) = [e for e in json_lines(err) if e["logger"] == "app.main"]
    assert entry["request_id"] == response.json()["request_id"] == response.headers["X-Request-ID"]
    assert entry["error_type"] == "RuntimeError"
    assert EXC_CANARY not in err


def test_concurrent_requests_never_see_each_others_request_id(capsys):
    app = create_app()
    probe_logger = logging.getLogger("app.test_logging_probe")
    first_logged, second_logged = asyncio.Event(), asyncio.Event()

    @app.post("/api/probe/{name}")
    async def probe(name: str, request: Request) -> dict:
        probe_logger.warning("probe start name=%s", name)
        if name == "first":
            first_logged.set()
            await second_logged.wait()  # Both requests are now in flight together
        else:
            await first_logged.wait()
            second_logged.set()
        probe_logger.warning("probe end name=%s", name)
        return {}

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await asyncio.gather(client.post("/api/probe/first"), client.post("/api/probe/second"))

    first, second = asyncio.run(run())

    ids = {"first": first.headers["X-Request-ID"], "second": second.headers["X-Request-ID"]}
    assert ids["first"] != ids["second"]
    probes = [e for e in json_lines(capsys.readouterr().err) if e["logger"] == "app.test_logging_probe"]
    assert len(probes) == 4
    for entry in probes:
        name = entry["message"].rsplit("=", 1)[1]
        assert entry["request_id"] == ids[name]
        uuid.UUID(entry["request_id"])


def test_request_id_context_is_reset_after_success_and_after_an_unhandled_error():
    seen = []

    async def inner_ok(scope, receive, send):
        seen.append(REQUEST_ID.get())
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def inner_raising(scope, receive, send):
        seen.append(REQUEST_ID.get())
        raise RuntimeError(EXC_CANARY)

    async def receive():
        return {"type": "http.request", "body": b""}

    async def send(message):
        pass

    async def run(inner):
        await RequestIdMiddleware(inner)({"type": "http", "path": "/api/x", "headers": []}, receive, send)
        return REQUEST_ID.get()

    after = [asyncio.run(run(inner_ok)), asyncio.run(run(inner_raising))]

    assert after == [None, None]  # Reset in this same context, even when the inner app raised
    assert all(seen) and seen[0] != seen[1]


# --- Third-party records with exceptions (review finding I1) ---------------------------------

ASYNCIO_CANARY = "CANARY-S210-asyncio-task-exception"


def test_unretrieved_asyncio_task_exception_message_never_reaches_the_output(capsys):
    # asyncio's "Task exception was never retrieved" message embeds repr(task), which holds the exception text.
    configure_logging()

    async def boom():
        raise RuntimeError(f"secret detail {ASYNCIO_CANARY}")

    async def main():
        asyncio.get_running_loop().create_task(boom())  # Never awaited, never retrieved
        await asyncio.sleep(0)

    asyncio.run(main())
    gc.collect()  # The task is reported when it is garbage-collected

    err = capsys.readouterr().err
    (entry,) = [e for e in json_lines(err) if e["logger"] == "asyncio"]
    assert (entry["level"], entry["exc_type"]) == ("ERROR", "RuntimeError")
    assert entry["message"] == THIRD_PARTY_EXCEPTION_MESSAGE
    for leaked in (ASYNCIO_CANARY, "secret detail", "Task exception was never retrieved", "future:", "Traceback"):
        assert leaked not in err


def test_third_party_exception_record_message_is_replaced_by_the_formatter():
    try:
        raise ValueError(ASYNCIO_CANARY)
    except ValueError as exc:
        record = logging.LogRecord(
            "somelib.worker", logging.ERROR, __file__, 1, "failed: %r", (exc,), (type(exc), exc, exc.__traceback__)
        )

    entry = json.loads(JsonFormatter().format(record))

    assert entry == {
        "timestamp": entry["timestamp"],
        "level": "ERROR",
        "logger": "somelib.worker",
        "message": THIRD_PARTY_EXCEPTION_MESSAGE,
        "exc_type": "ValueError",
    }


def test_app_exception_records_keep_their_message(capsys):
    configure_logging()

    try:
        raise RuntimeError(f"boom {EXC_CANARY}")
    except RuntimeError:
        logging.getLogger("app.core.router").exception("route failed request_id=%s", "req-9")

    (entry,) = json_lines(capsys.readouterr().err)
    assert (entry["logger"], entry["message"], entry["exc_type"]) == (
        "app.core.router",
        "route failed request_id=req-9",
        "RuntimeError",
    )


def test_third_party_records_without_an_exception_keep_their_message(capsys):
    configure_logging()

    logging.getLogger("somelib.pool").warning("pool size reached limit=%d", 5)

    (entry,) = json_lines(capsys.readouterr().err)
    assert (entry["logger"], entry["level"], entry["message"]) == (
        "somelib.pool",
        "WARNING",
        "pool size reached limit=5",
    )
    assert "exc_type" not in entry
