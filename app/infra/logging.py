"""
Structured JSON logging (Task S2.10; TRD §7, decision D8).

configure_logging() adds one Switchyard handler to the root logger: one compact JSON object per line on
the current sys.stderr. It is idempotent (create_app() calls it, and tests build many apps) and leaves
other handlers alone. `app.*` loggers emit from INFO; httpx and httpcore are held at WARNING because
their INFO request lines carry full URLs. Other libraries keep the root's level (WARNING by default).

Each line holds timestamp, level, logger and message, plus only the ALLOWED_FIELDS passed with
`extra=`; any other attribute (a prompt, a key, a URL) is never written. A record's exception is
reduced to its type name: no message, no traceback. request_id comes from the record, or else from
REQUEST_ID, which RequestIdMiddleware sets for the duration of each /api request.

Messages are written as given: callers keep them free of prompts, URLs, keys and exception text. The one
exception is a record from outside `app.*` that carries an exception: libraries can embed the exception
text in the message, so it is replaced by THIRD_PARTY_EXCEPTION_MESSAGE (logger, level and type remain).
"""

from __future__ import annotations

import json
import logging
import math
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Optional, TextIO

# Set by app.main.RequestIdMiddleware for the duration of each /api request.
REQUEST_ID: ContextVar[Optional[str]] = ContextVar("switchyard_request_id", default=None)

# The `extra=` keys the application's log calls use. Anything else is dropped.
ALLOWED_FIELDS = (
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
)
APP_LOGGER = "app"
URL_LOGGING_LIBRARIES = ("httpx", "httpcore")
THIRD_PARTY_EXCEPTION_MESSAGE = "Third-party exception log suppressed"


class JsonFormatter(logging.Formatter):
    """One JSON object per record: owned fields plus allowlisted extras. Reads the record, never changes it."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": _message(record),
        }
        attributes = vars(record)
        for key in ALLOWED_FIELDS:
            if key in attributes:
                entry[key] = _json_safe(attributes[key])
        if "request_id" not in entry and REQUEST_ID.get() is not None:
            entry["request_id"] = REQUEST_ID.get()
        if record.exc_info and record.exc_info[0] is not None:
            entry["exc_type"] = record.exc_info[0].__name__  # Never the message or traceback
        return json.dumps(entry, ensure_ascii=True, separators=(",", ":"))


def _message(record: logging.LogRecord) -> str:
    # Libraries may put exception text in the message itself (asyncio writes repr(task), which includes
    # the exception), so a third-party record that carries an exception keeps only a fixed message.
    third_party = record.name != APP_LOGGER and not record.name.startswith(APP_LOGGER + ".")
    if third_party and record.exc_info and record.exc_info[0] is not None:
        return THIRD_PARTY_EXCEPTION_MESSAGE
    return record.getMessage()


def _json_safe(value: Any) -> Any:
    # Plain JSON scalars only. Anything else is named by type, never repr()/str(), which could reveal secrets.
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return f"<{type(value).__name__}>"


class SwitchyardJsonHandler(logging.StreamHandler):
    """Writes to whatever sys.stderr is at emit time (pytest and hosts may replace it)."""

    def __init__(self) -> None:
        super().__init__()
        self.setFormatter(JsonFormatter())

    @property
    def stream(self) -> TextIO:  # type: ignore[override]
        return sys.stderr

    @stream.setter
    def stream(self, value: TextIO) -> None:
        pass  # Always the current sys.stderr


def configure_logging() -> None:
    root = logging.getLogger()
    if not any(isinstance(handler, SwitchyardJsonHandler) for handler in root.handlers):
        root.addHandler(SwitchyardJsonHandler())
    logging.getLogger(APP_LOGGER).setLevel(logging.INFO)
    for name in URL_LOGGING_LIBRARIES:
        logging.getLogger(name).setLevel(logging.WARNING)
