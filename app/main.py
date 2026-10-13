"""FastAPI application.

API endpoints, the OpenAPI schema, and docs are served under /api. The built Vite frontend
(web/dist) is served at / as low-priority routes, so API routes always match first.

Every /api response carries an X-Request-ID header (Task S2.1). RequestIdMiddleware creates one uuid4
per request (an inbound X-Request-ID is ignored), stores it in request.state.request_id for handlers,
and catches unhandled exceptions itself: Starlette's ServerErrorMiddleware re-raises them, and the
server would then log the full traceback and message. Logs carry the exception type and frame
locations only. 422 bodies drop FastAPI's `input` and `ctx`, which can echo the submitted prompt.

create_app() configures JSON-lines logging (app.infra.logging, S2.10). The middleware also sets
app.infra.logging.REQUEST_ID while the request runs, so every log line it causes carries the request ID.
"""

import logging
import traceback
import uuid
from os import PathLike
from pathlib import PurePath

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.infra.logging import REQUEST_ID, configure_logging
from app.routes import health
from app.routes import route as route_api
from app.routes.route import REQUEST_ID_HEADER, InternalErrorResponse, ValidationErrorItem, ValidationErrorResponse

logger = logging.getLogger(__name__)

# Relative to the working directory: the project root locally and on Vercel.
FRONTEND_DIR = "web/dist"


def create_app(frontend_dir: str | PathLike[str] = FRONTEND_DIR) -> FastAPI:
    configure_logging()
    application = FastAPI(
        title="Switchyard",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        redoc_url=None,
    )
    application.add_middleware(RequestIdMiddleware)
    application.add_exception_handler(RequestValidationError, _validation_error)
    application.include_router(health.router, prefix="/api")
    application.include_router(route_api.router, prefix="/api")

    # Unknown /api paths must be a JSON 404, not the frontend's index.html navigation fallback.
    # Keep this after every API router: Starlette uses the first full match, so API routes
    # registered later would be shadowed. GET only, so wrong-method requests to real API
    # routes still return 405.
    @application.get("/api", include_in_schema=False)
    @application.get("/api/{path:path}", include_in_schema=False)
    def api_not_found() -> None:
        raise HTTPException(status_code=404)

    # check_dir defaults to "auto": a missing build (or missing index.html) raises at startup
    # unless FASTAPI_ENV=development, where FastAPI only warns.
    application.frontend("/", directory=frontend_dir, fallback="index.html")
    return application


def _is_api_path(path: str) -> bool:
    return path == "/api" or path.startswith("/api/")


class RequestIdMiddleware:
    """Assigns a request ID to every /api request and turns unhandled exceptions into a generic 500."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not _is_api_path(scope["path"]):
            await self.app(scope, receive, send)
            return
        request_id = str(uuid.uuid4())
        scope.setdefault("state", {})["request_id"] = request_id
        token = REQUEST_ID.set(request_id)  # Log lines caused by this request carry its ID
        try:
            await self._handle(scope, receive, send, request_id)
        finally:
            REQUEST_ID.reset(token)

    async def _handle(self, scope: Scope, receive: Receive, send: Send, request_id: str) -> None:
        response_started = False

        async def send_with_request_id(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        except Exception as exc:
            _log_unhandled(request_id, exc)
            if response_started:
                return  # Too late for an error response; the server ends the incomplete one.
        else:
            return
        # Sent only once the except block has exited, so the original exception is no longer being
        # handled: a failure while sending cannot be chained to it and carry its message to the server.
        body = InternalErrorResponse(request_id=request_id, error="internal_error").model_dump()
        try:
            await JSONResponse(status_code=500, content=body)(scope, receive, send_with_request_id)
        except Exception as exc:  # e.g. the client disconnected; nothing more can be sent
            _log_unhandled(request_id, exc, event="error response failed")


def _log_unhandled(request_id: str, exc: Exception, event: str = "unhandled error") -> None:
    # Type and file:line:function per frame, outermost first. Never the message or source lines:
    # either can contain prompts, URLs or keys.
    error_type = type(exc).__name__
    stack = traceback.StackSummary.extract(traceback.walk_tb(exc.__traceback__), lookup_lines=False)
    frames = " > ".join(f"{PurePath(frame.filename).name}:{frame.lineno}:{frame.name}" for frame in stack)
    logger.error(
        event + " request_id=%s error_type=%s frames=%s",
        request_id,
        error_type,
        frames,
        extra={"request_id": request_id, "error_type": error_type, "frames": frames},
    )


async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    # loc, msg and type only: `input` and `ctx` can echo the submitted prompt or the parser's view of it.
    detail = [ValidationErrorItem(loc=list(err["loc"]), msg=err["msg"], type=err["type"]) for err in exc.errors()]
    body = ValidationErrorResponse(request_id=request.state.request_id, error="invalid_request", detail=detail)
    return JSONResponse(status_code=422, content=body.model_dump())


app = create_app()
