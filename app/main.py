"""FastAPI application.

API endpoints, the OpenAPI schema, and docs are served under /api. The built Vite frontend
(web/dist) is served at / as low-priority routes, so API routes always match first.
"""

from os import PathLike

from fastapi import FastAPI, HTTPException

from app.routes import health
from app.routes import route as route_api

# Relative to the working directory: the project root locally and on Vercel.
FRONTEND_DIR = "web/dist"


def create_app(frontend_dir: str | PathLike[str] = FRONTEND_DIR) -> FastAPI:
    application = FastAPI(
        title="Switchyard",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        redoc_url=None,
    )
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


app = create_app()
