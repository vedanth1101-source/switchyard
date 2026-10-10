"""
Health endpoint.

Sprint 0 Task 0.3 returns only {"status": "ok"}. The full contract in
docs/04-data-schema-and-api.md is {status, config_version, db, redis}; add those fields
when model config (Task 1.1) and DB/Redis connectivity (Task 0.5) exist.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
