import os
import sys
from pathlib import Path

import pytest

# Ensure repo root and app packages are discoverable
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# app.main serves web/dist via app.frontend(); in development FastAPI only warns when the
# frontend has not been built, so Python tests run without `npm run build`. Production
# (FASTAPI_ENV unset) still fails at startup on a missing build.
os.environ.setdefault("FASTAPI_ENV", "development")

# Removed for every unit test: a developer shell may export them (e.g. from .env), and unit tests assume
# the defaults (no database or Redis, default rate limit, no salt, socket peer as the client IP, cache off).
UNIT_TEST_UNSET_ENV = (
    "DATABASE_URL",
    "REDIS_URL",
    "RATE_LIMIT_IP_SALT",
    "CLIENT_IP_HEADER",
    "RATE_LIMIT_PER_MIN",
    "CACHE_NAMESPACE",
    "CACHE_KEY_SECRET",
    "CACHE_TTL_SECONDS",
)


@pytest.fixture(autouse=True)
def isolate_service_settings(request, monkeypatch):
    """Unit tests never see real service or rate-limit settings from the shell; integration tests keep them."""
    from app.core.settings import get_settings

    if request.node.get_closest_marker("integration") is None:
        for name in UNIT_TEST_UNSET_ENV:
            monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
