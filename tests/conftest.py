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


@pytest.fixture(autouse=True)
def isolate_service_settings(request, monkeypatch):
    """Unit tests never see real DATABASE_URL/REDIS_URL; integration tests keep them."""
    from app.core.settings import get_settings

    if request.node.get_closest_marker("integration") is None:
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("REDIS_URL", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
