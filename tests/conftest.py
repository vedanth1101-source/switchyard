import os
import sys
from pathlib import Path

# Ensure repo root and app packages are discoverable
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# app.main serves web/dist via app.frontend(); in development FastAPI only warns when the
# frontend has not been built, so Python tests run without `npm run build`. Production
# (FASTAPI_ENV unset) still fails at startup on a missing build.
os.environ.setdefault("FASTAPI_ENV", "development")
