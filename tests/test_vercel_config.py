"""Local consistency checks for the Vercel deployment configuration (Task 0.4).

These tests cannot prove Vercel's runtime behavior; they only check that the files
agree with each other and with the repository layout.
"""

import fnmatch
import json
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

ROOT = Path(__file__).resolve().parents[1]


def load_vercel_config() -> dict:
    return json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))


def expand_braces(pattern: str) -> list[str]:
    if pattern.startswith("{") and pattern.endswith("}"):
        return pattern[1:-1].split(",")
    return [pattern]


def test_vercel_json_uses_fastapi_preset_detection():
    config = load_vercel_config()

    assert config["$schema"] == "https://openapi.vercel.sh/vercel.json"
    # The FastAPI preset is detected from app/main.py; nothing may override or bypass it.
    for removed_key in ("framework", "outputDirectory", "rewrites", "routes"):
        assert removed_key not in config, removed_key


def test_removed_api_directory_design_is_gone():
    assert not (ROOT / "api").exists()
    for config_file in ("vercel.json", "pyproject.toml"):
        assert "api/index" not in (ROOT / config_file).read_text(encoding="utf-8"), config_file


def test_build_command_produces_the_directory_fastapi_serves():
    import app.main

    config = load_vercel_config()

    assert config["buildCommand"] == "cd web && npm ci && npm run build"
    assert (ROOT / "web" / "package-lock.json").is_file()  # required by `npm ci`
    assert "outDir" not in (ROOT / "web" / "vite.config.ts").read_text(encoding="utf-8")  # Vite default: dist
    assert app.main.FRONTEND_DIR == "web/dist"


def test_function_config_targets_fastapi_entrypoint():
    functions = load_vercel_config()["functions"]

    assert list(functions) == ["app/main.py"]
    assert (ROOT / "app" / "main.py").is_file()
    assert functions["app/main.py"]["maxDuration"] == 60


def test_exclude_files_fits_vercel_length_limit():
    # The Vercel API rejects the deployment (400) when excludeFiles is longer than 256 characters.
    exclude = load_vercel_config()["functions"]["app/main.py"]["excludeFiles"]

    assert len(exclude) <= 256, len(exclude)


def test_exclude_files_keep_runtime_code_and_frontend_build():
    exclude = load_vercel_config()["functions"]["app/main.py"]["excludeFiles"]
    patterns = expand_braces(exclude)
    kept = [
        "app/main.py",
        "app/routes/health.py",
        "app/core/policy.py",
        "app/core/jev_client.py",
        "app/core/classifier.py",
        "requirements.txt",
        "pyproject.toml",
        # app.frontend() checks web/dist at startup, so the build must stay in the bundle.
        "web/dist/index.html",
        "web/dist/assets/index-abc123.js",
    ]
    excluded = [
        "tests/test_health.py",
        "web/src/App.tsx",
        "web/node_modules/vite/package.json",
        "web/package.json",
        "web/index.html",
        "docs/03-project-flow.md",
    ]

    # fnmatch's "*" also crosses "/", so it over-matches compared with a real glob:
    # no match here means no match on Vercel either.
    for path in kept:
        assert not any(fnmatch.fnmatch(path, p) for p in patterns), path
    for path in excluded:
        assert any(fnmatch.fnmatch(path, p) for p in patterns), path


def test_pyproject_does_not_exclude_frontend_from_function_bundle():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    static = pyproject.get("tool", {}).get("vercel", {}).get("fastapi", {}).get("static", {})

    # exclude = true would remove web/dist from the function, and app.frontend() would then
    # fail its startup directory check on Vercel.
    assert static.get("exclude", False) is False


def test_pyproject_and_requirements_declare_identical_runtime_dependencies():
    # Vercel accepts either file and does not document precedence, so both must agree.
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requirement_lines = [
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    assert sorted(pyproject["project"]["dependencies"]) == sorted(requirement_lines)
    assert all(str(Requirement(line).specifier).startswith("==") for line in requirement_lines)


def test_python_version_pin_is_supported_and_satisfies_requires_python():
    pinned = (ROOT / ".python-version").read_text(encoding="utf-8").strip()
    requires_python = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["requires-python"]

    assert pinned == "3.12"  # Vercel supports 3.12 (default), 3.13, 3.14
    assert SpecifierSet(requires_python).contains(pinned)


def test_web_package_pins_node_major():
    package = json.loads((ROOT / "web" / "package.json").read_text(encoding="utf-8"))

    assert package["engines"]["node"] == "24.x"
