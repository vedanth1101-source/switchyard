"""Offline checks for the live smoke test's credential guard. Nothing external is contacted."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_live_route import LIVE_KEYS, require_live_keys

FAKE = {name: f"fake-{name.lower()}-value" for name in LIVE_KEYS}


def test_accepts_when_every_live_key_is_set():
    assert require_live_keys(FAKE) is None


@pytest.mark.parametrize("missing", LIVE_KEYS)
@pytest.mark.parametrize("value", [None, "", "   "], ids=["absent", "empty", "blank"])
def test_fails_naming_the_missing_key(missing, value):
    environ = dict(FAKE)
    if value is None:
        del environ[missing]
    else:
        environ[missing] = value

    with pytest.raises(pytest.fail.Exception, match=missing):
        require_live_keys(environ)


def test_failure_message_does_not_echo_values():
    environ = {LIVE_KEYS[0]: "present-secret-value-should-not-appear"}

    with pytest.raises(pytest.fail.Exception) as excinfo:
        require_live_keys(environ)

    assert "present-secret-value-should-not-appear" not in str(excinfo.value)


def test_guard_failure_output_never_prints_the_environment():
    # pytest's default traceback prints frame arguments; the guard receives os.environ, so its frame
    # must be hidden or a set key (or any other variable) could appear in the failure output.
    root = Path(__file__).resolve().parents[1]
    env = {name: value for name, value in os.environ.items() if name not in LIVE_KEYS}
    env["AI_GATEWAY_API_KEY"] = "fake-gateway-value-must-not-print"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-m",
            "live",
            "-p",
            "no:cacheprovider",
            "--no-cov",
            "tests/test_live_route.py::test_live_dry_run_classifies_with_jev",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 1
    assert "export JEVSTATION_API_KEY" in output
    assert "fake-gateway-value-must-not-print" not in output
    assert "environ(" not in output
