import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sandbox import make_sandbox  # noqa: E402

BACKEND = os.environ.get("SANDBOX_BACKEND", "docker")


def _unavailable():
    if BACKEND == "docker":
        if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
            return "Docker is not running"
    elif not shutil.which("apptainer"):
        return "apptainer is not on PATH"
    return None


@pytest.fixture(scope="session")
def sandbox():
    reason = _unavailable()
    if reason:
        # On the cluster SANDBOX_REQUIRED=1, so a missing runtime fails the job
        # instead of silently skipping every sandbox test.
        if os.environ.get("SANDBOX_REQUIRED") == "1":
            pytest.fail(f"{reason}; SANDBOX_REQUIRED=1")
        pytest.skip(f"{reason}; sandbox tests need it")
    with make_sandbox(timeout=3.0, workers=4) as sb:
        yield sb
