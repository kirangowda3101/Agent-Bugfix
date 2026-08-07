import shutil
import subprocess
import sys
from pathlib import Path

problem = sys.argv[1]
version = sys.argv[2]

source = Path("dataset") / problem
workspace = Path("workspace")

shutil.rmtree(workspace, ignore_errors=True)
workspace.mkdir()

shutil.copy(source / version / "solution.py", workspace)
shutil.copy(source / "tests" / "test_solution.py", workspace)

try:
    result = subprocess.run(
        [
            "docker", "run", "--rm",
            "-v", f"{workspace.resolve()}:/app",
            "-w", "/app",
            "bugfix-sandbox",
            "timeout", "25", "python", "-m", "pytest", "-q",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    passed = result.returncode == 0
    output = f"exit code: {result.returncode}\n" + result.stdout + result.stderr
except subprocess.TimeoutExpired:
    passed = False
    output = "TIMEOUT: tests took longer than 30 seconds"

print("PASS" if passed else "FAIL")
print(output)