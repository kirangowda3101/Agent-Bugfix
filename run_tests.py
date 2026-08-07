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

result = subprocess.run(
    ["python", "-m", "pytest", "-q"],
    cwd=workspace,
    capture_output=True,
    text=True,
)

print("PASS" if result.returncode == 0 else "FAIL")
print(result.stdout)