import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic
import shutil
import subprocess

problem = sys.argv[1]

buggy_code = (Path("dataset") / problem / "buggy" / "solution.py").read_text()

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=1000,
    system="You fix bugs in Python code. Return only the corrected code. No explanation, no markdown fences, no commentary.",
    messages=[
        {"role": "user", "content": f"Fix the bug in this code:\n\n{buggy_code}"}
    ],
)

print(response.content[0].text)

fixed_code = response.content[0].text

workspace = Path("workspace")
shutil.rmtree(workspace, ignore_errors=True)

shutil.copytree(Path("dataset") / problem / "buggy", workspace)
workspace.joinpath("solution.py").write_text(fixed_code)
shutil.copy(Path("dataset") / problem / "tests" / "test_solution.py", workspace)

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

print("PASS" if result.returncode == 0 else "FAIL")
print(result.stdout + result.stderr)