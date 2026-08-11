import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic

problem = sys.argv[1]

workspace = Path("workspace")
source = Path("dataset") / problem

buggy_code = (source / "buggy" / "solution.py").read_text()


def run_pytest():
    try:
        proc = subprocess.run(
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
        return proc.returncode, f"exit code: {proc.returncode}\n" + proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT: tests took longer than 30 seconds"


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

fixed_code = response.content[0].text
print(fixed_code)

shutil.rmtree(workspace, ignore_errors=True)
shutil.copytree(source / "buggy", workspace)
workspace.joinpath("solution.py").write_text(fixed_code)
shutil.copy(source / "tests" / "test_solution.py", workspace)

returncode, final_output = run_pytest()

result = {
    "version": "v0",
    "problem": problem,
    "verdict": "PASS" if returncode == 0 else "FAIL",
    "steps": 1,
    "input_tokens": response.usage.input_tokens,
    "output_tokens": response.usage.output_tokens,
}

Path("results").mkdir(exist_ok=True)
(Path("results") / f"v0_{problem}.json").write_text(json.dumps(result, indent=2))

print()
print("=" * 50)
for key, value in result.items():
    print(f"{key}: {value}")
print()
print(final_output)