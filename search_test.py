import subprocess
import sys
from pathlib import Path

repo = Path("repos/django")


def search_files(pattern, max_results=30):
    proc = subprocess.run(
        ["grep", "-rn", "--include=*.py", pattern, "."],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=30,
    )
    lines = proc.stdout.splitlines()
    if not lines:
        return f"No matches found for '{pattern}'"
    truncated = lines[:max_results]
    out = "\n".join(truncated)
    if len(lines) > max_results:
        out += f"\n... and {len(lines) - max_results} more matches (showing first {max_results})"
    return out


print(search_files(sys.argv[1]))