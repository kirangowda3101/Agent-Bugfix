import base64
import subprocess
from mcp.server import MCPServer

mcp = MCPServer("bugfix-tools")

CONTAINER = None


def set_container(name):
    """Called by the agent script before the loop starts."""
    global CONTAINER
    CONTAINER = name


def run_in_container(cmd):
    proc = subprocess.run(
        ["docker", "exec", "-w", "/testbed", CONTAINER, "bash", "-c", cmd],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return proc.stdout + proc.stderr


@mcp.tool()
def search_code(pattern: str) -> str:
    """Search the repository for a pattern using grep. Returns matching file:line:content results."""
    output = run_in_container(f"grep -rn --include=*.py '{pattern}' .")
    lines = output.splitlines()
    if len(lines) > 30:
        return "\n".join(lines[:30]) + f"\n... ({len(lines) - 30} more matches)"
    return output


@mcp.tool()
def read_file(filename: str) -> str:
    """Read the contents of a file in the repository."""
    output = run_in_container(f"cat {filename}")
    lines = output.splitlines()
    if len(lines) > 400:
        return "\n".join(lines[:400]) + f"\n... (truncated, {len(lines) - 400} more lines)"
    return output


@mcp.tool()
def write_file(filename: str, content: str) -> str:
    """Write content to a file in the repository. Overwrites the file completely."""
    path = filename.lstrip("./")
    if path.startswith("testbed/"):
        path = path[len("testbed/"):]
    encoded = base64.b64encode(content.encode()).decode()
    result = run_in_container(f"echo {encoded} | base64 -d > /testbed/{path}")
    if result.strip():
        return f"Error writing {path}: {result}"
    return f"Wrote {len(content)} characters to {path}"


@mcp.tool()
def run_tests(test_module: str) -> str:
    """Run a Django test module and return the results. Use this to verify whether your fix actually works before finishing."""
    output = run_in_container(
        "source /opt/miniconda3/bin/activate && conda activate testbed && "
        f"./tests/runtests.py --settings=test_sqlite {test_module}"
    )
    lines = output.splitlines()
    return "\n".join(lines[-50:])


if __name__ == "__main__":
    mcp.run()