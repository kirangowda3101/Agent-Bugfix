import base64
import subprocess

CONTAINER = "swe_test"

content = '''def hello():
    print("it's working")
    return {"key": "value"}
'''

encoded = base64.b64encode(content.encode()).decode()

subprocess.run(
    [
        "docker", "exec", CONTAINER,
        "bash", "-c",
        f"echo {encoded} | base64 -d > /testbed/scratch_test.py",
    ],
    check=True,
)

proc = subprocess.run(
    ["docker", "exec", CONTAINER, "cat", "/testbed/scratch_test.py"],
    capture_output=True,
    text=True,
)

print("--- what came back ---")
print(proc.stdout)
print("--- match?", proc.stdout == content, "---")