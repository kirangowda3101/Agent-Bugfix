"""Runs inside the sandbox container. Never import this on the host.

Reads one JSON request on stdin:
    {"nonce": str, "timeout": float, "workers": int,
     "jobs": [{"id": str, "code": str, "setup": str, "tests": [str, ...]}]}
Writes one JSON response on stdout:
    {"results": [{"id": str, "passed": [bool, ...], "errors": [str, ...]}]}

Every (job, test) pair runs in its own fresh Python process, so one test cannot
see state left behind by another. Each child installs Docker's default seccomp
allowlist on itself (seccomp_filter.py) before running model code, and runs
under memory, CPU-time, file-size and process-count rlimits. A test only counts as passed if the child
prints the nonce *after* the assert has run; exiting early (sys.exit(0),
os._exit(0)) therefore scores as a failure, not a pass.
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

MEM_BYTES = 512 * 1024 * 1024
OUTPUT_BYTES = 1024 * 1024
NPROC = 64

# After the prelude below, the child closes stdin, then runs setup, model code
# and one assert in a fresh namespace. __name__ is not "__main__", so
# `if __name__ == "__main__":` blocks in model code (often input() loops) are skipped.
CHILD = r"""
os.close(0)
sys.stdin = None
nonce = payload.pop("nonce")
ns = {"__name__": "solution", "__builtins__": __builtins__}
exec(compile(payload["setup"], "<setup>", "exec"), ns)
exec(compile(payload["code"], "<solution>", "exec"), ns)
exec(compile(payload["test"], "<test>", "exec"), ns)
sys.stdout.flush()
os.write(1, ("\n" + nonce + "\n").encode())
"""

# Runs first in the child: read the payload, then install the seccomp filter
# before any model code. If the filter cannot be installed the child exits
# non-zero and the test scores as failed (fail closed).
CHILD_PRELUDE = r"""
import json, os, sys
payload = json.loads(sys.stdin.read())
sys.path.insert(0, {harness_dir!r})
import seccomp_filter
seccomp_filter.install({profile!r})
sys.path.pop(0)
"""
HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILE = os.path.join(HARNESS_DIR, "seccomp-default.json")


def _limits(cpu_seconds):
    def apply():
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (MEM_BYTES, MEM_BYTES))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_FSIZE, (OUTPUT_BYTES, OUTPUT_BYTES))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        # Counted per uid. Under Apptainer the uid is shared with the trainer,
        # so forks may fail outright there; MBPP solutions never need to fork.
        resource.setrlimit(resource.RLIMIT_NPROC, (NPROC, NPROC))
    return apply


def run_one(code, setup, test, nonce, timeout):
    workdir = tempfile.mkdtemp(dir="/tmp")
    out_path = os.path.join(workdir, "out")
    payload = json.dumps({"code": code, "setup": setup, "test": test, "nonce": nonce})
    try:
        with open(out_path, "wb") as out:
            proc = subprocess.Popen(
                [sys.executable, "-I", "-c", CHILD_PRELUDE.format(harness_dir=HARNESS_DIR, profile=PROFILE) + CHILD],
                stdin=subprocess.PIPE, stdout=out, stderr=subprocess.PIPE,
                cwd=workdir, start_new_session=True,
                preexec_fn=_limits(int(timeout) + 1),
                env={"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": workdir},
            )
            try:
                _, err = proc.communicate(payload.encode(), timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.communicate()
                return False, "timeout"
            finally:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)  # reap any children it spawned
                except ProcessLookupError:
                    pass
        with open(out_path, "rb") as f:
            f.seek(max(0, os.path.getsize(out_path) - 256))
            tail = f.read().decode(errors="replace")
        lines = tail.strip().splitlines()
        if proc.returncode == 0 and lines and lines[-1] == nonce:
            return True, ""
        last = err.decode(errors="replace").strip().splitlines()[-1:] or [f"exit {proc.returncode}, no completion marker"]
        return False, last[0][:300]
    except Exception as e:  # harness-side failure: score 0, never crash the batch
        return False, f"harness error: {type(e).__name__}: {e}"[:300]
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main():
    req = json.loads(sys.stdin.read())
    nonce, timeout = req["nonce"], float(req["timeout"])
    pairs = [(j, i) for j in req["jobs"] for i in range(len(j["tests"]))]
    with ThreadPoolExecutor(max_workers=int(req.get("workers", 4))) as pool:
        outcomes = list(pool.map(
            lambda p: run_one(p[0]["code"], p[0].get("setup", ""), p[0]["tests"][p[1]], nonce, timeout),
            pairs,
        ))
    by_job = {}
    for (job, _), (ok, err) in zip(pairs, outcomes):
        r = by_job.setdefault(job["id"], {"id": job["id"], "passed": [], "errors": []})
        r["passed"].append(ok)
        r["errors"].append(err)
    results = [by_job.get(j["id"], {"id": j["id"], "passed": [], "errors": []}) for j in req["jobs"]]
    sys.stdout.write(json.dumps({"results": results}))


if __name__ == "__main__":
    main()
