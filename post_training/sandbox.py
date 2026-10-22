"""Host-side client for the sandbox that runs model-generated code.

Model code never runs on the host. Two backends, selected with
SANDBOX_BACKEND=docker (default, local) or SANDBOX_BACKEND=apptainer (cluster):

docker     One long-lived container: no network, read-only root filesystem,
           all capabilities dropped, no-new-privileges, memory/CPU/pids limits,
           and an explicit seccomp profile. The Docker Desktop daemon on the
           development machine runs containers with seccomp *unconfined* by
           default, so the profile is passed per container.
apptainer  A fresh unprivileged `apptainer exec` per batch: --containall (own
           PID/IPC namespaces, clean environment, no host home or /tmp),
           --net --network none, nothing bind-mounted, no GPU. Apptainer's
           seccomp and cgroup options need root, so those protections come
           from the harness (per-child seccomp filter and rlimits) and from
           SLURM's job cgroup instead.

In both, batches of (code, tests) go to the harness over stdin; nothing from
the host is mounted. The harness runs every assert in a fresh child process
that installs Docker's default seccomp allowlist on itself before model code.
"""
import json
import os
import secrets
import shutil
import subprocess
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
SANDBOX_DIR = HERE / "sandbox"
SECCOMP_PROFILE = SANDBOX_DIR / "seccomp-default.json"  # moby v28.1.1 default profile
IMAGE = "mbpp-sandbox"
BASE_IMAGE = "bugfix-sandbox"


class SandboxError(RuntimeError):
    pass


def _run(cmd, input=None, timeout=60):
    return subprocess.run(cmd, input=input, capture_output=True, text=True, timeout=timeout)


def _docker(*args, input=None, timeout=60):
    return _run(["docker", *args], input=input, timeout=timeout)


def build_image():
    if _docker("image", "inspect", BASE_IMAGE).returncode != 0:
        root = HERE.parent
        r = _docker("build", "-t", BASE_IMAGE, str(root), timeout=600)
        if r.returncode != 0:
            raise SandboxError(f"building {BASE_IMAGE} failed:\n{r.stderr}")
    r = _docker("build", "-t", IMAGE, str(SANDBOX_DIR), timeout=600)
    if r.returncode != 0:
        raise SandboxError(f"building {IMAGE} failed:\n{r.stderr}")


def make_sandbox(**kwargs):
    backend = os.environ.get("SANDBOX_BACKEND", "docker")
    if backend == "docker":
        return Sandbox(**kwargs)
    if backend == "apptainer":
        return ApptainerSandbox(**kwargs)
    raise ValueError(f"unknown SANDBOX_BACKEND {backend!r}")


class Sandbox:
    """Docker backend."""
    backend = "docker"

    def __init__(self, timeout=5.0, workers=4, cpus=4, memory="2g"):
        self.timeout = timeout
        self.workers = workers
        self.cpus = cpus
        self.memory = memory
        self.name = None
        self.restarts = 0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def start(self):
        build_image()  # cached layers make this fast; never run a stale harness
        self.name = f"mbpp-sandbox-{uuid.uuid4().hex[:8]}"
        r = _docker(
            "run", "-d", "--rm", "--name", self.name,
            "--init",  # PID 1 must reap orphans, or killed fork bombs leave zombies that exhaust --pids-limit
            "--network", "none",
            "--read-only", "--tmpfs", "/tmp:rw,nosuid,nodev,size=256m",
            "--memory", self.memory, "--memory-swap", self.memory,
            "--pids-limit", "256", "--cpus", str(self.cpus),
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--security-opt", f"seccomp={SECCOMP_PROFILE}",
            "--user", "sandbox",
            IMAGE, "sleep", "infinity",
        )
        if r.returncode != 0:
            raise SandboxError(f"starting sandbox failed:\n{r.stderr}")

    def stop(self):
        if self.name:
            _docker("rm", "-f", self.name)
            self.name = None

    def restart(self):
        self.stop()
        self.restarts += 1
        self.start()

    def _harness_cmd(self):
        return ["docker", "exec", "-i", self.name, "python", "-I", "/harness/harness.py"]

    def run(self, jobs):
        """jobs: list of {"id", "code", "setup", "tests"}. Returns results in the same order.

        If the sandbox fails or the batch exceeds its deadline, every job in the
        batch scores 0 with the failure recorded, and the sandbox is restarted.
        """
        if not jobs:
            return []
        req = {"nonce": secrets.token_hex(16), "timeout": self.timeout, "workers": self.workers, "jobs": jobs}
        n_tests = sum(len(j["tests"]) for j in jobs)
        deadline = 30 + (self.timeout + 1) * n_tests / self.workers
        try:
            r = _run(self._harness_cmd(), input=json.dumps(req), timeout=deadline)
            if r.returncode != 0:
                raise SandboxError(f"harness exit {r.returncode}: {r.stderr[-500:]}")
            results = json.loads(r.stdout)["results"]
        except (subprocess.TimeoutExpired, SandboxError, json.JSONDecodeError) as e:
            self.restart()
            reason = f"sandbox failure: {type(e).__name__}: {str(e)[:200]}"
            return [{"id": j["id"], "passed": [False] * len(j["tests"]), "errors": [reason] * len(j["tests"])}
                    for j in jobs]
        return results

    def child_seccomp_status(self):
        """(Seccomp mode, number of filters) seen by a harness child, i.e. where model code runs."""
        code = ("import re\nS = open('/proc/self/status').read()\n"
                "MODE = int(re.search(r'Seccomp:\\s+(\\d+)', S).group(1))\n"
                "N = int(re.search(r'Seccomp_filters:\\s+(\\d+)', S).group(1))")
        # The assert always fails; its message carries the values back in the error field.
        res = self.run([{"id": "0", "code": code, "setup": "", "tests": ["assert False, (MODE, N)"]}])[0]
        err = res["errors"][0]
        if not err.startswith("AssertionError: ("):
            raise SandboxError(f"seccomp probe failed: {err}")
        return tuple(int(x) for x in err[len("AssertionError: ("):-1].split(","))

    def expected_filters(self):
        return 2  # at least: the container's profile plus the harness child's own filter

    def seccomp_active(self):
        """True if model code runs under the expected number of seccomp filters.

        For Docker, also checks the container-level profile on a process that is
        not a harness child, so a missing --security-opt is caught on its own.
        """
        r = _docker("exec", self.name, "grep", "^Seccomp:", "/proc/self/status")
        container_ok = r.returncode == 0 and r.stdout.split()[-1] == "2"
        mode, n = self.child_seccomp_status()
        return container_ok and mode == 2 and n >= self.expected_filters()


class ApptainerSandbox(Sandbox):
    """Apptainer backend: one fresh unprivileged container per batch."""
    backend = "apptainer"

    def __init__(self, timeout=5.0, workers=4, sif=None, **_ignored):
        super().__init__(timeout=timeout, workers=workers)
        self.sif = sif or os.environ.get("SANDBOX_SIF")
        if not self.sif or not Path(self.sif).exists():  # a .sif file, or an unpacked image directory
            raise SandboxError(f"SANDBOX_SIF must point to the built mbpp-sandbox.sif (got {self.sif!r})")
        self.apptainer = shutil.which("apptainer")
        if not self.apptainer:
            raise SandboxError("apptainer not found on PATH")

    def start(self):
        pass

    def stop(self):
        pass

    def restart(self):
        self.restarts += 1  # each batch already gets a fresh container

    def _harness_cmd(self):
        return [
            self.apptainer, "exec",
            "--containall",           # own PID/IPC namespaces, clean env, private /tmp and home
            "--no-home",
            "--no-mount", "home,cwd,bind-paths,hostfs",
            "--net", "--network", "none",
            "--pwd", "/tmp",
            self.sif, "python", "-I", "/harness/harness.py",
        ]

    def expected_filters(self):
        return 1  # at least the harness child's own filter (Apptainer adds none unprivileged)

    def seccomp_active(self):
        mode, n = self.child_seccomp_status()
        return mode == 2 and n >= self.expected_filters()
