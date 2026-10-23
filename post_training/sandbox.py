"""Host-side client for the sandbox that runs model-generated code.

Model code never runs on the host. Two backends, selected with
SANDBOX_BACKEND=docker (default, local) or SANDBOX_BACKEND=apptainer (cluster):

docker     A fresh container per batch: no network, read-only root filesystem,
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
import signal
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
    """Docker backend: one fresh container per batch."""
    backend = "docker"

    def __init__(self, timeout=5.0, workers=4, cpus=4, memory="2g"):
        self.timeout = timeout
        self.workers = workers
        self.cpus = cpus
        self.memory = memory
        self.restarts = 0
        self._batch = None  # name of the container running the current batch

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def start(self):
        build_image()  # cached layers make this fast; never run a stale harness

    def stop(self):
        pass  # containers live for one batch only (--rm)

    def restart(self):
        self.restarts += 1  # the next batch gets a fresh container anyway

    def _container_flags(self):
        return [
            "--init",  # PID 1 must reap orphans, or killed fork bombs leave zombies that exhaust --pids-limit
            "--network", "none",
            "--read-only", "--tmpfs", "/tmp:rw,nosuid,nodev,size=256m",
            "--memory", self.memory, "--memory-swap", self.memory,
            "--pids-limit", "256", "--cpus", str(self.cpus),
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--security-opt", f"seccomp={SECCOMP_PROFILE}",
            "--user", "sandbox",
        ]

    def _harness_cmd(self):
        # A fresh container per batch, so files and processes a batch leaves
        # behind cannot reach the next batch (a long-lived container let both through).
        self._batch = f"mbpp-sandbox-{uuid.uuid4().hex[:8]}"
        return ["docker", "run", "--rm", "-i", "--name", self._batch, *self._container_flags(),
                IMAGE, "python", "-I", "/harness/harness.py"]

    def _kill_batch(self, proc):
        """After a deadline: kill the client's process group, and the batch's container."""
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if self._batch:
            _docker("rm", "-f", self._batch)

    def run(self, jobs):
        """jobs: list of {"id", "code", "setup", "tests"}. Returns results in the same order.

        If the sandbox fails or the batch exceeds its deadline, every job in the
        batch scores 0 with the failure recorded.
        """
        if not jobs:
            return []
        req = {"nonce": secrets.token_hex(16), "timeout": self.timeout, "workers": self.workers, "jobs": jobs}
        n_tests = sum(len(j["tests"]) for j in jobs)
        deadline = 30 + (self.timeout + 1) * n_tests / self.workers
        try:
            proc = subprocess.Popen(self._harness_cmd(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True, start_new_session=True)
            try:
                out, err = proc.communicate(json.dumps(req), timeout=deadline)
            except subprocess.TimeoutExpired:
                self._kill_batch(proc)
                proc.communicate()
                raise
            if proc.returncode != 0:
                raise SandboxError(f"harness exit {proc.returncode}: {err[-500:]}")
            results = json.loads(out)["results"]
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
        r = _docker("run", "--rm", *self._container_flags(), IMAGE, "grep", "^Seccomp:", "/proc/self/status")
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

    def _kill_batch(self, proc):
        # Same as Docker minus the container removal: there is no named container.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

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
