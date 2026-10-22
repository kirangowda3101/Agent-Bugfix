import os
import time
from pathlib import Path

import pytest

from rewards import extract_code, make_reward_fn, score

TESTS = ["assert add(1, 2) == 3", "assert add(0, 0) == 0", "assert add(-1, 5) == 4"]


def fenced(code):
    return f"Here you go:\n```python\n{code}\n```\nDone."


def reward(sandbox, code, tests=TESTS, setup=""):
    rewards, details = score(sandbox, [fenced(code)], [tests], [setup])
    return rewards[0], details[0]


# --- extraction (no sandbox) -------------------------------------------------

def test_extract_prefers_python_block():
    text = "```\nnot this\n```\n```python\ndef f(): pass\n```"
    assert extract_code(text) == "def f(): pass\n"


def test_extract_falls_back_to_plain_fence():
    assert extract_code("```\nx = 1\n```") == "x = 1\n"


def test_extract_none_without_fence():
    assert extract_code("def add(a, b): return a + b") is None


def test_extract_conversational_format():
    msgs = [{"role": "assistant", "content": fenced("x = 1")}]
    assert extract_code(msgs) == "x = 1\n"


# --- reward values -------------------------------------------------------------

@pytest.mark.sandbox
def test_correct_solution_scores_one(sandbox):
    assert reward(sandbox, "def add(a, b):\n    return a + b")[0] == 1.0


@pytest.mark.sandbox
def test_partial_solution_scores_fraction(sandbox):
    # abs() is right for the first two asserts, wrong for add(-1, 5).
    r, d = reward(sandbox, "def add(a, b):\n    return abs(a) + abs(b)")
    assert d["passed"] == [True, True, False]
    assert r == pytest.approx(2 / 3)


@pytest.mark.sandbox
def test_syntax_error_scores_zero(sandbox):
    r, d = reward(sandbox, "def add(a, b) return a + b")
    assert r == 0.0 and "SyntaxError" in d["errors"][0]


@pytest.mark.sandbox
def test_no_code_block_scores_zero_without_running(sandbox):
    rewards, details = score(sandbox, ["def add(a, b): return a + b"], [TESTS], [""])
    assert rewards == [0.0] and details[0]["errors"][0] == "no code block"


@pytest.mark.sandbox
def test_setup_code_is_run(sandbox):
    r, _ = reward(sandbox, "def f():\n    return K", tests=["assert f() == 7"], setup="K = 7")
    assert r == 1.0


@pytest.mark.sandbox
def test_main_block_is_not_executed(sandbox):
    code = "def add(a, b):\n    return a + b\nif __name__ == '__main__':\n    input()"
    assert reward(sandbox, code)[0] == 1.0


# --- containment ---------------------------------------------------------------

@pytest.mark.sandbox
def test_infinite_loop_times_out(sandbox):
    start = time.monotonic()
    r, d = reward(sandbox, "def add(a, b):\n    while True: pass", tests=TESTS[:1])
    assert r == 0.0 and d["errors"][0] == "timeout"
    assert time.monotonic() - start < 15


@pytest.mark.sandbox
def test_memory_blowup_is_contained(sandbox):
    r, _ = reward(sandbox, "x = bytearray(4 * 1024**3)\ndef add(a, b): return a + b", tests=TESTS[:1])
    assert r == 0.0


@pytest.mark.sandbox
def test_fork_bomb_is_contained(sandbox):
    code = "import os\nwhile True:\n    try: os.fork()\n    except OSError: pass"
    r, _ = reward(sandbox, code, tests=TESTS[:1])
    assert r == 0.0
    # The sandbox must still work afterwards.
    assert reward(sandbox, "def add(a, b):\n    return a + b")[0] == 1.0


@pytest.mark.sandbox
def test_huge_output_is_contained(sandbox):
    r, _ = reward(sandbox, "while True: print('x' * 10000)\ndef add(a, b): return a + b", tests=TESTS[:1])
    assert r == 0.0


@pytest.mark.sandbox
def test_no_network(sandbox):
    code = (
        "import socket\n"
        "def add(a, b):\n"
        "    s = socket.socket(); s.settimeout(2)\n"
        "    s.connect(('1.1.1.1', 80))\n"
        "    return a + b"
    )
    assert reward(sandbox, code, tests=TESTS[:1])[0] == 0.0


@pytest.mark.sandbox
def test_runs_as_non_root(sandbox):
    assert reward(sandbox, "import os\nUID = os.getuid()", tests=["assert UID != 0"])[0] == 1.0


@pytest.mark.sandbox
def test_root_filesystem_is_read_only(sandbox):
    code = "import os\nRO = bool(os.statvfs('/').f_flag & os.ST_RDONLY)"
    assert reward(sandbox, code, tests=["assert RO"])[0] == 1.0
    # Docker: /home/sandbox is owned by the sandbox user, so only --read-only stops this write.
    code = "def try_write():\n    open('/home/sandbox/x', 'w')"
    assert reward(sandbox, code, tests=["try_write()"])[0] == 0.0


@pytest.mark.sandbox
def test_no_capabilities_and_no_new_privs(sandbox):
    code = "S = open('/proc/self/status').read()"
    tests = ["assert 'CapEff:\\t0000000000000000' in S", "assert 'NoNewPrivs:\\t1' in S"]
    assert reward(sandbox, code, tests=tests)[0] == 1.0


@pytest.mark.sandbox
def test_host_files_not_visible(sandbox):
    # Files that exist on the host must not be readable inside. (Directories are
    # not checked: Apptainer --containall creates an empty home directory.)
    host_files = [str(Path(__file__).resolve())]
    if os.environ.get("SANDBOX_SIF"):
        host_files.append(os.environ["SANDBOX_SIF"])
    home_files = sorted(p for p in Path.home().iterdir() if p.is_file())
    if home_files:
        host_files.append(str(home_files[0]))
    code = f"import os\nVISIBLE = [p for p in {host_files!r} if os.path.exists(p)]"
    r, d = reward(sandbox, code, tests=["assert VISIBLE == [], VISIBLE"])
    assert r == 1.0, d["errors"]


@pytest.mark.sandbox
def test_no_gpu_devices(sandbox):
    code = "import glob\nDEV = glob.glob('/dev/nvidia*') + glob.glob('/dev/dri/*')"
    assert reward(sandbox, code, tests=["assert DEV == [], DEV"])[0] == 1.0


@pytest.mark.sandbox
def test_seccomp_filter_is_active(sandbox):
    # The Docker daemon default on the dev machine is seccomp=unconfined, and
    # unprivileged Apptainer cannot apply a profile at all, so the filters must
    # come from the per-container profile (Docker) and the harness (both).
    mode, n = sandbox.child_seccomp_status()
    assert mode == 2 and n >= sandbox.expected_filters(), (mode, n)
    assert sandbox.seccomp_active()


@pytest.mark.sandbox
@pytest.mark.parametrize("call", [
    "libc.unshare(0x10000000)",      # CLONE_NEWUSER: succeeds in this image when no filter is applied
    "libc.socket(40, 1, 0)",         # AF_VSOCK: opens in this image when no filter is applied
    "libc.mount(b'none', b'/tmp', b'tmpfs', 0, None)",
])
def test_seccomp_blocks_syscalls(sandbox, call):
    code = f"import ctypes\nlibc = ctypes.CDLL(None, use_errno=True)\nRC = {call}"
    assert reward(sandbox, code, tests=["assert RC >= 0"])[0] == 0.0


# --- reward-hacking attempts ---------------------------------------------------

@pytest.mark.sandbox
@pytest.mark.parametrize("exit_call", ["import sys\nsys.exit(0)", "import os\nos._exit(0)", "exit(0)", "quit()"])
def test_early_exit_does_not_pass(sandbox, exit_call):
    assert reward(sandbox, f"def add(a, b):\n    return 0\n{exit_call}")[0] == 0.0


@pytest.mark.sandbox
def test_printing_fake_success_does_not_pass(sandbox):
    code = "import os\nos.write(1, b'\\nPASSED\\n')\nos._exit(0)"
    assert reward(sandbox, code)[0] == 0.0


@pytest.mark.sandbox
def test_tests_are_isolated_from_each_other(sandbox):
    # State written while running one assert must not leak into the next.
    code = (
        "import builtins\n"
        "builtins.COUNT = getattr(builtins, 'COUNT', 0) + 1\n"
        "def f():\n    return builtins.COUNT"
    )
    assert reward(sandbox, code, tests=["assert f() == 1"] * 3)[0] == 1.0


@pytest.mark.sandbox
def test_always_equal_object_is_a_known_gap(sandbox):
    # Documented limitation: an object whose __eq__ always returns True passes
    # equality asserts. The sandbox cannot tell; the audit flags it statically.
    code = "class A:\n    def __eq__(self, o): return True\ndef add(a, b):\n    return A()"
    assert reward(sandbox, code)[0] == 1.0


# --- batching and TRL interface ------------------------------------------------

@pytest.mark.sandbox
def test_batch_keeps_order_and_isolates_failures(sandbox):
    good = fenced("def add(a, b):\n    return a + b")
    loop = fenced("def add(a, b):\n    while True: pass")
    rewards, _ = score(sandbox, [good, "no code", loop, good], [TESTS] * 4, [""] * 4)
    assert rewards == [1.0, 0.0, 0.0, 1.0]


@pytest.mark.sandbox
def test_trl_reward_fn_signature(sandbox):
    fn = make_reward_fn(sandbox)
    completions = [[{"role": "assistant", "content": fenced("def add(a, b):\n    return a + b")}]]
    assert fn(completions=completions, test_list=[TESTS], test_setup_code=[""], prompts=None) == [1.0]
