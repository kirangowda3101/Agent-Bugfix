import os
import time
import uuid
from pathlib import Path

import pytest

from rewards import extract_code, make_reward_fn, score

TESTS = ["assert add(1, 2) == 3", "assert add(0, 0) == 0", "assert add(-1, 5) == 4"]


def fenced(code):
    return f"Here you go:\n```python\n{code}\n```\nDone."


def reward(sandbox, code, tests=TESTS, setup=""):
    rewards, details = score(sandbox, [fenced(code)], [tests], [setup])
    return rewards[0], details[0]


def expect(sandbox, code, want, tests=TESTS, setup=""):
    """Assert the reward, and show the harness's error text if it is wrong."""
    r, d = reward(sandbox, code, tests, setup)
    assert r == pytest.approx(want), f"reward {r}, expected {want}; harness errors: {d['errors']}"
    return d


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
    expect(sandbox, "def add(a, b):\n    return a + b", 1.0)


@pytest.mark.sandbox
def test_partial_solution_scores_fraction(sandbox):
    # abs() is right for the first two asserts, wrong for add(-1, 5).
    d = expect(sandbox, "def add(a, b):\n    return abs(a) + abs(b)", 2 / 3)
    assert d["passed"] == [True, True, False], d["errors"]


@pytest.mark.sandbox
def test_syntax_error_scores_zero(sandbox):
    d = expect(sandbox, "def add(a, b) return a + b", 0.0)
    assert "SyntaxError" in d["errors"][0], d["errors"]


@pytest.mark.sandbox
def test_no_code_block_scores_zero_without_running(sandbox):
    rewards, details = score(sandbox, ["def add(a, b): return a + b"], [TESTS], [""])
    assert rewards == [0.0] and details[0]["errors"][0] == "no code block", details[0]["errors"]


@pytest.mark.sandbox
def test_setup_code_is_run(sandbox):
    expect(sandbox, "def f():\n    return K", 1.0, tests=["assert f() == 7"], setup="K = 7")


@pytest.mark.sandbox
def test_main_block_is_not_executed(sandbox):
    expect(sandbox, "def add(a, b):\n    return a + b\nif __name__ == '__main__':\n    input()", 1.0)


# --- containment ---------------------------------------------------------------

@pytest.mark.sandbox
def test_infinite_loop_times_out(sandbox):
    start = time.monotonic()
    d = expect(sandbox, "def add(a, b):\n    while True: pass", 0.0, tests=TESTS[:1])
    assert d["errors"][0] == "timeout", d["errors"]
    assert time.monotonic() - start < 15


@pytest.mark.sandbox
def test_memory_blowup_is_contained(sandbox):
    expect(sandbox, "x = bytearray(4 * 1024**3)\ndef add(a, b): return a + b", 0.0, tests=TESTS[:1])


@pytest.mark.sandbox
def test_fork_bomb_is_contained(sandbox):
    expect(sandbox, "import os\nwhile True:\n    try: os.fork()\n    except OSError: pass", 0.0, tests=TESTS[:1])
    # The sandbox must still work afterwards.
    expect(sandbox, "def add(a, b):\n    return a + b", 1.0)


@pytest.mark.sandbox
def test_huge_output_is_contained(sandbox):
    expect(sandbox, "while True: print('x' * 10000)\ndef add(a, b): return a + b", 0.0, tests=TESTS[:1])


@pytest.mark.sandbox
def test_no_network(sandbox):
    code = (
        "import socket\n"
        "def add(a, b):\n"
        "    s = socket.socket(); s.settimeout(2)\n"
        "    s.connect(('1.1.1.1', 80))\n"
        "    return a + b"
    )
    expect(sandbox, code, 0.0, tests=TESTS[:1])


@pytest.mark.sandbox
def test_runs_as_non_root(sandbox):
    expect(sandbox, "import os\nUID = os.getuid()", 1.0, tests=["assert UID != 0, UID"])


# Probe used by the read-only test: try a real write in every image directory.
# statvfs is reported but not relied on: on the cluster's GPU nodes (Apptainer
# 1.2.5) / is fuse-overlayfs, which reports "rw" yet refuses every write (EROFS).
WRITE_PROBE = """
import errno, os
ME = os.getuid()
RESULT, OWNED = {}, []
for d in ["/", "/usr", "/usr/local/lib", "/etc", "/harness", "/root", "/home", "/home/sandbox", "/var", "/opt"]:
    if not os.path.isdir(d):
        continue
    if os.stat(d).st_uid == ME:
        OWNED.append(d)
    p = os.path.join(d, ".write_probe_%d" % os.getpid())
    try:
        open(p, "w").close()
        os.remove(p)
        RESULT[d] = "WRITABLE"
    except OSError as e:
        RESULT[d] = errno.errorcode.get(e.errno, str(e.errno))
STATVFS = "ro" if os.statvfs("/").f_flag & os.ST_RDONLY else "rw"
"""


@pytest.mark.sandbox
def test_image_is_not_writable(sandbox):
    expect(sandbox, WRITE_PROBE, 1.0, tests=[
        # Every image directory refuses writes...
        "assert all(v in ('EROFS', 'EACCES') for v in RESULT.values()), (RESULT, STATVFS)",
        # ...and directories this uid owns refuse with EROFS, so it is the filesystem,
        # not file permissions, that stops the write. (Docker: /home/sandbox. Apptainer:
        # the image was built by the same user, so its directories belong to this uid.)
        "assert OWNED and all(RESULT[d] == 'EROFS' for d in OWNED), (OWNED, RESULT, STATVFS)",
    ])


@pytest.mark.sandbox
def test_nothing_persists_between_batches(sandbox):
    # Batch 1 writes files in the writable areas and leaves a detached process
    # (its own session, so the harness's process-group kill misses it) that
    # writes more files 2 s later. Batch 2, after 4 s, must see none of them.
    tag = uuid.uuid4().hex[:8]
    paths = [f"/tmp/leak_{tag}", f"/var/tmp/leak_{tag}", f"/tmp/late_{tag}", f"/var/tmp/late_{tag}"]
    batch1 = (
        "import os, time\n"
        f"for p in {paths[:2]!r}:\n"
        "    try: open(p, 'w').write('x')\n"
        "    except OSError: pass\n"
        "try:\n"
        "    if os.fork() == 0:\n"
        "        os.setsid(); time.sleep(2)\n"
        f"        for p in {paths[2:]!r}:\n"
        "            try: open(p, 'w').write('x')\n"
        "            except OSError: pass\n"
        "        os._exit(0)\n"
        "except OSError:\n"
        "    pass  # fork refused (process limit): the file part of the test still runs\n"
    )
    expect(sandbox, batch1, 1.0, tests=["pass"])
    time.sleep(4)
    expect(sandbox, f"import os\nSEEN = [p for p in {paths!r} if os.path.exists(p)]", 1.0,
           tests=["assert SEEN == [], SEEN"])


@pytest.mark.sandbox
def test_no_capabilities_and_no_new_privs(sandbox):
    code = "S = open('/proc/self/status').read()"
    expect(sandbox, code, 1.0, tests=["assert 'CapEff:\\t0000000000000000' in S", "assert 'NoNewPrivs:\\t1' in S"])


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
    expect(sandbox, code, 1.0, tests=["assert VISIBLE == [], VISIBLE"])


@pytest.mark.sandbox
def test_no_gpu_devices(sandbox):
    code = "import glob\nDEV = glob.glob('/dev/nvidia*') + glob.glob('/dev/dri/*')"
    expect(sandbox, code, 1.0, tests=["assert DEV == [], DEV"])


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
    code = f"import ctypes\nlibc = ctypes.CDLL(None, use_errno=True)\nRC = {call}\nERR = ctypes.get_errno()"
    expect(sandbox, code, 0.0, tests=["assert RC >= 0, (RC, ERR)"])


# --- reward-hacking attempts ---------------------------------------------------

@pytest.mark.sandbox
@pytest.mark.parametrize("exit_call", ["import sys\nsys.exit(0)", "import os\nos._exit(0)", "exit(0)", "quit()"])
def test_early_exit_does_not_pass(sandbox, exit_call):
    expect(sandbox, f"def add(a, b):\n    return 0\n{exit_call}", 0.0)


@pytest.mark.sandbox
def test_printing_fake_success_does_not_pass(sandbox):
    expect(sandbox, "import os\nos.write(1, b'\\nPASSED\\n')\nos._exit(0)", 0.0)


@pytest.mark.sandbox
def test_tests_are_isolated_from_each_other(sandbox):
    # Python state written while running one assert must not leak into the next.
    code = (
        "import builtins\n"
        "builtins.COUNT = getattr(builtins, 'COUNT', 0) + 1\n"
        "def f():\n    return builtins.COUNT"
    )
    expect(sandbox, code, 1.0, tests=["assert f() == 1"] * 3)


@pytest.mark.sandbox
def test_always_equal_object_is_a_known_gap(sandbox):
    # Documented limitation: an object whose __eq__ always returns True passes
    # equality asserts. The sandbox cannot tell; the audit flags it statically.
    expect(sandbox, "class A:\n    def __eq__(self, o): return True\ndef add(a, b):\n    return A()", 1.0)


@pytest.mark.sandbox
def test_children_in_one_batch_share_tmp_known_gap(sandbox):
    # Documented limitation: children running in the same batch are in the same
    # container, under the same uid, so a file one writes in /tmp is visible to
    # another. Separating them needs separate uids or mount namespaces, which an
    # unprivileged sandbox (and the seccomp profile, which blocks unshare) rules out.
    # If this test starts failing, the gap has closed: update the README.
    tag = uuid.uuid4().hex[:8]
    writer = fenced(f"open('/tmp/shared_{tag}', 'w').write('x')\ndef f(): return 1")
    reader = fenced(
        "import os, time\n"
        f"for _ in range(30):\n    if os.path.exists('/tmp/shared_{tag}'): break\n    time.sleep(0.1)\n"
        f"SEEN = os.path.exists('/tmp/shared_{tag}')"
    )
    rewards, details = score(sandbox, [writer, reader], [["assert f() == 1"], ["assert SEEN"]], ["", ""])
    assert rewards == [1.0, 1.0], [d["errors"] for d in details]


@pytest.mark.sandbox
def test_test_source_is_readable_known_gap(sandbox):
    # Documented limitation: model code runs in the same Python process as the
    # assert it is scored on, so it can read that assert and return the expected
    # value without solving anything. Here it reads the harness's payload; frame
    # or gc introspection would work too. Closing this means comparing results in
    # a separate process; until then the audit flags introspection statically.
    code = (
        "import __main__\n"
        "EXPECTED = eval(__main__.payload['test'].split('==', 1)[1])\n"
        "def add(a, b):\n    return EXPECTED"
    )
    expect(sandbox, code, 1.0)


# --- batching and TRL interface ------------------------------------------------

@pytest.mark.sandbox
def test_batch_keeps_order_and_isolates_failures(sandbox):
    good = fenced("def add(a, b):\n    return a + b")
    loop = fenced("def add(a, b):\n    while True: pass")
    rewards, details = score(sandbox, [good, "no code", loop, good], [TESTS] * 4, [""] * 4)
    assert rewards == [1.0, 0.0, 0.0, 1.0], [d["errors"] for d in details]


@pytest.mark.sandbox
def test_trl_reward_fn_signature(sandbox):
    fn = make_reward_fn(sandbox)
    completions = [[{"role": "assistant", "content": fenced("def add(a, b):\n    return a + b")}]]
    assert fn(completions=completions, test_list=[TESTS], test_setup_code=[""], prompts=None) == [1.0]
