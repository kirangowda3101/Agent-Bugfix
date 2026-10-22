"""Install Docker's default seccomp allowlist on the current process, via libseccomp.

Runs inside each harness child before any model code. Under Docker this is a
second filter on top of the container's profile; under unprivileged Apptainer,
where `--security seccomp:` is root-only, it is the only one. A process with
no_new_privs set (both runtimes set it) may install a filter without root.

Translation of the moby profile, for a process with no capabilities:
  - rules that `includes` capabilities are skipped (we hold none)
  - rules that `excludes` capabilities apply
  - arch includes/excludes are matched against this machine
  - minKernel is ignored (the rule applies; unknown syscall names are skipped)
Fails closed: any error raises, and the caller must not run model code.
"""
import ctypes
import ctypes.util
import json
import platform

SCMP_ACT_ALLOW = 0x7FFF0000
_OPS = {"SCMP_CMP_NE": 1, "SCMP_CMP_LT": 2, "SCMP_CMP_LE": 3, "SCMP_CMP_EQ": 4,
        "SCMP_CMP_GE": 5, "SCMP_CMP_GT": 6, "SCMP_CMP_MASKED_EQ": 7}
_ARCH = {"x86_64": "amd64", "aarch64": "arm64"}[platform.machine()]


class _ArgCmp(ctypes.Structure):
    _fields_ = [("arg", ctypes.c_uint), ("op", ctypes.c_int),
                ("datum_a", ctypes.c_uint64), ("datum_b", ctypes.c_uint64)]


def _errno(n):
    return 0x00050000 | (n & 0xFFFF)


def _applies(rule):
    inc, exc = rule.get("includes", {}), rule.get("excludes", {})
    if inc.get("caps"):
        return False
    if inc.get("arches") and _ARCH not in inc["arches"]:
        return False
    if exc.get("arches") and _ARCH in exc["arches"]:
        return False
    return True


def install(profile_path):
    lib = ctypes.CDLL(ctypes.util.find_library("seccomp") or "libseccomp.so.2", use_errno=True)
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int,
                                           ctypes.c_uint, ctypes.POINTER(_ArgCmp)]
    lib.seccomp_load.argtypes = [ctypes.c_void_p]

    profile = json.load(open(profile_path))
    ctx = lib.seccomp_init(_errno(profile.get("defaultErrnoRet", 1)))
    if not ctx:
        raise OSError("seccomp_init failed")
    for rule in profile["syscalls"]:
        if not _applies(rule):
            continue
        action = SCMP_ACT_ALLOW if rule["action"] == "SCMP_ACT_ALLOW" else _errno(rule.get("errnoRet", 1))
        args = rule.get("args") or []
        arr = (_ArgCmp * max(len(args), 1))(*[
            _ArgCmp(a["index"], _OPS[a["op"]], a["value"], a.get("valueTwo", 0)) for a in args])
        for name in rule["names"]:
            nr = lib.seccomp_syscall_resolve_name(name.encode())
            if nr < 0:
                continue  # not a syscall on this architecture
            rc = lib.seccomp_rule_add_array(ctx, action, nr, len(args), arr)
            if rc != 0:
                raise OSError(f"seccomp_rule_add({name}) failed: {rc}")
    rc = lib.seccomp_load(ctx)
    if rc != 0:
        raise OSError(f"seccomp_load failed: {rc}")
