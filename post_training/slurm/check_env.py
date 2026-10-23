"""Check every installed package version against requirements.txt; exit non-zero on any mismatch."""
import sys
from importlib.metadata import version

bad = []
for line in open(sys.argv[1]):
    if "==" not in line or line.startswith("#"):
        continue
    name, want = line.strip().split("==")
    got = version(name)
    print(f"{name:14s} {got}")
    if got.split("+")[0] != want:
        bad.append((name, want, got))

import torch  # noqa: E402

print("torch CUDA build:", torch.version.cuda)
if bad:
    sys.exit(f"VERSION MISMATCH: {bad}")
