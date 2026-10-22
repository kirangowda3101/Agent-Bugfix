#!/bin/bash
# Login node, once: create the conda env in scratch and install pinned packages.
#   bash post_training/slurm/setup_env.sh
source "$(dirname "$0")/common.sh"

mkdir -p "$SCRATCH_ROOT/envs" "$LOG_DIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS"
conda info --base > "$SCRATCH_ROOT/envs/conda_base"

if [ ! -x "$ENV_PREFIX/bin/python" ]; then
    # conda-forge only, so the Anaconda defaults-channel terms are not involved.
    conda create -y -p "$ENV_PREFIX" -c conda-forge --override-channels python=3.11
fi
activate_env

# torch 2.14.1 has no CUDA 12.8 wheel; the CUDA 12.6 build runs on the 570 driver.
python -m pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.14.1
python -m pip install -r "$PT/requirements.txt"

# Check every installed version against requirements.txt.
python - "$PT/requirements.txt" <<'EOF'
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
import torch
print("torch CUDA build:", torch.version.cuda)
if bad:
    sys.exit(f"VERSION MISMATCH: {bad}")
print("ENV OK")
EOF
