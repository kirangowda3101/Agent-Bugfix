#!/bin/bash
# Create the conda env directly on scratch and install pinned packages. Needs internet.
# Slow on a shared filesystem (tens of thousands of small files): prefer build_env_pack.sh.
# Runs inside a CPU job (setup_cpu.sbatch); the login node's limits kill the conda solver.
#
# Safe to rerun: the env counts as finished only once every version check has
# passed and $ENV_PREFIX/.setup_complete exists. An env without that marker is
# a leftover from an interrupted run and is deleted and rebuilt.
source "$(dirname "$0")/common.sh"

mkdir -p "$SCRATCH_ROOT/envs" "$LOG_DIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS"
if [ ! -s "$SCRATCH_ROOT/envs/conda_base" ]; then
    echo "missing $SCRATCH_ROOT/envs/conda_base; on the login node run: conda info --base > $SCRATCH_ROOT/envs/conda_base"
    exit 1
fi
CONDA="$(cat "$SCRATCH_ROOT/envs/conda_base")/bin/conda"

if [ -f "$ENV_PREFIX/.setup_complete" ]; then
    echo "env already complete at $ENV_PREFIX"
else
    if [ -e "$ENV_PREFIX" ]; then
        echo "removing incomplete env at $ENV_PREFIX"
        rm -rf "$ENV_PREFIX"
    fi
    # conda-forge only, so the Anaconda defaults-channel terms are not involved.
    "$CONDA" create -y -p "$ENV_PREFIX" -c conda-forge --override-channels python=3.11
fi
set +u; source "$(cat "$SCRATCH_ROOT/envs/conda_base")/etc/profile.d/conda.sh"; conda activate "$ENV_PREFIX"; set -u

# torch 2.14.1 has no CUDA 12.8 wheel; the CUDA 12.6 build runs on the 570 driver.
python -m pip install --index-url "$TORCH_INDEX_URL" torch==2.14.1
python -m pip install -r "$PT/requirements.txt"

python "$PT/slurm/check_env.py" "$PT/requirements.txt"
date -Is > "$ENV_PREFIX/.setup_complete"
echo "ENV OK"
