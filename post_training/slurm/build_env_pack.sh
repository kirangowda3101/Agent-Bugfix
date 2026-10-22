#!/bin/bash
# Once, in a CPU job (setup_cpu.sbatch envpack): build the env on node-local
# disk and pack it into ONE file on scratch. Needs internet.
#   sbatch post_training/slurm/setup_cpu.sbatch envpack
#
# Installing straight onto scratch writes tens of thousands of small files to
# the shared filesystem, which is slow there. Here conda, pip and their caches
# all write to local disk; scratch receives a single tar (conda-pack). GPU jobs
# unpack it to their own local disk (activate_env in common.sh).
source "$(dirname "$0")/common.sh"

if [ ! -s "$SCRATCH_ROOT/envs/conda_base" ]; then
    echo "missing $SCRATCH_ROOT/envs/conda_base; on the login node run: conda info --base > $SCRATCH_ROOT/envs/conda_base"
    exit 1
fi
CONDA="$(cat "$SCRATCH_ROOT/envs/conda_base")/bin/conda"

base=$(node_local_dir 30)
WORK="$base/posttrain-build-${SLURM_JOB_ID:-$$}"
LOCAL_CLEANUP+=("$WORK")
mkdir -p "$WORK/tmp" "$SCRATCH_ROOT/envs"
export CONDA_PKGS_DIRS=$WORK/pkgs PIP_CACHE_DIR=$WORK/pip_cache TMPDIR=$WORK/tmp
ENV=$WORK/env

t0=$(date +%s)
# conda-forge only, so the Anaconda defaults-channel terms are not involved.
"$CONDA" create -y -p "$ENV" -c conda-forge --override-channels python=3.11
t1=$(date +%s)
# torch 2.14.1 has no CUDA 12.8 wheel; the CUDA 12.6 build runs on the 570 driver.
"$ENV/bin/python" -m pip install --index-url "$TORCH_INDEX_URL" torch==2.14.1
"$ENV/bin/python" -m pip install -r "$PT/requirements.txt" conda-pack==0.9.2
t2=$(date +%s)
"$ENV/bin/python" "$PT/slurm/check_env.py" "$PT/requirements.txt"
"$ENV/bin/python" -m pip freeze > "$WORK/manifest.txt"
echo "files in env: $(find "$ENV" -type f | wc -l), size: $(du -sh "$ENV" | cut -f1)"

"$ENV/bin/conda-pack" -p "$ENV" -o "$WORK/posttrain-env.tar" --format tar --n-threads "${SLURM_CPUS_PER_TASK:-4}"
t3=$(date +%s)

# Copy under a temporary name, then rename, so a half-copied tar is never used.
cp "$WORK/posttrain-env.tar" "$ENV_PACK.partial"
mv "$ENV_PACK.partial" "$ENV_PACK"
cp "$WORK/manifest.txt" "$ENV_PACK.manifest.txt"
t4=$(date +%s)
ls -l "$ENV_PACK"
echo "timings (s): conda $((t1 - t0)), pip $((t2 - t1)), check+pack $((t3 - t2)), copy to scratch $((t4 - t3))"
echo "ENV PACK OK"
