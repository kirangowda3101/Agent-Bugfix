# Sourced by every setup script and job. All paths live here and only here.
# Everything goes to scratch so nothing uses the home-directory quota.
set -euo pipefail

export SCRATCH_ROOT=/scratch/ramanagarajayaram.k
export REPO=$SCRATCH_ROOT/agent-bugfix
export PT=$REPO/post_training
export ENV_PREFIX=$SCRATCH_ROOT/envs/posttrain
export LOG_DIR=$SCRATCH_ROOT/logs

export HF_HOME=$SCRATCH_ROOT/hf_cache
export PIP_CACHE_DIR=$SCRATCH_ROOT/pip_cache
export CONDA_PKGS_DIRS=$SCRATCH_ROOT/conda_pkgs
export APPTAINER_CACHEDIR=$SCRATCH_ROOT/apptainer_cache
export APPTAINER_TMPDIR=$SCRATCH_ROOT/apptainer_tmp
export PYTHONNOUSERSITE=1   # ignore packages in ~/.local

export SANDBOX_BACKEND=apptainer
export SANDBOX_SIF=$SCRATCH_ROOT/sif/mbpp-sandbox.sif
# build_sandbox.sh records which image actually runs (the SIF, or its directory fallback).
if [ -f "$SCRATCH_ROOT/sif/image_path" ]; then
    SANDBOX_SIF=$(cat "$SCRATCH_ROOT/sif/image_path")
fi
export SANDBOX_REQUIRED=1   # a missing sandbox fails tests instead of skipping them

activate_env() {
    # setup_env.sh records conda's location, because batch shells may not have conda on PATH.
    local base
    base=$(cat "$SCRATCH_ROOT/envs/conda_base")
    set +u; source "$base/etc/profile.d/conda.sh"; conda activate "$ENV_PREFIX"; set -u
}

# Start of every GPU job: check the GPU, then run the full sandbox test suite on
# this node. Any failed, skipped or erroring test stops the job before training.
gpu_job_start() {
    cd "$PT"
    activate_env
    export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
    echo "job $SLURM_JOB_ID on $(hostname) at $(date -Is), commit $(git -C "$REPO" rev-parse --short HEAD)"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    python -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(), 'bf16', torch.cuda.is_bf16_supported())"

    local report="$LOG_DIR/sandbox_tests_${SLURM_JOB_ID}.txt"
    set +e
    python -m pytest -q -rA tests/ > "$report" 2>&1
    local rc=$?
    set -e
    tail -n 3 "$report"
    if [ $rc -ne 0 ] || grep -qE "[0-9]+ (failed|skipped|error)" "$report"; then
        echo "SANDBOX TESTS DID NOT ALL PASS (see $report); stopping before any training."
        exit 1
    fi
    echo "sandbox tests: all passed"
}
