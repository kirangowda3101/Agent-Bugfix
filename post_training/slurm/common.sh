# Sourced by every setup script and job. All paths live here and only here.
# Everything goes to scratch so nothing uses the home-directory quota.
set -euo pipefail

export SCRATCH_ROOT=/scratch/ramanagarajayaram.k
export REPO=$SCRATCH_ROOT/agent-bugfix
export PT=$REPO/post_training
export ENV_PREFIX=$SCRATCH_ROOT/envs/posttrain          # env installed directly on scratch (slow to build)
export ENV_PACK=$SCRATCH_ROOT/envs/posttrain-env.tar     # the same env packed into one file (build_env_pack.sh)
export LOG_DIR=$SCRATCH_ROOT/logs
export TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}

export HF_HOME=$SCRATCH_ROOT/hf_cache
export PIP_CACHE_DIR=$SCRATCH_ROOT/pip_cache
export CONDA_PKGS_DIRS=$SCRATCH_ROOT/conda_pkgs
export APPTAINER_CACHEDIR=$SCRATCH_ROOT/apptainer_cache
export APPTAINER_TMPDIR=$SCRATCH_ROOT/apptainer_tmp
export PYTHONNOUSERSITE=1   # ignore packages in ~/.local

export SANDBOX_BACKEND=apptainer
export SANDBOX_SIF=$SCRATCH_ROOT/sif/mbpp-sandbox.sif
# build_sandbox.sh records which image actually runs: the SIF, or a tar of the
# unpacked image directory (unpacked to local disk per job, see prepare_sandbox_image).
SANDBOX_IMAGE_RECORD=$SCRATCH_ROOT/sif/image_path
if [ -f "$SANDBOX_IMAGE_RECORD" ]; then
    SANDBOX_SIF=$(cat "$SANDBOX_IMAGE_RECORD")
fi
export SANDBOX_REQUIRED=1   # a missing sandbox fails tests instead of skipping them

# Directories created on node-local disk by this job, removed when the job's shell exits.
LOCAL_CLEANUP=()
cleanup_local() {
    local d
    for d in "${LOCAL_CLEANUP[@]}"; do rm -rf "$d"; done
}
trap cleanup_local EXIT

# Print a node-local, writable directory with at least $1 GB free. Network
# filesystems are skipped: the point is to avoid them. tmpfs is accepted but
# reported, because files there count against the job's memory.
node_local_dir() {
    local need=$1 d fstype avail
    for d in ${SLURM_TMPDIR:-} ${TMPDIR:-} /tmp /local /lscratch /scratch_local; do
        [ -n "$d" ] && [ -d "$d" ] && [ -w "$d" ] || continue
        fstype=$(stat -f -c %T "$d")
        case "$fstype" in
            nfs*|lustre|gpfs|beegfs|cifs|smb*|panfs|wekafs|ceph|fuse*) continue ;;
        esac
        avail=$(df -Pk "$d" | awk 'NR==2 {print int($4 / 1048576)}')
        if [ "$avail" -ge "$need" ]; then
            echo "node-local dir: $d ($fstype, ${avail} GB free)" >&2
            [ "$fstype" = "tmpfs" ] && echo "  note: $d is tmpfs, so its files use this job's RAM" >&2
            echo "$d"
            return 0
        fi
    done
    echo "NO NODE-LOCAL DISK with ${need} GB free among: ${SLURM_TMPDIR:-} ${TMPDIR:-} /tmp /local /lscratch /scratch_local" >&2
    df -h /tmp >&2 || true
    return 1
}

# Prefer the packed env, unpacked to node-local disk (one big sequential read of
# scratch instead of tens of thousands of small files). Fall back to an env
# installed directly on scratch, if one finished.
activate_env() {
    if [ -f "$ENV_PACK" ]; then
        local base dir t0
        base=$(node_local_dir 15)
        dir="$base/posttrain-env-${SLURM_JOB_ID:-$$}"
        LOCAL_CLEANUP+=("$dir")
        mkdir -p "$dir"
        t0=$(date +%s)
        tar -xf "$ENV_PACK" -C "$dir"
        set +u; source "$dir/bin/activate"; set -u
        conda-unpack
        export TORCHINDUCTOR_CACHE_DIR=$dir/.torchinductor   # keep torch's cache inside the job dir
        echo "env: unpacked $ENV_PACK to $dir in $(( $(date +%s) - t0 ))s"
    elif [ -f "$ENV_PREFIX/.setup_complete" ]; then
        local base
        base=$(cat "$SCRATCH_ROOT/envs/conda_base")
        set +u; source "$base/etc/profile.d/conda.sh"; conda activate "$ENV_PREFIX"; set -u
        echo "env: $ENV_PREFIX (on scratch)"
    else
        echo "NO USABLE ENV: neither $ENV_PACK nor a completed $ENV_PREFIX exists"
        return 1
    fi
}

# If the sandbox image is a tar of the image directory, unpack it to local disk.
prepare_sandbox_image() {
    if [[ "$SANDBOX_SIF" == *.tar ]]; then
        local base dir
        base=$(node_local_dir 2)
        dir="$base/mbpp-sandbox-${SLURM_JOB_ID:-$$}"
        LOCAL_CLEANUP+=("$dir")
        mkdir -p "$dir"
        tar -xf "$SANDBOX_SIF" -C "$dir"
        export SANDBOX_SIF="$dir/mbpp-sandbox.dir"
    fi
    echo "sandbox image: $SANDBOX_SIF"
}

# Start of every GPU job: check the GPU, then run the full sandbox test suite on
# this node. Any failed, skipped or erroring test stops the job before training.
gpu_job_start() {
    cd "$PT"
    activate_env
    prepare_sandbox_image
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
