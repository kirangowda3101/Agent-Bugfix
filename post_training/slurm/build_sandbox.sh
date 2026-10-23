#!/bin/bash
# Once, in a CPU job (setup_cpu.sbatch sandbox): build the sandbox image without root or fakeroot.
#   sbatch post_training/slurm/setup_cpu.sbatch sandbox
#
# Building from a .def file unprivileged makes Apptainer imply --fakeroot, which
# needs /etc/subuid mappings that a cluster account may not have. Instead:
#   1. pull the pinned base image into a directory (docker:// source: no fakeroot)
#   2. copy the harness files in
#   3. pack the directory into a SIF
# Base: python:3.11-slim index digest (Python 3.11.17, Debian 13, libseccomp 2.6.0).
# Apptainer rejects tag+digest references, so the digest is used alone.
#
# The image directory is built on node-local disk; scratch only receives single
# files: the SIF, or (if the SIF cannot run here, e.g. FUSE problems) a tar of
# the directory that each job unpacks to its own local disk.
#
# No model code runs here. The only check is that the image starts and has
# Python and libseccomp. Both forms are mounted read-only at run time.
source "$(dirname "$0")/common.sh"

BASE=docker://python@sha256:cce12316475d7d56d605f76e1d811d0dc4e7227f2d4afd29951521264f504ff8
SIF_DIR=$SCRATCH_ROOT/sif
SIF=$SIF_DIR/mbpp-sandbox.sif
DIR_TAR=$SIF_DIR/mbpp-sandbox.dir.tar

base=$(node_local_dir 3)
WORK="$base/mbpp-sandbox-build-${SLURM_JOB_ID:-$$}"
LOCAL_CLEANUP+=("$WORK")
IMAGE_DIR=$WORK/mbpp-sandbox.dir
mkdir -p "$SIF_DIR" "$APPTAINER_CACHEDIR" "$WORK/tmp"
export APPTAINER_TMPDIR=$WORK/tmp

apptainer build --sandbox "$IMAGE_DIR" "$BASE"
mkdir -p "$IMAGE_DIR/harness"
cp "$PT/sandbox/harness.py" "$PT/sandbox/seccomp_filter.py" "$PT/sandbox/seccomp-default.json" "$IMAGE_DIR/harness/"
chmod 0444 "$IMAGE_DIR"/harness/*
apptainer build --force "$WORK/mbpp-sandbox.sif" "$IMAGE_DIR"

check() {
    apptainer exec --containall --no-home --no-mount home,cwd,bind-paths,hostfs --net --network none --pwd /tmp "$1" \
        python -c "import ctypes.util, os, sys; print(sys.version.split()[0], ctypes.util.find_library('seccomp'), sorted(os.listdir('/harness')))"
}
if check "$WORK/mbpp-sandbox.sif"; then
    cp "$WORK/mbpp-sandbox.sif" "$SIF.partial" && mv "$SIF.partial" "$SIF"
    sha256sum "$SIF" | tee "$SIF.sha256"
    echo "$SIF" > "$SANDBOX_IMAGE_RECORD"
elif check "$IMAGE_DIR"; then
    echo "WARNING: the packed SIF does not run here; using a tar of the directory image instead."
    tar -cf "$DIR_TAR.partial" -C "$WORK" mbpp-sandbox.dir && mv "$DIR_TAR.partial" "$DIR_TAR"
    sha256sum "$DIR_TAR" | tee "$DIR_TAR.sha256"
    echo "$DIR_TAR" > "$SANDBOX_IMAGE_RECORD"
else
    echo "SANDBOX IMAGE FAILED: neither the SIF nor the directory image runs."
    exit 1
fi
echo "sandbox image: $(cat "$SANDBOX_IMAGE_RECORD")"
echo "SANDBOX IMAGE OK"
