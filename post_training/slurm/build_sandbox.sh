#!/bin/bash
# Login node, once: build the sandbox image without root and without fakeroot.
#   bash post_training/slurm/build_sandbox.sh
#
# Building from a .def file unprivileged makes Apptainer imply --fakeroot, which
# needs /etc/subuid mappings that a cluster account may not have. Instead:
#   1. pull the pinned base image into a directory (docker:// source: no fakeroot)
#   2. copy the harness files in
#   3. pack the directory into a SIF
# Base: python:3.11-slim index digest (Python 3.11.17, Debian 13, libseccomp 2.6.0).
# Apptainer rejects tag+digest references, so the digest is used alone.
#
# No model code runs here. The only check is that the image starts and has
# Python and libseccomp. If the packed SIF cannot run (FUSE problems), the
# directory image is used instead; both are mounted read-only at run time.
source "$(dirname "$0")/common.sh"

BASE=docker://python@sha256:cce12316475d7d56d605f76e1d811d0dc4e7227f2d4afd29951521264f504ff8
SIF_DIR=$(dirname "$SANDBOX_SIF")
IMAGE_DIR=$SIF_DIR/mbpp-sandbox.dir
mkdir -p "$SIF_DIR" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

rm -rf "$IMAGE_DIR"
apptainer build --sandbox "$IMAGE_DIR" "$BASE"
mkdir -p "$IMAGE_DIR/harness"
cp "$PT/sandbox/harness.py" "$PT/sandbox/seccomp_filter.py" "$PT/sandbox/seccomp-default.json" "$IMAGE_DIR/harness/"
chmod 0444 "$IMAGE_DIR"/harness/*
apptainer build --force "$SANDBOX_SIF" "$IMAGE_DIR"

check() {
    apptainer exec --containall --no-home --no-mount home,cwd,bind-paths,hostfs --net --network none --pwd /tmp "$1" \
        python -c "import ctypes.util, os, sys; print(sys.version.split()[0], ctypes.util.find_library('seccomp'), sorted(os.listdir('/harness')))"
}
if check "$SANDBOX_SIF"; then
    echo "$SANDBOX_SIF" > "$SIF_DIR/image_path"
    sha256sum "$SANDBOX_SIF" | tee "$SANDBOX_SIF.sha256"
elif check "$IMAGE_DIR"; then
    echo "WARNING: the packed SIF does not run here; using the directory image instead."
    echo "$IMAGE_DIR" > "$SIF_DIR/image_path"
else
    echo "SANDBOX IMAGE FAILED: neither the SIF nor the directory image runs."
    exit 1
fi
echo "sandbox image: $(cat "$SIF_DIR/image_path")"
echo "SANDBOX IMAGE OK"
