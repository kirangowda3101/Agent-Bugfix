#!/bin/bash -l
#SBATCH -J mbpp-setup
#SBATCH -p short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH -o /scratch/ramanagarajayaram.k/logs/%x-%j.out
# CPU job for one-time setup steps that the login node's limits kill.
#   sbatch post_training/slurm/setup_cpu.sbatch env       # conda env + packages
#   sbatch post_training/slurm/setup_cpu.sbatch sandbox   # sandbox image (later step)
#   sbatch post_training/slurm/setup_cpu.sbatch download  # model + dataset (later step)
# Every step needs the internet, so the job first checks whether this compute
# node can reach each site, and stops before changing anything if one is unreachable.
source /scratch/ramanagarajayaram.k/agent-bugfix/post_training/slurm/common.sh
STEP=${1:?usage: sbatch setup_cpu.sbatch env|sandbox|download}
echo "job $SLURM_JOB_ID step '$STEP' on $(hostname) at $(date -Is)"
echo "proxy settings: http_proxy=${http_proxy:-unset} https_proxy=${https_proxy:-unset}"

# Any HTTP status means the site was reached (401 is normal for the Docker registry);
# 000 means no connection at all.
ok=1
for url in https://pypi.org/simple/pip/ \
           https://download.pytorch.org/whl/cu126/ \
           https://conda.anaconda.org/conda-forge/noarch/current_repodata.json \
           https://huggingface.co/api/models/Qwen/Qwen2.5-Coder-0.5B-Instruct \
           https://registry-1.docker.io/v2/; do
    code=$(curl -sS -m 20 -o /dev/null -w "%{http_code}" -I "$url" 2>/dev/null || true)
    echo "  ${code:-000}  $url"
    [ "${code:-000}" = "000" ] && ok=0
done
if [ $ok -ne 1 ]; then
    echo "NO INTERNET FROM COMPUTE NODE: nothing was changed. Use the offline setup instead."
    exit 2
fi
echo "INTERNET OK"

case "$STEP" in
    env)      bash "$PT/slurm/setup_env.sh" ;;
    sandbox)  bash "$PT/slurm/build_sandbox.sh" ;;
    download) bash "$PT/slurm/download.sh" ;;
    *)        echo "unknown step $STEP"; exit 1 ;;
esac
