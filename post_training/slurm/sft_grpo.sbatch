#!/bin/bash -l
#SBATCH -J mbpp-sft-grpo
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH -o /scratch/ramanagarajayaram.k/logs/%x-%j.out
# All four models are trained and evaluated on H200s, so results share one GPU type.
# Full run (only after the GPU smoke test is reviewed). Safe to resubmit:
# training resumes from the newest checkpoint and finished evaluations are skipped.
source /scratch/ramanagarajayaram.k/agent-bugfix/post_training/slurm/common.sh
acquire_run_lock sft_grpo   # before anything else: a duplicate job exits here
gpu_job_start

test -f runs/sft/best_merged/config.json || { echo "runs/sft/best_merged missing: run sft.sbatch first"; exit 1; }
python train_grpo.py --config configs/sft_grpo_full.yaml
python select_and_eval.py --name sft_grpo --run runs/sft_grpo --weights runs/sft/best_merged
