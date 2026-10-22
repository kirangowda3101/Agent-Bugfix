#!/bin/bash -l
#SBATCH -J mbpp-sft-grpo
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH -o /scratch/ramanagarajayaram.k/logs/%x-%j.out
# Full run (only after the A100 smoke test is reviewed). Safe to resubmit:
# training resumes from the newest checkpoint and finished evaluations are skipped.
source /scratch/ramanagarajayaram.k/agent-bugfix/post_training/slurm/common.sh
gpu_job_start

test -f runs/sft/best_merged/config.json || { echo "runs/sft/best_merged missing: run sft.sbatch first"; exit 1; }
python train_grpo.py --config configs/sft_grpo_full.yaml
python select_and_eval.py --name sft_grpo --run runs/sft_grpo --weights runs/sft/best_merged
