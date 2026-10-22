#!/bin/bash -l
#SBATCH -J mbpp-sft
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

python train_sft.py --config configs/sft_full.yaml
python select_and_eval.py --name sft --run runs/sft --merge-to runs/sft/best_merged
