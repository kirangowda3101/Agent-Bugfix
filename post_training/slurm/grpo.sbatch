#!/bin/bash -l
#SBATCH -J mbpp-grpo
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH -o /scratch/ramanagarajayaram.k/logs/%x-%j.out
# All four models are trained and evaluated on H200s, so results share one GPU type.
# Full run (only after the GPU smoke test is reviewed). Safe to resubmit:
# training resumes from the newest checkpoint and finished evaluations are skipped.
source /scratch/ramanagarajayaram.k/agent-bugfix/post_training/slurm/common.sh
gpu_job_start

python train_grpo.py --config configs/grpo_full.yaml
python select_and_eval.py --name grpo --run runs/grpo
