#!/bin/bash -l
#SBATCH -J mbpp-smoke
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH -o /scratch/ramanagarajayaram.k/logs/%x-%j.out
# A100 smoke test: sandbox tests, 5 GRPO steps, 20-problem VALIDATION eval, summary.
#   sbatch post_training/slurm/smoke.sbatch
source /scratch/ramanagarajayaram.k/agent-bugfix/post_training/slurm/common.sh
gpu_job_start

rm -rf runs/grpo_a100_smoke runs/eval_a100_smoke_val20.json   # a smoke test always starts fresh
python train_grpo.py --config configs/grpo_a100_smoke.yaml
python evaluate.py --split validation --limit 20 --batch 32 --dtype bfloat16 \
    --out runs/eval_a100_smoke_val20.json
python slurm/summarize_smoke.py --out results/a100_smoke_summary.json
