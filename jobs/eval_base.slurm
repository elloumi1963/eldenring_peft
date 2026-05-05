#!/bin/bash
#SBATCH --job-name=eval_base
#SBATCH --output=logs/eval_base_%j.out
#SBATCH --error=logs/eval_base_%j.err
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00

cd ~/scratch/eldenring_peft || exit 1

PYTHONPATH=src ~/.conda/envs/eval_env/bin/python scripts/eval.py \
  --method base \
  --test_jsonl test.jsonl \
  --data_dir combined_data
