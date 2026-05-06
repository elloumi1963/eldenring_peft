#!/bin/bash
#SBATCH --job-name=eval_adalora
#SBATCH --output=logs/eval_adalora_%j.out
#SBATCH --error=logs/eval_adalora_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=coc-gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00

cd ~/scratch/eldenring_peft || exit 1

PYTHONPATH=src ~/.conda/envs/eval_env/bin/python scripts/eval.py \
  --method adalora \
  --weights_path outputs/adalora/weights \
  --test_jsonl test.jsonl \
  --data_dir combined_data
