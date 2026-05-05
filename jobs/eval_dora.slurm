#!/bin/bash
#SBATCH --job-name=eval_dora
#SBATCH --output=logs/eval_dora_%j.out
#SBATCH --error=logs/eval_dora_%j.err
#SBATCH --partition=coc-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00

cd ~/scratch/eldenring_peft || exit 1

PYTHONPATH=src ~/.conda/envs/eval_env/bin/python scripts/eval.py \
  --method dora \
  --weights_path outputs/dora/weights \
  --test_jsonl test.jsonl \
  --data_dir combined_data
