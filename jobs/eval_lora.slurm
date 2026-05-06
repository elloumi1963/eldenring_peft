#!/bin/bash
#SBATCH --job-name=eval_lora
#SBATCH --output=logs/eval_lora_%j.out
#SBATCH --error=logs/eval_lora_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=coc-gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00

cd ~/scratch/eldenring_peft || exit 1

PYTHONPATH=src ~/.conda/envs/eval_env/bin/python scripts/eval.py \
  --method lora \
  --weights_path outputs/lora/weights \
  --test_jsonl test.jsonl \
  --data_dir combined_data
