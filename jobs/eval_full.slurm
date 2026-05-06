#!/bin/bash
#SBATCH --job-name=eval_full
#SBATCH --output=logs/eval_full_%j.out
#SBATCH --error=logs/eval_full_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=coc-gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00

cd ~/scratch/eldenring_peft || exit 1
mkdir -p logs outputs/full/eval/samples

PYTHONPATH=src ~/.conda/envs/eval_env/bin/python scripts/eval.py \
  --method full \
  --weights_path outputs/full/weights \
  --test_jsonl test.jsonl \
  --data_dir combined_data \
  --output_file outputs/full/eval/results.json \
  --samples_dir outputs/full/eval/samples
