#!/bin/bash
#SBATCH --job-name=eval_dcft
#SBATCH --output=logs/eval_dcft_%j.out
#SBATCH --error=logs/eval_dcft_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=coc-gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00

cd ~/scratch/eldenring_peft || exit 1

mkdir -p logs outputs/dcft/eval/samples

echo "PWD=$(pwd)"
echo "Starting DCFT eval at $(date)"

PYTHONPATH=src ~/.conda/envs/eval_env/bin/python -u scripts/eval.py \
  --method dcft \
  --weights_path outputs/dcft/weights \
  --test_jsonl test.jsonl \
  --data_dir combined_data \
  --output_file outputs/dcft/eval/results.json \
  --samples_dir outputs/dcft/eval/samples \
  --dcft_r 56 \
  --dcft_kernel_size 4 \
  --dcft_stride 2 \
  --dcft_dropout 0.1
