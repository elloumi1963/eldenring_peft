#!/bin/bash
#SBATCH --job-name=dcft
#SBATCH --output=logs/dcft_%j.out
#SBATCH --error=logs/dcft_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=coc-gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00

cd ~/scratch/eldenring_peft || exit 1

mkdir -p logs
mkdir -p outputs/dcft/weights
mkdir -p outputs/dcft/plots
mkdir -p outputs/dcft/metrics

echo "PWD=$(pwd)"
echo "Starting DCFT run at $(date)"

PYTHONPATH=src ~/.conda/envs/eval_env/bin/python -u scripts/finetune_sd.py \
  --method dcft \
  --lr 5e-5 \
  --batch_size 2 \
  --grad_acc_steps 2 \
  --dcft_r 56 \
  --dcft_kernel_size 4 \
  --dcft_stride 2 \
  --dcft_dropout 0.1 \
  --epochs 10 \
  --weight_decay 0.01 \
  --eps 1e-08 \
  --output_dir outputs/dcft/weights \
  --plot_out outputs/dcft/plots/train.png \
  --metrics_out outputs/dcft/metrics/results.json
