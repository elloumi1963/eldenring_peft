#!/bin/bash
#SBATCH --job-name=lora
#SBATCH --output=logs/lora_%j.out
#SBATCH --error=logs/lora_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=coc-gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00

cd ~/scratch/eldenring_peft || exit 1

PYTHONPATH=src ~/.conda/envs/eval_env/bin/python scripts/finetune_sd.py \
  --method lora \
  --lr 0.0001 \
  --batch_size 4 \
  --grad_acc_steps 2 \
  --lora_r 16 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --epochs 10 \
  --weight_decay 0.01 \
  --eps 1e-08 \
  --output_dir outputs/lora/weights \
  --plot_out outputs/lora/plots/train.png \
  --metrics_out outputs/lora/metrics/results.json
