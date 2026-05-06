#!/bin/bash
#SBATCH --job-name=adalora
#SBATCH --output=logs/adalora_%j.out
#SBATCH --error=logs/adalora_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=coc-gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00

cd ~/scratch/eldenring_peft || exit 1

PYTHONPATH=src ~/.conda/envs/eval_env/bin/python scripts/finetune_sd.py \
  --method adalora \
  --lr 0.0001 \
  --batch_size 4 \
  --grad_acc_steps 2 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --adalora_init_r 16 \
  --adalora_target_r 8 \
  --adalora_beta1 0.85 \
  --adalora_beta2 0.85 \
  --adalora_orth_reg_weight 0.5 \
  --epochs 10 \
  --weight_decay 0.01 \
  --eps 1e-08 \
  --output_dir outputs/adalora/weights \
  --plot_out outputs/adalora/plots/train.png \
  --metrics_out outputs/adalora/metrics/results.json
