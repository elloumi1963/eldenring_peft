# Elden Ring PEFT
### Parameter-Efficient Fine-Tuning for Stable Diffusion on Elden Ring Gameplay Images

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-orange)
![Diffusers](https://img.shields.io/badge/HuggingFace-Diffusers-yellow)
![PEFT](https://img.shields.io/badge/PEFT-LoRA%20%7C%20DoRA%20%7C%20AdaLoRA%20%7C%20DCFT-green)

This repository investigates **Parameter-Efficient Fine-Tuning (PEFT)** techniques for adapting **Stable Diffusion v1.5** to the visual style of **Elden Ring** gameplay.

Instead of updating the entire UNet (≈860M parameters), we compare lightweight adapter-based methods capable of achieving competitive image quality while training **less than 1%** of the model parameters.

The project was developed as part of the **CS7643 Deep Learning** course at **Georgia Tech**.

---

# Table of Contents

- [Overview](#overview)
- [Methods](#methods)
- [Results](#results)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Reproducibility](#reproducibility)
- [Key Findings](#key-findings)
- [Future Work](#future-work)

---

# Overview

The objective is to compare several PEFT methods for diffusion models and evaluate whether they can approach the performance of full fine-tuning while drastically reducing computational cost.

The following methods are implemented:

- Full UNet Fine-Tuning
- LoRA
- DoRA
- AdaLoRA
- DCFT

All methods are trained and evaluated using the exact same dataset, optimizer, training pipeline and evaluation metrics.

---

# Methods

The project is built on **Stable Diffusion v1.5**.

During training:

- CLIP Text Encoder is frozen
- VAE is frozen
- Only the UNet is adapted

For PEFT approaches, adapters are inserted into:

- Query projections (`to_q`)
- Key projections (`to_k`)
- Value projections (`to_v`)
- Output projections (`to_out.0`)
- Feed-forward layers

The following methods are compared:

| Method | Description |
|---------|-------------|
| Full Fine-Tuning | Updates every UNet parameter |
| LoRA | Low-Rank Adaptation |
| DoRA | Weight-Decomposed LoRA |
| AdaLoRA | Adaptive Rank Allocation |
| DCFT | Deconvolution-based Fine-Tuning |

---

# Results

| Method | Trainable Params | FID ↓ | KID ↓ | CLIPScore ↑ |
|---------|----------------:|------:|------:|------------:|
| Base SD 1.5 | 0 | 248.30 | 0.0431 | 36.26 |
| Full Fine-Tuning | 859M | **229.21** | **0.0240** | **39.26** |
| LoRA | 5.98M | 233.16 | 0.0470 | 37.86 |
| **DoRA** | **6.20M** | **233.01** | **0.0302** | 37.77 |
| AdaLoRA | 5.98M | 237.50 | 0.0297 | 37.23 |
| DCFT | 5.68M | 243.74 | 0.0371 | **38.03** |

## Main Findings

- Full fine-tuning achieves the highest image quality.
- DoRA provides the strongest overall PEFT performance.
- LoRA remains a strong baseline with very low computational cost.
- AdaLoRA improves KID but is less stable.
- DCFT improves texture quality but tends to overfit to HUD/UI elements.

---

# Project Structure

```text
.
├── jobs/
│   ├── train_lora.slurm
│   ├── train_dora.slurm
│   ├── train_adalora.slurm
│   ├── train_dcft.slurm
│   ├── train_full.slurm
│   ├── eval_*.slurm
│
├── scripts/
│   ├── process_data.py
│   ├── finetune_sd.py
│   └── eval.py
│
├── src/
│   └── eldenring_peft/
│       └── dataset.py
│
├── train.jsonl
├── val.jsonl
├── test.jsonl
├── requirements.txt
└── README.md
```

---

# Dataset

The dataset contains:

- **761** Elden Ring gameplay screenshots
- Automatically generated captions using GPT-4o-mini
- Images resized to **512×512**

Split:

| Split | Samples |
|--------|---------|
| Train | 608 |
| Validation | 76 |
| Test | 76 |

Captions begin with the trigger:

```text
eldenring gameplay style
```

---

# Installation

Create a Python environment:

```bash
conda create -n eldenring_peft python=3.10
conda activate eldenring_peft
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set the source directory:

```bash
export PYTHONPATH=src
```

---

# Training

Example: LoRA

```bash
PYTHONPATH=src python scripts/finetune_sd.py \
    --method lora \
    --data_dir combined_data \
    --train_jsonl train.jsonl \
    --val_jsonl val.jsonl
```

Replace `lora` by:

- dora
- adalora
- dcft
- full

to train the corresponding method.

SLURM scripts are also provided:

```bash
sbatch jobs/train_lora.slurm
sbatch jobs/train_dora.slurm
sbatch jobs/train_adalora.slurm
sbatch jobs/train_dcft.slurm
sbatch jobs/train_full.slurm
```

---

# Evaluation

Evaluation generates one image per test caption and computes:

- FID
- KID
- CLIPScore

Example:

```bash
PYTHONPATH=src python scripts/eval.py \
    --method lora \
    --weights_path outputs/lora/weights
```

The evaluation also generates qualitative samples for several custom prompts.

---

# Reproducibility

- Fixed train/validation/test split
- Seed = 42
- Deterministic evaluation seeds
- Mixed precision training
- Cosine learning rate scheduler

Some small variations between runs are expected due to CUDA and diffusion sampling nondeterminism.

---

# Key Findings

- PEFT methods recover most of the performance of full fine-tuning while updating less than **1%** of the model.
- DoRA offers the best trade-off between computational cost and image quality.
- The main limitation is the relatively small dataset (761 images), leading to overfitting and reproduction of HUD/UI artifacts.

---

# Future Work

- Larger gameplay datasets
- HUD/UI removal before training
- Stronger data augmentation
- Newer Stable Diffusion backbones
- Human evaluation alongside automatic metrics

---


