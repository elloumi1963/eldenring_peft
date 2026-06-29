# Elden Ring PEFT: Efficient Stable Diffusion Fine-Tuning

!\[Python](https://img.shields.io/badge/Python-3.10%2B-blue)
!\[PyTorch](https://img.shields.io/badge/PyTorch-2.6-orange)
!\[Diffusers](https://img.shields.io/badge/HuggingFace-Diffusers-yellow)
!\[PEFT](https://img.shields.io/badge/PEFT-LoRA%20%7C%20DoRA%20%7C%20AdaLoRA%20%7C%20DCFT-green)

This repository compares **parameter-efficient fine-tuning (PEFT)** methods for adapting **Stable Diffusion v1.5** to an **Elden Ring gameplay-style text-to-image generation task**.

The project evaluates whether lightweight adapter-based methods can approach full UNet fine-tuning quality while updating less than 1% of the model parameters. The implemented methods are:

* **Full UNet fine-tuning** as an upper-bound baseline
* **LoRA**: low-rank adaptation of attention and feed-forward projections
* **DoRA**: weight-decomposed LoRA with learnable magnitude updates
* **AdaLoRA**: adaptive rank allocation under a fixed parameter budget
* **DCFT**: deconvolution-based low-rank subspace adaptation

The main finding is that **DoRA provides the best overall PEFT trade-off**, achieving the strongest PEFT FID and KID under a similar parameter budget to LoRA, while all PEFT methods remain dramatically cheaper than full fine-tuning.

\---

## Results Summary

|Method|Trainable Params|Adapter %|FID ↓|KID Mean ↓|CLIPScore ↑|
|-|-:|-:|-:|-:|-:|
|Base SD 1.5|0|0.00%|248.2981|0.0431|36.2623|
|Full UNet FT|859,520,964|100.00%|**229.2130**|**0.0240**|**39.2600**|
|LoRA|5,984,256|0.69%|233.1600|0.0470|37.8600|
|DoRA|6,196,416|0.72%|**233.0123**|0.0302|37.7681|
|AdaLoRA|5,984,416|0.69%|237.5016|0.0297|37.2290|
|DCFT|5,681,376|0.66%|243.7423|0.0371|**38.0344**|

**Interpretation:**

* **Full fine-tuning** reaches the best absolute quality, but requires updating the entire UNet.
* **DoRA** is the strongest PEFT method overall, improving over LoRA in both FID and KID with only a small increase in trainable parameters.
* **LoRA** is competitive and simple, but shows more texture degradation and UI overfitting.
* **AdaLoRA** improves KID but is less stable and underperforms on FID and CLIPScore.
* **DCFT** improves local texture and prompt alignment, but hurts global image structure and overfits to HUD/UI artifacts.

\---

## Project Structure

```text
.
├── scripts/
│   ├── process\_data.py      # Collect images, generate GPT captions, split JSONL files
│   ├── finetune\_sd.py       # Train LoRA, DoRA, AdaLoRA, DCFT, or full UNet fine-tuning
│   └── eval.py              # Generate samples and compute FID, KID, and CLIPScore
├── src/
│   └── eldenring\_peft/
│       └── dataset.py       # PyTorch dataset for image-caption pairs
├── jobs/
│   ├── train\_lora.slurm
│   ├── train\_dora.slurm
│   ├── train\_adalora.slurm
│   ├── train\_dcft.slurm
│   ├── train\_full.slrum     # Full fine-tuning SLURM script; consider renaming to .slurm
│   ├── eval\_\*.slurm
├── train.jsonl              # Training split
├── val.jsonl                # Validation split
├── test.jsonl               # Test split
├── requirements.txt
└── README.md
```

\---

## Dataset

The dataset contains **761 paired RGB gameplay screenshots and captions** extracted from approximately 30 minutes of Elden Ring gameplay.

The data is split with a fixed random seed into:

* **608 training samples**
* **76 validation samples**
* **76 test samples**

Captions are generated using GPT-4o-mini and consistently start with the trigger phrase:

```text
eldenring gameplay style
```

Each caption describes the main subject, action, camera angle, environment, lighting, weather, and color palette. Images are resized to **512×512**, randomly horizontally flipped during training, converted to tensors, and normalized to **\[-1, 1]** for the Stable Diffusion VAE.

Expected data layout:

```text
combined\_data/
├── 0000.png
├── 0001.png
├── ...
└── 0760.png

train.jsonl
val.jsonl
test.jsonl
```

Each JSONL row should follow:

```json
{"file\_name": "0000.png", "text": "eldenring gameplay style, ..."}
```

\---

## Methodology

The project uses **Stable Diffusion v1.5** as the base model. The **VAE** and **CLIP text encoder** are frozen, and training is restricted to the **UNet**.

The training objective is the standard latent diffusion denoising loss:

```text
L = E\[ || epsilon - epsilon\_theta(z\_t, t, text\_condition) ||^2 ]
```

For PEFT methods, adapter modules are injected into the UNet target layers:

```python
DEFAULT\_TARGET\_MODULES = \[
    "to\_q",
    "to\_k",
    "to\_v",
    "to\_out.0",
    "ff.net.0.proj",
    "ff.net.2",
]
```

This means the experiments adapt cross-attention projections and selected feed-forward layers while keeping the pretrained backbone mostly frozen.

\---

## Installation

Create and activate a Python environment:

```bash
conda create -n eldenring\_peft python=3.10 -y
conda activate eldenring\_peft
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For local imports, run commands with:

```bash
export PYTHONPATH=src
```

or prefix commands with `PYTHONPATH=src`.

You also need access to the Stable Diffusion v1.5 checkpoint used by Diffusers:

```text
sd-legacy/stable-diffusion-v1-5
```

\---

## Data Processing

### 1\. Collect PNG frames

```bash
PYTHONPATH=src python scripts/process\_data.py collect \\
  --src /path/to/raw/gameplay\_frames \\
  --dest combined\_data
```

This recursively finds `.png` files, copies them into a single folder, and renames them sequentially.

### 2\. Generate captions

Set your OpenAI API key first:

```bash
export OPENAI\_API\_KEY="your\_api\_key"
```

Then run:

```bash
PYTHONPATH=src python scripts/process\_data.py caption \\
  --dir combined\_data \\
  --out labels.jsonl
```

The captioning script supports resume behavior by skipping images already present in the output JSONL file.

### 3\. Split into train/validation/test

```bash
PYTHONPATH=src python scripts/process\_data.py split \\
  --input labels.jsonl \\
  --train\_ratio 0.8 \\
  --val\_ratio 0.1
```

This produces `train.jsonl`, `val.jsonl`, and `test.jsonl` using a fixed seed for reproducibility.

\---

## Training

All training is handled by `scripts/finetune\_sd.py`.

Common configuration:

* Optimizer: **AdamW**
* Learning rate: **1e-4** for PEFT methods
* Weight decay: **0.01**
* Batch size: **4**
* Gradient accumulation: **2**
* Effective batch size: **8**
* Epochs: **10**
* Scheduler: cosine schedule with warmup
* Loss: MSE noise prediction loss
* Mixed precision: enabled on CUDA

### Train LoRA

```bash
PYTHONPATH=src python scripts/finetune\_sd.py \\
  --method lora \\
  --data\_dir combined\_data \\
  --train\_jsonl train.jsonl \\
  --val\_jsonl val.jsonl \\
  --lr 1e-4 \\
  --batch\_size 4 \\
  --grad\_acc\_steps 2 \\
  --lora\_r 16 \\
  --lora\_alpha 16 \\
  --lora\_dropout 0.05 \\
  --epochs 10 \\
  --output\_dir outputs/lora/weights \\
  --plot\_out outputs/lora/plots/train.png \\
  --metrics\_out outputs/lora/metrics/results.json
```

### Train DoRA

```bash
PYTHONPATH=src python scripts/finetune\_sd.py \\
  --method dora \\
  --data\_dir combined\_data \\
  --train\_jsonl train.jsonl \\
  --val\_jsonl val.jsonl \\
  --lr 1e-4 \\
  --batch\_size 4 \\
  --grad\_acc\_steps 2 \\
  --lora\_r 16 \\
  --lora\_alpha 16 \\
  --lora\_dropout 0.05 \\
  --epochs 10 \\
  --output\_dir outputs/dora/weights \\
  --plot\_out outputs/dora/plots/train.png \\
  --metrics\_out outputs/dora/metrics/results.json
```

### Train AdaLoRA

```bash
PYTHONPATH=src python scripts/finetune\_sd.py \\
  --method adalora \\
  --data\_dir combined\_data \\
  --train\_jsonl train.jsonl \\
  --val\_jsonl val.jsonl \\
  --lr 1e-4 \\
  --batch\_size 4 \\
  --grad\_acc\_steps 2 \\
  --lora\_alpha 16 \\
  --lora\_dropout 0.05 \\
  --adalora\_init\_r 16 \\
  --adalora\_target\_r 8 \\
  --adalora\_beta1 0.85 \\
  --adalora\_beta2 0.85 \\
  --adalora\_orth\_reg\_weight 0.5 \\
  --epochs 10 \\
  --output\_dir outputs/adalora/weights \\
  --plot\_out outputs/adalora/plots/train.png \\
  --metrics\_out outputs/adalora/metrics/results.json
```

### Train DCFT

```bash
PYTHONPATH=src python scripts/finetune\_sd.py \\
  --method dcft \\
  --data\_dir combined\_data \\
  --train\_jsonl train.jsonl \\
  --val\_jsonl val.jsonl \\
  --lr 5e-5 \\
  --batch\_size 2 \\
  --grad\_acc\_steps 2 \\
  --dcft\_r 56 \\
  --dcft\_kernel\_size 4 \\
  --dcft\_stride 2 \\
  --dcft\_dropout 0.1 \\
  --epochs 10 \\
  --output\_dir outputs/dcft/weights \\
  --plot\_out outputs/dcft/plots/train.png \\
  --metrics\_out outputs/dcft/metrics/results.json
```

### Train full UNet baseline

```bash
PYTHONPATH=src python scripts/finetune\_sd.py \\
  --method full \\
  --data\_dir combined\_data \\
  --train\_jsonl train.jsonl \\
  --val\_jsonl val.jsonl \\
  --lr 1e-5 \\
  --batch\_size 1 \\
  --grad\_acc\_steps 4 \\
  --epochs 10 \\
  --output\_dir outputs/full/weights \\
  --plot\_out outputs/full/plots/train.png \\
  --metrics\_out outputs/full/metrics/results.json
```

\---

## Evaluation

Evaluation generates one image per held-out test prompt, compares generated images to real images, and reports:

* **FID**: distributional visual similarity; lower is better
* **KID**: kernel-based distributional similarity; lower is better
* **CLIPScore**: image-text alignment; higher is better

### Evaluate base model

```bash
PYTHONPATH=src python scripts/eval.py \\
  --method base \\
  --test\_jsonl test.jsonl \\
  --data\_dir combined\_data \\
  --output\_file outputs/base/eval/results.json \\
  --samples\_dir outputs/base/eval/samples
```

### Evaluate LoRA / DoRA / AdaLoRA

```bash
PYTHONPATH=src python scripts/eval.py \\
  --method lora \\
  --weights\_path outputs/lora/weights \\
  --test\_jsonl test.jsonl \\
  --data\_dir combined\_data \\
  --output\_file outputs/lora/eval/results.json \\
  --samples\_dir outputs/lora/eval/samples
```

Replace `lora` and `outputs/lora/weights` with `dora` or `adalora` as needed.

### Evaluate DCFT

```bash
PYTHONPATH=src python scripts/eval.py \\
  --method dcft \\
  --weights\_path outputs/dcft/weights \\
  --test\_jsonl test.jsonl \\
  --data\_dir combined\_data \\
  --output\_file outputs/dcft/eval/results.json \\
  --samples\_dir outputs/dcft/eval/samples \\
  --dcft\_r 56 \\
  --dcft\_kernel\_size 4 \\
  --dcft\_stride 2 \\
  --dcft\_dropout 0.1
```

### Evaluate full UNet fine-tuning

```bash
PYTHONPATH=src python scripts/eval.py \\
  --method full \\
  --weights\_path outputs/full/weights \\
  --test\_jsonl test.jsonl \\
  --data\_dir combined\_data \\
  --output\_file outputs/full/eval/results.json \\
  --samples\_dir outputs/full/eval/samples
```

\---

## SLURM Usage

SLURM scripts are provided for Georgia Tech PACE-style GPU execution:

```bash
sbatch jobs/train\_lora.slurm
sbatch jobs/train\_dora.slurm
sbatch jobs/train\_adalora.slurm
sbatch jobs/train\_dcft.slurm
sbatch jobs/train\_full.slrum

sbatch jobs/eval\_lora.slurm
sbatch jobs/eval\_dora.slurm
sbatch jobs/eval\_adalora.slurm
sbatch jobs/eval\_dcft.slurm
sbatch jobs/eval\_full.slurm
```

Recommended cleanup before use:

```bash
mv jobs/train\_full.slrum jobs/train\_full.slurm
```

\---

## Qualitative Prompts

The evaluation script also generates images from three fixed custom prompts:

```text
eldenring gameplay style, third person view, player character fighting a massive dragon boss, dynamic motion, sparks, fire, cinematic camera angle
```

```text
eldenring gameplay style, boss arena, grotesque humanoid monster, detailed armor, high contrast lighting, epic composition
```

```text
eldenring gameplay style, open world exploration, distant castle, broken bridges, dead trees, misty environment, wide shot
```

These prompts are useful for visually comparing prompt adherence, local texture quality, global structure, and overfitting to gameplay UI elements.

\---

## Key Takeaways

* PEFT methods can recover much of the benefit of full Stable Diffusion fine-tuning while updating less than 1% of UNet parameters.
* DoRA is the strongest PEFT method in this project, especially for visual quality and distributional similarity.
* AdaLoRA is not consistently beneficial in this diffusion setting, likely because dynamic rank allocation introduces optimization instability across denoising timesteps.
* DCFT improves local texture and prompt alignment but tends to overfit, especially to HUD/UI artifacts.
* The biggest limitation is not adapter capacity, but the small and biased dataset.

\---

## Reproducibility Notes

* The data split uses a fixed seed of `42`.
* Training supports a `--seed` argument and an optional `--deterministic` flag.
* Evaluation uses deterministic prompt-level seeds derived from the base seed.
* Some variation is still expected because diffusion sampling, CUDA kernels, mixed precision, and hardware-level nondeterminism can affect results.

\---

## Limitations

* The dataset is small: only 761 image-caption pairs.
* Many screenshots contain HUD/UI artifacts, which the models learn and reproduce.
* The validation curves show rapid overfitting after early epochs.
* FID, KID, and CLIPScore do not fully capture perceptual realism or gameplay-style fidelity.
* Experiments use a fixed Stable Diffusion v1.5 backbone and a fixed set of target modules.

\---

## Future Work

* Remove or mask HUD/UI elements from training images.
* Add stronger data augmentation and regularization.
* Compare different target module selections, including attention-only vs attention-plus-FFN adapters.
* Test larger datasets and newer diffusion backbones.
* Add human preference evaluation for qualitative realism.

\---

## 

