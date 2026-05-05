import argparse
import json
import math
import os
import random
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import AdaLoraConfig, LoraConfig, get_peft_model
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, get_cosine_schedule_with_warmup

from eldenring_peft.dataset import EldenRingDataset


DEFAULT_TARGET_MODULES = [
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
]


def parse_target_modules(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Stable Diffusion v1.5 with LoRA, DoRA, AdaLoRA, or full UNet fine-tuning."
    )

    # Method
    parser.add_argument(
        "--method",
        type=str,
        default="lora",
        choices=["lora", "dora", "adalora", "full"],
        help="Fine-tuning method: lora, dora, adalora, or full.",
    )

    # Data
    parser.add_argument("--data_dir", type=str, default="combined_data")
    parser.add_argument("--train_jsonl", type=str, default="train.jsonl")
    parser.add_argument("--val_jsonl", type=str, default="val.jsonl")
    parser.add_argument("--num_workers", type=int, default=4)

    # Base model
    parser.add_argument("--model_id", type=str, default="sd-legacy/stable-diffusion-v1-5")

    # Training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_acc_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--epochs", "--epoch", dest="epochs", type=int, default=10)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="More deterministic training. Slower, but reduces run-to-run variance.",
    )

    # Scheduler
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=None,
        help="If omitted, uses max(10, int(total_steps * warmup_ratio)).",
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.05)

    # LoRA / DoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        type=str,
        default=",".join(DEFAULT_TARGET_MODULES),
        help="Comma-separated target modules for PEFT methods.",
    )

    # AdaLoRA
    parser.add_argument("--adalora_init_r", type=int, default=16)
    parser.add_argument("--adalora_target_r", type=int, default=8)
    parser.add_argument("--adalora_beta1", type=float, default=0.85)
    parser.add_argument("--adalora_beta2", type=float, default=0.85)
    parser.add_argument("--adalora_orth_reg_weight", type=float, default=0.5)
    parser.add_argument(
        "--adalora_tinit",
        type=int,
        default=None,
        help="AdaLoRA initial warmup steps before rank allocation starts. Auto if omitted.",
    )
    parser.add_argument(
        "--adalora_tfinal",
        type=int,
        default=None,
        help="AdaLoRA final fine-tuning steps after rank allocation ends. Auto if omitted.",
    )
    parser.add_argument(
        "--adalora_deltaT",
        type=int,
        default=None,
        help="AdaLoRA allocation update interval. Auto if omitted.",
    )

    # Output
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--plot_out", type=str, default=None)
    parser.add_argument("--metrics_out", type=str, default=None)
    parser.add_argument(
        "--save_final",
        action="store_true",
        help="Also save final checkpoint in addition to best checkpoint.",
    )

    return parser.parse_args()


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device_and_amp() -> Tuple[str, bool, torch.dtype]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    dtype = torch.float16 if device == "cuda" else torch.float32
    return device, use_amp, dtype


def compute_total_steps(num_batches: int, grad_acc_steps: int, epochs: int) -> int:
    # Correct formula: use ceil so the last partial accumulation step is counted.
    return math.ceil(num_batches / grad_acc_steps) * epochs


def compute_warmup_steps(total_steps: int, warmup_steps: Optional[int], warmup_ratio: float) -> int:
    if warmup_steps is not None:
        return min(warmup_steps, max(0, total_steps - 1))

    auto_warmup = int(total_steps * warmup_ratio)
    auto_warmup = max(10, auto_warmup)
    return min(auto_warmup, max(0, total_steps - 1))


def compute_adalora_schedule(
    total_steps: int,
    tinit: Optional[int],
    tfinal: Optional[int],
    deltaT: Optional[int],
) -> Tuple[int, int, int]:
    """
    AdaLoRA needs a valid budgeting phase:
        tinit + tfinal < total_steps

    This auto-schedule keeps it safe for short runs.
    """
    if total_steps < 10:
        auto_tinit = 1
        auto_tfinal = 1
        auto_deltaT = 1
    else:
        auto_tinit = max(1, int(total_steps * 0.10))
        auto_tfinal = max(1, int(total_steps * 0.10))
        auto_deltaT = max(1, int(total_steps * 0.02))

    tinit = auto_tinit if tinit is None else tinit
    tfinal = auto_tfinal if tfinal is None else tfinal
    deltaT = auto_deltaT if deltaT is None else deltaT

    if tinit + tfinal >= total_steps:
        # Leave at least one step for budget allocation.
        safe_side = max(1, (total_steps - 1) // 3)
        tinit = safe_side
        tfinal = safe_side
        if tinit + tfinal >= total_steps:
            tinit = 1
            tfinal = 1

    deltaT = max(1, min(deltaT, max(1, total_steps - tinit - tfinal)))
    return tinit, tfinal, deltaT


def setup_dataloaders(args, tokenizer):
    train_dataset = EldenRingDataset(args.data_dir, args.train_jsonl, tokenizer)
    val_dataset = EldenRingDataset(args.data_dir, args.val_jsonl, tokenizer)

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_dataset, val_dataset, train_dataloader, val_dataloader


def setup_model(args, device: str, dtype: torch.dtype, total_steps: int):
    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder").to(device, dtype=dtype)
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae").to(device, dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet").to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    vae.eval()

    target_modules = parse_target_modules(args.target_modules)

    if args.method == "full":
        # Full UNet fine-tuning. VAE and text encoder remain frozen.
        unet.requires_grad_(True)
        peft_config = None
        print("Method: full UNet fine-tuning")

    else:
        unet.requires_grad_(False)

        if args.method == "lora":
            peft_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=target_modules,
                lora_dropout=args.lora_dropout,
            )
            print("Method: LoRA")

        elif args.method == "dora":
            peft_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=target_modules,
                lora_dropout=args.lora_dropout,
                use_dora=True,
            )
            print("Method: DoRA")

        elif args.method == "adalora":
            tinit, tfinal, deltaT = compute_adalora_schedule(
                total_steps,
                args.adalora_tinit,
                args.adalora_tfinal,
                args.adalora_deltaT,
            )
            peft_config = AdaLoraConfig(
                init_r=args.adalora_init_r,
                target_r=args.adalora_target_r,
                total_step=total_steps,
                tinit=tinit,
                tfinal=tfinal,
                deltaT=deltaT,
                beta1=args.adalora_beta1,
                beta2=args.adalora_beta2,
                orth_reg_weight=args.adalora_orth_reg_weight,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=target_modules,
            )
            print("Method: AdaLoRA")
            print(f"AdaLoRA schedule: tinit={tinit}, tfinal={tfinal}, deltaT={deltaT}, total_step={total_steps}")

        else:
            raise ValueError(f"Unsupported method: {args.method}")

        unet = get_peft_model(unet, peft_config)
        unet.print_trainable_parameters()

    return tokenizer, text_encoder, vae, unet, noise_scheduler, peft_config


def save_checkpoint(unet, output_dir: str, method: str):
    os.makedirs(output_dir, exist_ok=True)

    if method == "adalora":
        # AdaLoRA can crash with safetensors in some PEFT versions.
        unet.save_pretrained(output_dir, safe_serialization=False)
    else:
        unet.save_pretrained(output_dir)


def train_one_epoch(
    unet,
    vae,
    text_encoder,
    noise_scheduler,
    train_loader,
    optimizer,
    lr_scheduler,
    scaler,
    device,
    weight_dtype,
    grad_acc_steps,
    global_step,
    training_method="lora",
):
    unet.train()
    total_loss = 0.0

    progress_bar = tqdm(train_loader, desc=f"Training epoch", leave=False)

    for step, batch in enumerate(progress_bar):

        # -----------------------------------
        # Encode images to latent space
        # -----------------------------------
        with torch.no_grad():
            latents = vae.encode(
                batch["pixel_values"].to(device, dtype=weight_dtype)
            ).latent_dist.sample()
            latents = latents * 0.18215

            encoder_hidden_states = text_encoder(
                batch["input_ids"].to(device)
            )[0]

        # -----------------------------------
        # Sample noise and timesteps
        # -----------------------------------
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=device
        ).long()

        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # -----------------------------------
        # Forward + loss
        # -----------------------------------
        with autocast(enabled=(device == "cuda")):
            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states
            ).sample

            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            loss = loss / grad_acc_steps

        # -----------------------------------
        # Backward
        # -----------------------------------
        scaler.scale(loss).backward()

        # -----------------------------------
        # Optimizer step
        # -----------------------------------
        if (step + 1) % grad_acc_steps == 0:

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()

            global_step += 1

            # -----------------------------------
            # AdaLoRA rank reallocation ONLY
            # -----------------------------------
            if training_method == "adalora":
                if hasattr(unet, "peft_config") and hasattr(unet, "update_and_allocate"):
                    adalora_tinit = getattr(unet.peft_config["default"], "tinit", 0)

                    if global_step >= adalora_tinit:

                        # protect against None gradients
                        for p in unet.parameters():
                            if p.requires_grad and p.grad is None:
                                p.grad = torch.zeros_like(p)

                        unet.update_and_allocate(global_step)

        total_loss += loss.item() * grad_acc_steps

        current_lr = optimizer.param_groups[0]["lr"]

        progress_bar.set_postfix({
            "loss": f"{loss.item() * grad_acc_steps:.4f}",
            "lr": f"{current_lr:.2e}",
            "step": global_step
        })

    avg_loss = total_loss / len(train_loader)

    return avg_loss, global_step


@torch.no_grad()
def validate(unet, vae, text_encoder, noise_scheduler, val_dataloader, device: str, use_amp: bool, dtype: torch.dtype):
    unet.eval()
    total_val_loss = 0.0
    val_steps = 0

    with autocast(device_type=device, enabled=use_amp):
        for batch in tqdm(val_dataloader, desc="Validation"):
            pixel_values = batch["pixel_values"].to(device, dtype=dtype)
            input_ids = batch["input_ids"].to(device)

            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device,
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            encoder_hidden_states = text_encoder(input_ids)[0]
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            val_loss = F.mse_loss(noise_pred.float(), noise.float())

            total_val_loss += val_loss.item()
            val_steps += 1

    return total_val_loss / max(1, val_steps)


def plot_results(epoch_train_losses, epoch_val_losses, step_losses, learning_rates, plot_out: str):
    num_epochs = len(epoch_train_losses)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    ax1.plot(range(1, num_epochs + 1), epoch_train_losses, label="Training Loss", marker="o", linewidth=2)
    ax1.plot(range(1, num_epochs + 1), epoch_val_losses, label="Validation Loss", marker="o", linewidth=2)
    ax1.set_title("Training and Validation Loss over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if step_losses:
        ax2.plot(step_losses, linewidth=1, alpha=0.7)
        ax2.set_title("Step-wise Training Loss")
        ax2.set_xlabel("Optimizer Steps")
        ax2.set_ylabel("MSE Loss")
        ax2.grid(True, alpha=0.3)

    if learning_rates:
        ax3.plot(learning_rates, linewidth=2)
        ax3.set_title("Learning Rate Schedule")
        ax3.set_xlabel("Optimizer Steps")
        ax3.set_ylabel("Learning Rate")
        ax3.set_yscale("log")
        ax3.grid(True, alpha=0.3)

    if len(step_losses) > 100:
        window_size = max(1, len(step_losses) // 50)
        smoothed = np.convolve(step_losses, np.ones(window_size) / window_size, mode="same")
        ax4.plot(smoothed, linewidth=2)
        ax4.set_title("Smoothed Training Loss Trend")
        ax4.set_xlabel("Optimizer Steps")
        ax4.set_ylabel("MSE Loss")
        ax4.grid(True, alpha=0.3)
    elif len(step_losses) > 50:
        ax4.hist(step_losses[25:], bins=20, alpha=0.7, edgecolor="black")
        ax4.set_title("Training Loss Distribution")
        ax4.set_xlabel("MSE Loss")
        ax4.set_ylabel("Frequency")
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(plot_out), exist_ok=True)
    plt.savefig(plot_out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_output_paths(args):
    output_dir = args.output_dir or f"elden_ring_{args.method}_weights"
    plot_out = args.plot_out or f"{args.method}_train_results.png"
    metrics_out = args.metrics_out or f"{args.method}_training_results.json"
    return output_dir, plot_out, metrics_out


def main():
    args = parse_args()
    set_seed(args.seed, args.deterministic)

    device, use_amp, dtype = get_device_and_amp()
    output_dir, plot_out, metrics_out = build_output_paths(args)

    print(f"Running on device: {device}")
    print(f"AMP enabled: {use_amp}")

    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    train_dataset, val_dataset, train_dataloader, val_dataloader = setup_dataloaders(args, tokenizer)

    total_steps = compute_total_steps(len(train_dataloader), args.grad_acc_steps, args.epochs)
    warmup_steps = compute_warmup_steps(total_steps, args.warmup_steps, args.warmup_ratio)

    # Load tokenizer again inside setup_model for clean structure.
    tokenizer, text_encoder, vae, unet, noise_scheduler, peft_config = setup_model(args, device, dtype, total_steps)

    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found. Check method/config.")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=args.eps,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    epoch_train_losses = []
    epoch_val_losses = []
    step_losses = []
    learning_rates = []

    global_step = 0
    best_val_loss = float("inf")
    best_epoch = None

    print(f"Training method: {args.method}")
    print(f"Training for {args.epochs} epochs")
    print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}")
    print(f"Batches per epoch: Train={len(train_dataloader)}, Val={len(val_dataloader)}")
    print(f"Batch size: {args.batch_size}, Gradient accumulation: {args.grad_acc_steps}")
    print(f"Effective batch size: {args.batch_size * args.grad_acc_steps}")
    print(f"Total optimizer steps: {total_steps}")
    print(f"Learning rate: {args.lr}, Warmup steps: {warmup_steps}")
    print(f"Output dir: {output_dir}")

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")

        avg_train_loss, global_step = train_one_epoch(
            epoch=epoch,
            num_epochs=args.epochs,
            unet=unet,
            vae=vae,
            text_encoder=text_encoder,
            noise_scheduler=noise_scheduler,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            trainable_params=trainable_params,
            args=args,
            device=device,
            use_amp=use_amp,
            dtype=dtype,
            global_step=global_step,
            step_losses=step_losses,
            learning_rates=learning_rates,
        )

        epoch_train_losses.append(avg_train_loss)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        avg_val_loss = validate(
            unet=unet,
            vae=vae,
            text_encoder=text_encoder,
            noise_scheduler=noise_scheduler,
            val_dataloader=val_dataloader,
            device=device,
            use_amp=use_amp,
            dtype=dtype,
        )

        epoch_val_losses.append(avg_val_loss)
        current_lr = scheduler.get_last_lr()[0]

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            save_checkpoint(unet, output_dir, args.method)
            print(f"✓ New best model saved at epoch {epoch + 1} (val_loss={avg_val_loss:.4f})")

        print(f"Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.2e}")

    if args.save_final:
        final_dir = f"{output_dir}_final"
        save_checkpoint(unet, final_dir, args.method)
        print(f"Final checkpoint saved to {final_dir}")

    print(f"Training complete. Best checkpoint saved to {output_dir}.")

    plot_results(epoch_train_losses, epoch_val_losses, step_losses, learning_rates, plot_out)
    print(f"Training plot saved to {plot_out}")

    hyperparameters: Dict[str, object] = {
        "method": args.method,
        "model_id": args.model_id,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_acc_steps,
        "effective_batch_size": args.batch_size * args.grad_acc_steps,
        "num_epochs": args.epochs,
        "warmup_steps": warmup_steps,
        "warmup_ratio": args.warmup_ratio,
        "total_steps": total_steps,
        "actual_global_steps": global_step,
        "weight_decay": args.weight_decay,
        "eps": args.eps,
        "max_grad_norm": args.max_grad_norm,
        "seed": args.seed,
        "target_modules": parse_target_modules(args.target_modules),
    }

    if args.method in ["lora", "dora"]:
        hyperparameters.update(
            {
                "lora_rank": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "use_dora": args.method == "dora",
            }
        )
    elif args.method == "adalora":
        tinit, tfinal, deltaT = compute_adalora_schedule(
            total_steps,
            args.adalora_tinit,
            args.adalora_tfinal,
            args.adalora_deltaT,
        )
        hyperparameters.update(
            {
                "adalora_init_r": args.adalora_init_r,
                "adalora_target_r": args.adalora_target_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "adalora_tinit": tinit,
                "adalora_tfinal": tfinal,
                "adalora_deltaT": deltaT,
                "adalora_beta1": args.adalora_beta1,
                "adalora_beta2": args.adalora_beta2,
                "adalora_orth_reg_weight": args.adalora_orth_reg_weight,
            }
        )

    training_metrics = {
        "epoch_train_losses": epoch_train_losses,
        "epoch_val_losses": epoch_val_losses,
        "step_losses": step_losses,
        "learning_rates": learning_rates,
        "final_metrics": {
            "final_train_loss": epoch_train_losses[-1] if epoch_train_losses else None,
            "final_val_loss": epoch_val_losses[-1] if epoch_val_losses else None,
            "best_train_loss": min(epoch_train_losses) if epoch_train_losses else None,
            "best_val_loss": best_val_loss if epoch_val_losses else None,
            "best_epoch": best_epoch,
        },
        "hyperparameters": hyperparameters,
    }
    
    os.makedirs(os.path.dirname(metrics_out), exist_ok=True)

    with open(metrics_out, "w") as f:
        json.dump(training_metrics, f, indent=2)

    print(f"Training metrics saved to {metrics_out}")

    if epoch_train_losses and epoch_val_losses:
        print("\n--- Training Summary ---")
        print(f"Initial train loss: {epoch_train_losses[0]:.4f}")
        print(f"Final train loss: {epoch_train_losses[-1]:.4f}")
        print(f"Best train loss: {min(epoch_train_losses):.4f}")
        print(f"Initial val loss: {epoch_val_losses[0]:.4f}")
        print(f"Final val loss: {epoch_val_losses[-1]:.4f}")
        print(f"Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
