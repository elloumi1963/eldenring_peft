import os
import json
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
from tqdm.auto import tqdm
from peft import PeftModel


DEFAULT_TARGET_MODULES = [
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
]


def parse_target_modules(value):
    return [x.strip() for x in value.split(",") if x.strip()]


class DCFTLinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, r=16, kernel_size=4, stride=2, dropout=0.1):
        super().__init__()

        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features

        self.weight = linear_layer.weight
        self.bias = linear_layer.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        self.r = r
        self.compressed_dim = max(1, r // stride)

        self.down_proj = nn.Linear(self.in_features, self.compressed_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout)

        self.deconv = nn.ConvTranspose1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )

        deconv_out_len = (self.compressed_dim - 1) * stride - 2 * (kernel_size // 2) + kernel_size
        self.up_proj = nn.Linear(deconv_out_len, self.out_features, bias=False)

    def forward(self, x):
        base_out = F.linear(x, self.weight, self.bias)

        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        subspace_x = self.down_proj(x_flat)
        subspace_x = self.dropout(subspace_x)

        subspace_x = subspace_x.unsqueeze(1)
        deconv_x = self.deconv(subspace_x).squeeze(1)

        dcft_out = self.up_proj(deconv_x)
        dcft_out = dcft_out.reshape(*orig_shape[:-1], self.out_features)

        return base_out + dcft_out


def inject_dcft(module, target_modules, r=16, kernel_size=4, stride=2, dropout=0.1):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and any(t in name for t in target_modules):
            setattr(
                module,
                name,
                DCFTLinear(
                    child,
                    r=r,
                    kernel_size=kernel_size,
                    stride=stride,
                    dropout=dropout,
                ),
            )
        else:
            inject_dcft(child, target_modules, r, kernel_size, stride, dropout)


def load_test_data(jsonl_file, img_dir):
    data = []
    with open(jsonl_file, "r") as f:
        for line in f:
            item = json.loads(line)
            data.append({
                "prompt": item["text"],
                "real_img_path": os.path.join(img_dir, item["file_name"]),
            })
    return data


def preprocess_image(image, size=512):
    image = image.resize((size, size))
    image = TF.pil_to_tensor(image)
    return image.unsqueeze(0)


def patch_clipscore(clip_score):
    def extract_tensor(features):
        if isinstance(features, torch.Tensor):
            return features
        if hasattr(features, "image_embeds"):
            return features.image_embeds
        if hasattr(features, "text_embeds"):
            return features.text_embeds
        if hasattr(features, "pooler_output"):
            return features.pooler_output
        return features

    original_get_image = clip_score.model.get_image_features
    clip_score.model.get_image_features = lambda *args, **kwargs: extract_tensor(
        original_get_image(*args, **kwargs)
    )

    original_get_text = clip_score.model.get_text_features
    clip_score.model.get_text_features = lambda *args, **kwargs: extract_tensor(
        original_get_text(*args, **kwargs)
    )


def load_peft_model(pipe, peft_path, method):
    if peft_path is None:
        raise ValueError(f"--weights_path is required for method={method}")

    print(f"Loading {method} weights from: {peft_path}")

    if not os.path.exists(peft_path):
        raise FileNotFoundError(f"Weights path does not exist: {peft_path}")

    is_peft_format = os.path.exists(os.path.join(peft_path, "adapter_config.json"))

    if is_peft_format:
        pipe.unet = PeftModel.from_pretrained(
            pipe.unet,
            peft_path,
            local_files_only=True,
        )
        print("Loaded via PeftModel.from_pretrained()")
    else:
        pipe.load_lora_weights(peft_path, local_files_only=True)
        print("Loaded via pipe.load_lora_weights()")

    return pipe


def load_full_finetune(pipe, weights_path, device, dtype):
    if weights_path is None:
        raise ValueError("--weights_path is required for method=full")

    print(f"Loading full UNet fine-tune from: {weights_path}")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights path does not exist: {weights_path}")

    pipe.unet = UNet2DConditionModel.from_pretrained(
        weights_path,
        torch_dtype=dtype,
    ).to(device)

    return pipe


def load_dcft_model(pipe, weights_path, device, args):
    if weights_path is None:
        raise ValueError("--weights_path is required for method=dcft")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights path does not exist: {weights_path}")

    dcft_weights_path = weights_path
    if os.path.isdir(weights_path):
        dcft_weights_path = os.path.join(weights_path, "dcft_weights.pt")

    if not os.path.exists(dcft_weights_path):
        raise FileNotFoundError(f"Could not find DCFT weights: {dcft_weights_path}")

    target_modules = parse_target_modules(args.target_modules)

    print(
        f"Injecting DCFT before loading weights "
        f"(r={args.dcft_r}, k={args.dcft_kernel_size}, "
        f"s={args.dcft_stride}, dropout={args.dcft_dropout})"
    )

    inject_dcft(
        pipe.unet,
        target_modules=target_modules,
        r=args.dcft_r,
        kernel_size=args.dcft_kernel_size,
        stride=args.dcft_stride,
        dropout=args.dcft_dropout,
    )

    state_dict = torch.load(dcft_weights_path, map_location="cpu")
    missing, unexpected = pipe.unet.load_state_dict(state_dict, strict=False)

    print(f"Loaded DCFT weights from: {dcft_weights_path}")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    pipe.unet.to(device)
    return pipe


def count_adapter_params(unet, method):
    total_params = sum(p.numel() for p in unet.parameters())

    if method == "base":
        return total_params, 0, total_params

    if method == "full":
        return 0, total_params, total_params

    adapter_params = 0
    base_params = 0

    if method == "dcft":
        adapter_keys = ["down_proj", "up_proj", "deconv"]
    else:
        adapter_keys = [
            "lora_A",
            "lora_B",
            "lora_magnitude_vector",
            "ranknum",
        ]

    for name, param in unet.named_parameters():
        if any(k in name for k in adapter_keys):
            adapter_params += param.numel()
        else:
            base_params += param.numel()

    return base_params, adapter_params, total_params


def build_default_paths(args):
    if args.output_file is None:
        args.output_file = f"outputs/{args.method}/eval/results.json"

    if args.samples_dir is None:
        args.samples_dir = f"outputs/{args.method}/eval/samples"

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    os.makedirs(args.samples_dir, exist_ok=True)

    return args


def main(args):
    args = build_default_paths(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Running evaluation on {device}")
    print(f"Method: {args.method}")

    fid = FrechetInceptionDistance(feature=2048).to(device)
    kid = KernelInceptionDistance(subset_size=args.kid_subset_size).to(device)

    clip_score = None
    if not args.skip_clip:
        clip_score = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)
        patch_clipscore(clip_score)

    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    pipe.set_progress_bar_config(disable=True)

    if args.method == "base":
        print("Evaluating base model only. No weights loaded.")

    elif args.method in ["lora", "dora", "adalora"]:
        pipe = load_peft_model(pipe, args.weights_path, args.method)

    elif args.method == "dcft":
        pipe = load_dcft_model(pipe, args.weights_path, device, args)

    elif args.method == "full":
        pipe = load_full_finetune(pipe, args.weights_path, device, dtype)

    else:
        raise ValueError(f"Unknown method: {args.method}")

    base_params, adapter_params, total_params = count_adapter_params(pipe.unet, args.method)
    adapter_pct = round(100 * adapter_params / total_params, 4) if total_params > 0 else 0.0

    print("\nUNet Parameter Breakdown:")
    print(f"  Base params:     {base_params:,}")
    print(f"  Adapter params:  {adapter_params:,}")
    print(f"  Total params:    {total_params:,}")
    print(f"  Adapter %:       {adapter_pct}%")

    test_data = load_test_data(args.test_jsonl, args.data_dir)
    print(f"Loaded {len(test_data)} test samples.")

    print("Generating images and updating metrics...")

    for idx, item in enumerate(tqdm(test_data, desc="Evaluating")):
        prompt = item["prompt"]

        real_img_pil = Image.open(item["real_img_path"]).convert("RGB")
        real_img_tensor = preprocess_image(real_img_pil).to(device)

        generator = torch.Generator(device=device).manual_seed(args.seed + idx)

        fake_img_pil = pipe(
            prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0]

        fake_img_tensor = preprocess_image(fake_img_pil).to(device)

        comparison = Image.new("RGB", (1024, 512))
        comparison.paste(real_img_pil.resize((512, 512)), (0, 0))
        comparison.paste(fake_img_pil.resize((512, 512)), (512, 0))
        comparison.save(os.path.join(args.samples_dir, f"sample_{idx:03d}.png"))

        fid.update(real_img_tensor, real=True)
        fid.update(fake_img_tensor, real=False)

        kid.update(real_img_tensor, real=True)
        kid.update(fake_img_tensor, real=False)

        if clip_score is not None:
            clip_score.update(fake_img_tensor.squeeze(0), prompt)

    print("\nComputing final scores...")

    final_fid = fid.compute().item()
    final_kid = kid.compute()
    final_clip = clip_score.compute().item() if clip_score is not None else None

    results = {
        "method": args.method,
        "model_id": args.model_id,
        "weights_path": args.weights_path,
        "Base Parameters": base_params,
        "Adapter Parameters": adapter_params,
        "Total Parameters": total_params,
        "Adapter %": adapter_pct,
        "FID (Lower is better)": round(final_fid, 4),
        "KID Mean (Lower is better)": round(final_kid[0].item(), 4),
        "KID Std": round(final_kid[1].item(), 4),
        "CLIPScore (Higher is better)": round(final_clip, 4) if final_clip is not None else "SKIPPED",
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
    }

    print("\n--- Evaluation Results ---")
    for k, v in results.items():
        print(f"{k}: {v}")

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {args.output_file}")
    print(f"Samples saved to {args.samples_dir}")

    custom_prompts = [
        "eldenring gameplay style, third person view, player character fighting a massive dragon boss, dynamic motion, sparks, fire, cinematic camera angle",
        "eldenring gameplay style, boss arena, grotesque humanoid monster, detailed armor, high contrast lighting, epic composition",
        "eldenring gameplay style, open world exploration, distant castle, broken bridges, dead trees, misty environment, wide shot",
    ]

    print(f"\nGenerating {len(custom_prompts)} custom Elden Ring style images...")

    for i, prompt in enumerate(custom_prompts, 1):
        generator = torch.Generator(device=device).manual_seed(args.seed + 1000 + i)

        custom_img = pipe(
            prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0]

        output_path = os.path.join(args.samples_dir, f"eldenring_custom_{i:02d}.png")
        custom_img.save(output_path)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate base, PEFT, DCFT, or full fine-tuned Stable Diffusion."
    )

    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["base", "lora", "dora", "adalora", "dcft", "full"],
    )

    parser.add_argument(
        "--weights_path",
        type=str,
        default=None,
        help="Path to weights. Required for lora/dora/adalora/dcft/full. Not used for base.",
    )

    parser.add_argument("--model_id", type=str, default="sd-legacy/stable-diffusion-v1-5")
    parser.add_argument("--test_jsonl", type=str, default="test.jsonl")
    parser.add_argument("--data_dir", type=str, default="combined_data")

    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--samples_dir", type=str, default=None)

    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kid_subset_size", type=int, default=50)
    parser.add_argument("--skip_clip", action="store_true")

    parser.add_argument("--target_modules", type=str, default=",".join(DEFAULT_TARGET_MODULES))
    parser.add_argument("--dcft_r", type=int, default=16)
    parser.add_argument("--dcft_kernel_size", type=int, default=4)
    parser.add_argument("--dcft_stride", type=int, default=2)
    parser.add_argument("--dcft_dropout", type=float, default=0.1)

    args = parser.parse_args()
    main(args)
