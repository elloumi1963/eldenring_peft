import os
import json
import argparse
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
from tqdm.auto import tqdm
from peft import PeftModel


def load_test_data(jsonl_file, img_dir):
    data = []
    with open(jsonl_file, "r") as f:
        for line in f:
            item = json.loads(line)
            data.append({
                "prompt": item["text"],
                "real_img_path": os.path.join(img_dir, item["file_name"])
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
        pipe.unet = PeftModel.from_pretrained(pipe.unet, peft_path)
        print("Loaded via PeftModel.from_pretrained()")
    else:
        pipe.load_lora_weights(peft_path)
        print("Loaded via pipe.load_lora_weights()")

    return pipe


def load_full_finetune(pipe, weights_path, device, dtype):
    if weights_path is None:
        raise ValueError("--weights_path is required for method=full")

    print(f"Loading full UNet fine-tune from: {weights_path}")

    pipe.unet = UNet2DConditionModel.from_pretrained(
        weights_path,
        torch_dtype=dtype
    ).to(device)

    return pipe


def count_adapter_params(unet):
    adapter_params = 0
    base_params = 0

    adapter_keys = [
        "lora_A", "lora_B",
        "lora_magnitude_vector",
        "ranknum",
    ]

    for name, param in unet.named_parameters():
        if any(k in name for k in adapter_keys):
            adapter_params += param.numel()
        else:
            base_params += param.numel()

    total_params = base_params + adapter_params
    return base_params, adapter_params, total_params


def build_default_paths(args):
    if args.output_file is None:
        args.output_file = f"outputs/{args.method}/eval/results.json"

    if args.samples_dir is None:
        args.samples_dir = f"outputs/{args.method}/eval/samples"

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
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

    elif args.method == "full":
        pipe = load_full_finetune(pipe, args.weights_path, device, dtype)

    else:
        raise ValueError(f"Unknown method: {args.method}")

    base_params, adapter_params, total_params = count_adapter_params(pipe.unet)
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

        clip_score.update(fake_img_tensor.squeeze(0), prompt)

    print("\nComputing final scores...")

    final_fid = fid.compute().item()
    final_kid = kid.compute()
    final_clip = clip_score.compute().item()

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
        "CLIPScore (Higher is better)": round(final_clip, 4),
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

    #custom_prompts = [
     #   "eldenring gameplay style, third person view, player character fighting a massive dragon boss, dynamic motion, sparks, fire, cinematic camera angle",
     #   "eldenring gameplay style, boss arena, grotesque humanoid monster, detailed armor, high contrast lighting, epic composition",
      #  "eldenring gameplay style, open world exploration, distant castle, broken bridges, dead trees, misty environment, wide shot",
    #]
    custom_prompts = [
        "eldenring gameplay style, third person view, player character fighting a massive dragon boss, dynamic motion, sparks, fire, cinematic camera angle",
        "eldenring gameplay style, boss arena, grotesque humanoid monster, detailed armor, high contrast lighting, epic composition",
        "eldenring gameplay style, open world exploration, distant castle, broken bridges, dead trees, misty environment, wide shot"
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
    parser = argparse.ArgumentParser(description="Evaluate base, PEFT, or full fine-tuned Stable Diffusion.")

    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["base", "lora", "dora", "adalora", "full"],
    )

    parser.add_argument(
        "--weights_path",
        type=str,
        default=None,
        help="Path to weights. Required for lora/dora/adalora/full. Not used for base.",
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

    args = parser.parse_args()
    main(args)
