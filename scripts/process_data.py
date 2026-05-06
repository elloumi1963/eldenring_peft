import os
import json
import base64
import shutil
import argparse
import time
import random
from pathlib import Path
from openai import OpenAI, RateLimitError

def collect_images(src_dir, dest_dir):
    """Recursively finds all PNGs, renames them sequentially, and moves them to one folder."""
    src_path = Path(src_dir)
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    png_files = list(src_path.rglob("*.png"))
    png_files.sort()
    print(f"Found {len(png_files)} PNG images in '{src_dir}'.")
    print(f"Copying and renaming to '{dest_dir}'...")
    
    for i, file_path in enumerate(png_files):
        new_name = f"{i:04d}.png"
        shutil.copy2(file_path, dest_path / new_name)
    print(f"Successfully collected {len(png_files)} images.")

def caption_images(image_dir, output_file):
    """Reads a folder of PNGs, sends them to OpenAI, and saves to a JSONL file with resume capability."""
    client = OpenAI()
    image_path = Path(image_dir)
    png_files = list(image_path.glob("*.png"))
    png_files.sort()
    
    if not png_files:
        print(f"No PNG files found in '{image_dir}'.")
        return

    processed_files = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_files.add(data.get('file_name'))
                except json.JSONDecodeError:
                    continue
                    
    print(f"Found {len(processed_files)} previously processed images. Resuming...")

    prompt_text = (
        "You are an expert image tagger for Stable Diffusion fine-tuning. Analyze this gameplay screenshot from the game Elden Ring. "
        "Your goal is to provide a concise, highly descriptive caption that is under 60 words. Do not use flowery or poetic language. "
        "Use simple, direct, comma-separated phrases. Always start the caption with the trigger phrase: 'eldenring gameplay style'. "
        "Describe the following elements in order: "
        "1. The main subject (character, enemy, or focal point). Be specific about armor types or creature features. "
        "2. The action or pose. "
        "3. The camera angle (e.g., third-person view, close-up, wide shot). "
        "4. The environment and background. "
        "5. The lighting, weather, and overall color palette."
    )

    with open(output_file, 'a', encoding='utf-8') as f:
        for i, file_path in enumerate(png_files):
            if file_path.name in processed_files:
                print(f"Skipping {file_path.name} (already processed)")
                continue
            
            print(f"Processing {file_path.name} ({i+1}/{len(png_files)})")
                
            with open(file_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode('utf-8')

            max_retries = 10
            success = False
            
            for attempt in range(max_retries):
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt_text},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                                    }
                                ]
                            }
                        ],
                        max_tokens=100
                    )
                    
                    caption = response.choices[0].message.content.strip()
                    json_data = {"file_name": file_path.name, "text": caption}
                    f.write(json.dumps(json_data) + "\n")
                    f.flush() # Force write to disk immediately in case of sudden crash
                    
                    print(f"Successfully captioned: {file_path.name}")
                    success = True
                    break
                    
                except RateLimitError:
                    if attempt == max_retries - 1:
                        print(f"Failed to process {file_path.name} after {max_retries} attempts due to rate limits")
                        break

                    wait_time = min(60, 5 * (2 ** attempt))
                    print(f"Rate limit hit on {file_path.name}. Waiting {wait_time} seconds before retry {attempt + 2}/{max_retries}...")
                    time.sleep(wait_time)
                    
                except Exception as e:
                    print(f"Error processing {file_path.name}: {e}")
                    break
            
            if not success:
                print(f"Skipping {file_path.name} due to persistent errors")

    print(f"Captioning complete! Labels saved to '{output_file}'.")

def split_jsonl(input_file, train_ratio, val_ratio):
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]
    
    random.seed(42) # Fixed seed for reproducibility
    random.shuffle(data)
    
    total = len(data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    def write_jsonl(data, filename):
        with open(filename, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        print(f"Saved {len(data)} items to {filename}")

    write_jsonl(train_data, 'train.jsonl')
    write_jsonl(val_data, 'val.jsonl')
    write_jsonl(test_data, 'test.jsonl')

def main():
    parser = argparse.ArgumentParser(description="A CLI to process and caption image data for Stable Diffusion fine-tuning.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommands")
    
    parser_collect = subparsers.add_parser("collect", help="Recursively read a folder, sequentially rename PNGs, and collect them.")
    parser_collect.add_argument("--src", required=True, help="Source directory to search for PNG images.")
    parser_collect.add_argument("--dest", required=True, help="Destination directory to save the collected images.")
    
    parser_caption = subparsers.add_parser("caption", help="Send a folder of images to OpenAI to generate a labels.jsonl file.")
    parser_caption.add_argument("--dir", required=True, help="Directory containing the collected PNG images.")
    parser_caption.add_argument("--out", default="labels.jsonl", help="Output filename for the JSONL labels (default: labels.jsonl).")

    parser_split = subparsers.add_parser("split", help="Split a JSONL file into train/val/test sets.")
    parser_split.add_argument("--input", required=True, help="Input JSONL file to split.")
    parser_split.add_argument("--train_ratio", default=0.8, type=float, help="Proportion of data to use for training (default: 0.8).")
    parser_split.add_argument("--val_ratio", default=0.1, type=float, help="Proportion of data to use for validation (default: 0.1).")
    
    args = parser.parse_args()
    
    if args.command == "collect":
        collect_images(args.src, args.dest)
    elif args.command == "caption":
        caption_images(args.dir, args.out)
    elif args.command == "split":
        split_jsonl(args.input, args.train_ratio, args.val_ratio)

if __name__ == "__main__":
    main()
