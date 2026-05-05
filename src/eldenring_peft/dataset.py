import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class EldenRingDataset(Dataset):
    def __init__(self, data_dir, jsonl_file, tokenizer, size=512):
        """
        Args:
            data_dir (str): Path to the directory containing images (e.g., 'combined_data').
            jsonl_file (str): Path to the split JSONL file (e.g., 'train.jsonl').
            tokenizer: The Hugging Face tokenizer for text encoding.
            size (int): Image resolution for resizing.
        """
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.size = size
        self.data = []
        
        with open(jsonl_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
                
        # Basic transformations
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) # Normalize to [-1, 1] for SD
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.data_dir, item['file_name'])
        image = Image.open(img_path).convert("RGB")
        # Tokenize the caption text
        text_inputs = self.tokenizer(
            item['text'], 
            padding="max_length", 
            max_length=self.tokenizer.model_max_length, 
            truncation=True, 
            return_tensors="pt"
        )
        
        return {
            "pixel_values": self.transform(image),
            "input_ids": text_inputs.input_ids[0]
        }