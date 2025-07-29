# dataloading.py
import torch

def collate_fn(batch):
    return {
        "image": torch.stack([torch.tensor(item["image"], dtype=torch.float32) for item in batch]),
        "input_ids": torch.tensor([item["input_ids"] for item in batch]),
        "attention_mask": torch.tensor([item["attention_mask"] for item in batch]),
    }
