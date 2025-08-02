# dataloading.py
import torch

def collate_fn(batch):
    return {
        "z_target": torch.stack([torch.tensor(item["z_target"]) for item in batch]),
        "report": [item["report"] for item in batch],
    }
