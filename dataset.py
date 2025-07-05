# dataset.py
from transformers import AutoTokenizer
import torch

class DataCollatorImageText:
    def __init__(self, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        images = torch.stack([torch.tensor(ex["image"]) for ex in batch])
        input_ids = [ex["input_ids"] for ex in batch]

        # Pad sequences
        enc = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        labels = enc["input_ids"].clone()
        return {
            "image": images,
            "decoder_input_ids": enc["input_ids"][:, :-1],
            "labels": labels[:, 1:]
        }