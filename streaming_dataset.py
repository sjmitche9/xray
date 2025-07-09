# --- streaming_dataset.py ---

import os
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_from_disk

class StreamingHFDataset(IterableDataset):
    def __init__(self, chunk_dir, batch_size=32, shuffle=True):
        self.chunk_dir = chunk_dir
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        chunk_idx = 0
        while True:
            chunk_path = os.path.join(self.chunk_dir, f"train_chunk_{chunk_idx}")
            if not os.path.exists(chunk_path):
                break

            dataset = load_from_disk(chunk_path)
            dataset.set_format(type="torch", columns=[
                "image", "input_ids", "attention_mask", "report"])

            loader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                shuffle=self.shuffle)

            for batch in loader:
                yield batch

            chunk_idx += 1