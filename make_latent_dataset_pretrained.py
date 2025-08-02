# make_latent_dataset_pretrained.py
import os
from datasets import load_from_disk, Dataset
from diffusers import AutoencoderKL
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to("cuda").eval()

input_root = "data/processed_dataset/fullset"
output_root = "data/processed_dataset/latent"
os.makedirs(output_root, exist_ok=True)

def encode_chunk(dataset, batch_size=4, num_workers=2):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    latents = []
    for batch in tqdm(loader, desc="Encoding batch", leave=False):
        imgs = batch["image"].to("cuda")
        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        with torch.no_grad():
            zs = vae.encode(imgs).latent_dist.sample().cpu()
        for z, report in zip(zs, batch["report"]):
            latents.append({
                "z_target": z.half(),  # float16 for space
                "report": report
            })
    return latents

def encode_and_save(dataset_path, output_path, split_desc):
    if os.path.exists(os.path.join(output_path, "dataset_info.json")):
        print(f"[skip] {split_desc} already complete")
        return

    print(f"[info] Processing {split_desc}")
    dataset = load_from_disk(dataset_path)
    dataset.set_format("torch", columns=["image", "report"])
    latents = encode_chunk(dataset)

    os.makedirs(output_path, exist_ok=True)
    Dataset.from_list(latents).save_to_disk(output_path)
    print(f"[done] Saved latent {split_desc} to {output_path}")

def main():
    # --- Train Chunks ---
    # chunk_id = 0
    # while True:
    #     chunk_path = os.path.join(input_root, f"train_chunk_{chunk_id}")
    #     if not os.path.exists(chunk_path):
    #         break
    #     out_path = os.path.join(output_root, f"latent_train_chunk_{chunk_id}")
    #     encode_and_save(chunk_path, out_path, f"train_chunk_{chunk_id}")
    #     chunk_id += 1

    # --- Val/Test ---
    for split in ["val", "test"]:
        split_path = os.path.join(input_root, split)
        if os.path.exists(split_path):
            out_path = os.path.join(output_root, f"latent_{split}")
            encode_and_save(split_path, out_path, split)

if __name__ == "__main__":
    main()