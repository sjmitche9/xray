# make_latent_dataset_pretrained.py
import os
from datasets import load_from_disk, Dataset, Features, Array3D, Value
from diffusers import AutoencoderKL
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import yaml

with open("config/config.yaml") as f:
	config = yaml.safe_load(f)

# Load VAE
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
ckpt = config["MODEL"].get("VAE_CHECKPOINT")
if ckpt and os.path.exists(ckpt):
    vae.load_state_dict(torch.load(ckpt, map_location="cpu"))
else:
    print(f"[warn] VAE_CHECKPOINT not found at {ckpt}. Using base sd-vae-ft-ema.")
vae = vae.to("cuda").eval()

# Paths
input_root = config["DATASET"]["OUTPUT_PATH"]
output_root = config["DATASET"]["LATENT_OUTPUT_PATH"]
os.makedirs(output_root, exist_ok=True)

def encode_chunk(dataset, batch_size=16, num_workers=4):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    latents = []
    for batch in tqdm(loader, desc="Encoding batch", leave=False):
        imgs = batch["image"].to("cuda")
        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)

        with torch.no_grad():
            zs = vae.encode(imgs).latent_dist.sample() * vae.config.scaling_factor
            zs = zs.float().cpu()


        for z, report in zip(zs, batch["report"]):
            z = z.cpu().float()

            # Add shape check + logging
            if z.shape != (config["MODEL"]["LATENT_DIM"], config["MODEL"]["LATENT_H"], config["MODEL"]["LATENT_W"]):
                raise ValueError(f"❌ Latent has wrong shape: {z.shape}")

            latents.append({
                "z_target": z.numpy().astype("float32"),
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
    features = Features({
        "z_target": Array3D(shape=(config["MODEL"]["LATENT_DIM"], config["MODEL"]["LATENT_H"], config["MODEL"]["LATENT_W"]), dtype="float32"),  # adjust shape as needed
        "report": Value("string")
    })

    Dataset.from_list(latents, features=features).save_to_disk(output_path)
    print(f"[done] Saved latent {split_desc} to {output_path}")

def main():
    # --- Train Chunks ---
    chunk_id = 0
    while True:
        chunk_path = os.path.join(input_root, f"train_chunk_{chunk_id}")
        if not os.path.exists(chunk_path):
            break
        out_path = os.path.join(output_root, f"latent_train_chunk_{chunk_id}")
        encode_and_save(chunk_path, out_path, f"train_chunk_{chunk_id}")
        chunk_id += 1
        break # use this to only create one chunk and skip to the val/test

    # --- Val/Test ---
    for split in ["val", "test"]:
        split_path = os.path.join(input_root, split)
        if os.path.exists(split_path):
            out_path = os.path.join(output_root, f"latent_{split}")
            encode_and_save(split_path, out_path, split)

if __name__ == "__main__":
    main()