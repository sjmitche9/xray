import os
import torch
import yaml
import numpy as np
from datasets import load_from_disk
from models.vae import VAE
from PIL import Image

# Load config
with open("config/config.yaml") as file:
    config = yaml.safe_load(file)

DATA_CFG = config["DATASET"]
MODEL_CFG = config["MODEL"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load sample from train_chunk_0
chunk_path = os.path.join(DATA_CFG["OUTPUT_PATH"], "train_chunk_0")
dataset = load_from_disk(chunk_path)
dataset.set_format(type="torch", columns=["image"])
sample = dataset[0]["image"].unsqueeze(0).to(device)

# Load model
vae = VAE(in_channels=1, latent_dim=MODEL_CFG["LATENT_DIM"]).to(device)
vae.load_state_dict(torch.load(MODEL_CFG["VAE_CHECKPOINT"], map_location=device))
vae.eval()

# Reconstruct
with torch.no_grad():
    recon, _, _ = vae(sample)

# Normalize original and recon to [0, 255]
def normalize_to_image(tensor):
    arr = tensor.detach().cpu().numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return (arr * 255).astype(np.uint8)

original_np = normalize_to_image(sample[0, 0])
recon_np = normalize_to_image(recon[0, 0])

original_img = Image.fromarray(original_np)
recon_img = Image.fromarray(recon_np)

# Combine side-by-side
combined = Image.new("L", (original_img.width * 2, original_img.height))
combined.paste(original_img, (0, 0))
combined.paste(recon_img, (original_img.width, 0))
combined.show()
