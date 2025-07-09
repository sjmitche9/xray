import os
import torch
from datasets import load_from_disk
import yaml
from models.vae import VAE
from PIL import Image
import numpy as np

# load config
with open("config/config.yaml") as file:
    config = yaml.safe_load(file)

DATA_CFG = config["DATASET"]
MODEL_CFG = config["MODEL"]

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load dataset (test split)
dataset = load_from_disk(os.path.join(DATA_CFG["OUTPUT_PATH"], "test"))
dataset.set_format(type="torch", columns=["image"])

# load model
vae = VAE(in_channels=1, latent_dim=MODEL_CFG["LATENT_DIM"]).to(device)
vae.load_state_dict(torch.load(MODEL_CFG["VAE_CHECKPOINT"], map_location=device))
vae.eval()

# test one sample
sample = dataset[0]["image"].unsqueeze(0).to(device)
with torch.no_grad():
    recon, _, _ = vae(sample)

# normalize and convert to uint8
image_tensor = recon[0, 0].detach().cpu()
image_np = image_tensor.numpy()
image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8)  # normalize
image_np = (image_np * 255).astype(np.uint8)

# convert to image and show
image = Image.fromarray(image_np)
image.show()
