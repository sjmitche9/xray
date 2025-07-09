# --- generate.py ---
import torch
import yaml
import numpy as np
from PIL import Image
from models.vae import VAE
from models.text_encoder import text_encoder
from models.unet import conditional_unet
from scheduler.ddpm_scheduler import ddpm_scheduler

# load config
with open("config/config.yaml") as file:
    config = yaml.safe_load(file)

MODEL_CFG = config["MODEL"]
TRAIN_CFG = config["TRAINING"]

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# init models
vae = VAE().to(device)
vae.load_state_dict(torch.load(MODEL_CFG["VAE_CHECKPOINT"], map_location=device))
vae.eval()

text_enc = text_encoder().to(device)
unet = conditional_unet().to(device)
scheduler = ddpm_scheduler()

# prompt
report = "lungs are clear. no acute cardiopulmonary abnormality."
context = text_enc([report]).to(device)
null_context = torch.zeros_like(context)

# sample
z = torch.randn(1, MODEL_CFG["LATENT_DIM"], MODEL_CFG["LATENT_H"], MODEL_CFG["LATENT_W"]).to(device)
for t_gen in reversed(range(scheduler.num_timesteps)):
    t_tensor = torch.full((1,), t_gen, device=device, dtype=torch.long)
    pred_cond = unet(z, t_tensor, context)
    pred_uncond = unet(z, t_tensor, null_context)
    pred = pred_uncond + TRAIN_CFG["GUIDANCE_SCALE"] * (pred_cond - pred_uncond)

    alpha_t = scheduler.alpha[t_gen]
    alpha_hat_t = scheduler.alpha_hat[t_gen]
    beta_t = scheduler.beta[t_gen]

    z = (1 / torch.sqrt(alpha_t)) * (z - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * pred)
    if t_gen > 0:
        z += torch.sqrt(beta_t) * torch.randn_like(z)

# decode and view
image = vae.decode(z)[0].squeeze(0).cpu().numpy()
image = (image * 255).astype(np.uint8)
Image.fromarray(image).show()