# models/vae_pretrained.py
import torch
import torch.nn as nn
from diffusers.models import AutoencoderKL

class PretrainedVAEWrapper(nn.Module):
    def __init__(self, model_name="stabilityai/sd-vae-ft-ema", freeze=False):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(model_name)

        if freeze:
            for param in self.vae.parameters():
                param.requires_grad = False
        else:
            self.vae.train()

    def encode(self, x):
        # Input: [B, 1, H, W] → Convert to 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # Convert grayscale to RGB
        latents = self.vae.encode(x).latent_dist.sample()
        latents = latents * 0.18215  # scale factor used in SD
        return latents

    def decode(self, z):
        z = z / 0.18215  # undo scale
        recon = self.vae.decode(z).sample
        return recon

    def forward(self, x):
        # Used for full reconstruction pipeline if needed
        return self.decode(self.encode(x))
