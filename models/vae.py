# --- models/vae.py ---
import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, in_channels=1, latent_dim=8):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),  # 256 -> 128
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),            # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),           # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(128, 2 * latent_dim, 1)      # Output: mean and logvar
        )

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar


class Decoder(nn.Module):

    def __init__(self, out_channels=1, latent_dim=8):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, 4, 2, 1),  # 32 -> 64
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),          # 64 -> 128
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),           # 128 -> 256
            nn.ReLU(),
            nn.Conv2d(32, out_channels, 3, padding=1),     # Final conv
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.decoder(z)


class VAE(nn.Module):

    def __init__(self, in_channels=1, latent_dim=8):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(in_channels, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def encode(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z

    def decode(self, z):
        return self.decoder(z)