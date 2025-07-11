# --- models/vae.py ---
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),  # 256 -> 128
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),           # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),          # 64 -> 32
            nn.ReLU(),
            nn.Flatten()
        )
        self.flattened_dim = 128 * 32 * 32
        self.fc_mu = nn.Linear(self.flattened_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_dim, latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, out_channels=1, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.decoder_input = nn.Linear(latent_dim, 128 * 32 * 32) # Match encoder's flattened_dim
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 32 -> 64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 64 -> 128
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, 4, 2, 1), # 128 -> 256
            nn.Sigmoid()
        )

    def forward(self, z):
        z = self.decoder_input(z)
        z = z.view(-1, 128, 32, 32)
        return self.decoder(z)


class VAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=64):
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
