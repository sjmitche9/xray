import torch.nn as nn
import torch

class EnhancedDecoder(nn.Module):
    def __init__(self, out_channels=1):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 32 -> 64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 64 -> 128
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, 4, 2, 1),  # 128 -> 256
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.decoder(z)


class EnhancedEncoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),   # 256 → 128
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),           # 128 → 64
            nn.ReLU(),
            nn.Conv2d(128, latent_dim, 4, 2, 1),   # 64 → 32
            nn.ReLU(),
            nn.Conv2d(latent_dim, latent_dim, 4, 2, 1),  # 32 → 16
        )

    def forward(self, x):
        z = self.encoder(x)  # shape: [B, latent_dim, 16, 16]
        return z, torch.zeros_like(z)  # use 0 logvar for now


class EnhancedVAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=256):
        super().__init__()
        self.encoder = EnhancedEncoder(in_channels, latent_dim)
        self.decoder = EnhancedDecoder(in_channels)

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