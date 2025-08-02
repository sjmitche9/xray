# models/vae_enhanced.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedEncoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),   # 256 → 128
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 4, 2, 1),           # 128 → 64
            nn.GroupNorm(8, 128),
            nn.ReLU(),

            nn.Conv2d(128, 256, 4, 2, 1),          # 64 → 32
            nn.GroupNorm(8, 256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(256, 256, 4, 2, 1),          # 32 → 16
            nn.GroupNorm(8, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.conv_mu = nn.Conv2d(256, latent_dim, kernel_size=1)
        self.conv_logvar = nn.Conv2d(256, latent_dim, kernel_size=1)

    def forward(self, x):
        h = self.encoder(x)
        mu = self.conv_mu(h)
        logvar = self.conv_logvar(h)
        return mu, logvar


class EnhancedDecoder(nn.Module):
    def __init__(self, out_channels=1):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 16 → 32
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 32 → 64
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.ConvTranspose2d(32, 16, 4, 2, 1),    # 64 → 128
            nn.ReLU(),

            nn.ConvTranspose2d(16, out_channels, 4, 2, 1),  # 128 → 256
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.decoder(z)



class EnhancedVAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=128):
        super().__init__()
        self.encoder = EnhancedEncoder(in_channels, latent_dim)
        self.decoder = EnhancedDecoder(in_channels)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        z = F.dropout(z, p=0.1, training=self.training)
        return z

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return self.reparameterize(mu, logvar)

    def decode(self, z):
        z = F.dropout(z, p=0.1, training=self.training)
        return self.decoder(z)
