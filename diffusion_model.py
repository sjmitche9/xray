import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=32):
        super(SimpleUNet, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features, features * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(features * 2, features * 2, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.upconv = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)

        self.decoder1 = nn.Sequential(
            nn.Conv2d(features * 2, features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(features, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, t_emb):
        enc1 = self.encoder1(x)
        pooled = self.pool(enc1)

        bottleneck = self.bottleneck(pooled)

        up = self.upconv(bottleneck)

        concat = torch.cat([up, enc1], dim=1)
        out = self.decoder1(concat)

        return out

class DiffusionModel(nn.Module):
    def __init__(self, unet_model, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.unet = unet_model
        self.timesteps = timesteps

        self.beta = torch.linspace(beta_start, beta_end, timesteps)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def forward(self, x, t):
        t_emb = self._time_embedding(t, x.shape)
        return self.unet(x, t_emb)

    def _time_embedding(self, t, shape):
        # Simple broadcast time embedding
        return t[:, None, None, None].expand(-1, shape[2], shape[3])
