# # --- models/unet.py ---

import torch
import torch.nn as nn

class residual_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.skip(x)

class sinusoidal_time_embedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=device) * -(torch.log(torch.tensor(10000.0)) / half_dim))
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

class conditional_unet(nn.Module):
    def __init__(self, in_channels=64, base_channels=96, time_dim=128, context_dim=768):
        super().__init__()

        # Time + context conditioning
        self.time_mlp = nn.Sequential(
            sinusoidal_time_embedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        self.context_proj = nn.Linear(context_dim, time_dim)
        self.cond_proj = nn.Linear(time_dim, base_channels * 4)

        # Encoder
        self.enc1 = residual_block(in_channels, base_channels)
        self.enc2 = residual_block(base_channels, base_channels * 2)
        self.enc3 = residual_block(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = residual_block(base_channels * 4, base_channels * 4)

        # Decoder
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec1 = residual_block(base_channels * 4, base_channels * 2)
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec2 = residual_block(base_channels * 2, base_channels)
        # self.up3 = nn.ConvTranspose2d(base_channels, base_channels, 2, stride=2)
        # self.dec3 = residual_block(base_channels * 2, base_channels)

        self.out_conv = nn.Conv2d(base_channels, in_channels, 1)

    def forward(self, x, t, context):
        t_emb = self.time_mlp(t)
        c_emb = self.context_proj(context)
        cond = t_emb + c_emb
        cond = self.cond_proj(cond).unsqueeze(-1).unsqueeze(-1)

        x1 = self.enc1(x)        # 16x16
        x2 = self.enc2(self.pool(x1))  # 8x8
        x3 = self.enc3(self.pool(x2))  # 4x4

        b = self.bottleneck(x3 + cond)  # Add conditioning in bottleneck

        u1 = self.up1(b)
        d1 = self.dec1(torch.cat([u1, x2], dim=1))
        u2 = self.up2(d1)
        d2 = self.dec2(torch.cat([u2, x1], dim=1))
        # u3 = self.up3(d2)
        # x1_up = nn.functional.interpolate(x1, size=u3.shape[2:], mode="bilinear", align_corners=False)
        # d3 = self.dec3(torch.cat([u3, x1_up], dim=1))

        # return self.out_conv(d3)
        return self.out_conv(d2)