# models/unet.py
import torch
import torch.nn as nn

class residual_block(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout2d(dropout)
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
    def __init__(self, in_channels=1, base_channels=96, time_dim=128, context_dim=768):
        super().__init__()

        self.time_mlp = nn.Sequential(
            sinusoidal_time_embedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        self.context_proj = nn.Linear(context_dim, time_dim)
        self.cond_proj = nn.Linear(time_dim, base_channels * 8)

        # Encoder
        self.enc1 = residual_block(in_channels, base_channels)            # 256 → 256
        self.enc2 = residual_block(base_channels, base_channels * 2)      # 128
        self.enc3 = residual_block(base_channels * 2, base_channels * 4)  # 64
        self.enc4 = residual_block(base_channels * 4, base_channels * 8)  # 32
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = residual_block(base_channels * 8, base_channels * 8)

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 8, 2, stride=2)
        self.dec3 = residual_block(base_channels * 8 + base_channels * 4, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, 2, stride=2)
        self.dec2 = residual_block(base_channels * 4 + base_channels * 2, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 2, stride=2)
        self.dec1 = residual_block(base_channels * 2 + base_channels, base_channels)

        # self.up0 = nn.ConvTranspose2d(base_channels, base_channels, 2, stride=2)
        # self.dec0 = residual_block(base_channels + base_channels, base_channels)

        self.out_conv = nn.Conv2d(base_channels, in_channels, 1)

    def forward(self, x, t, context):
        # Combine context and time
        t_emb = self.time_mlp(t)
        c_emb = self.context_proj(context)
        cond = t_emb + c_emb
        cond = self.cond_proj(cond).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

        # Encoder
        x1 = self.enc1(x)                    # [B, base, 256, 256]
        x2 = self.enc2(self.pool(x1))        # [B, base*2, 128, 128]
        x3 = self.enc3(self.pool(x2))        # [B, base*4, 64, 64]
        x4 = self.enc4(self.pool(x3))        # [B, base*8, 32, 32]

        # Bottleneck
        x_b = self.bottleneck(x4 + cond)

        # Decoder
        u3 = self.up3(x_b)                   # [B, base*8, 64, 64]
        d3 = self.dec3(torch.cat([u3, x3], dim=1))

        u2 = self.up2(d3)                    # [B, base*4, 128, 128]
        d2 = self.dec2(torch.cat([u2, x2], dim=1))

        u1 = self.up1(d2)                    # [B, base*2, 256, 256]
        d1 = self.dec1(torch.cat([u1, x1], dim=1))

        # u0 = self.up0(d1)                    # [B, base, 512, 512] if upsampled
        # d0 = self.dec0(torch.cat([u0, x], dim=1))

        return self.out_conv(d1)

# import torch
# import torch.nn as nn

# class residual_block(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 3, padding=1),
#             nn.GroupNorm(8, out_channels),
#             nn.SiLU(),
#             nn.Conv2d(out_channels, out_channels, 3, padding=1),
#             nn.GroupNorm(8, out_channels),
#             nn.SiLU()
#         )
#         self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

#     def forward(self, x):
#         return self.block(x) + self.skip(x)

# class sinusoidal_time_embedding(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim

#     def forward(self, t):
#         device = t.device
#         half_dim = self.dim // 2
#         emb = torch.exp(torch.arange(half_dim, device=device) * -(torch.log(torch.tensor(10000.0)) / half_dim))
#         emb = t[:, None] * emb[None, :]
#         return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

# class conditional_unet(nn.Module):
#     def __init__(self, in_channels=1, base_channels=64, time_dim=128, context_dim=768):
#         super().__init__()

#         # Time + context embedding
#         self.time_mlp = nn.Sequential(
#             sinusoidal_time_embedding(time_dim),
#             nn.Linear(time_dim, time_dim),
#             nn.SiLU(),
#             nn.Linear(time_dim, time_dim)
#         )
#         self.context_proj = nn.Linear(context_dim, time_dim)
#         self.cond_proj = nn.Linear(time_dim, base_channels * 4)

#         # Encoder
#         self.enc1 = residual_block(in_channels, base_channels)
#         self.enc2 = residual_block(base_channels, base_channels * 2)
#         self.enc3 = residual_block(base_channels * 2, base_channels * 4)
#         self.pool = nn.MaxPool2d(2)

#         # Bottleneck
#         self.bottleneck = residual_block(base_channels * 4, base_channels * 8)

#         # Decoder
#         self.up2 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
#         self.dec2 = residual_block(base_channels * 4 + base_channels * 4, base_channels * 2)

#         self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 2, stride=2)
#         self.dec1 = residual_block(base_channels * 2 + base_channels * 2, base_channels)

#         self.up0 = nn.ConvTranspose2d(base_channels, base_channels, 2, stride=2)
#         self.dec0 = residual_block(base_channels + base_channels, base_channels)

#         self.out_conv = nn.Conv2d(base_channels, in_channels, 1)

#     def forward(self, x, t, context):
#         # Conditioning
#         t_emb = self.time_mlp(t)
#         c_emb = self.context_proj(context)
#         cond = t_emb + c_emb
#         cond_proj = self.cond_proj(cond).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

#         # Encoder
#         x1 = self.enc1(x)              # [B, base, 256, 256]
#         x2 = self.enc2(self.pool(x1))  # [B, base*2, 128, 128]
#         x3 = self.enc3(self.pool(x2))  # [B, base*4, 64, 64]

#         # Bottleneck
#         x_b = self.bottleneck(self.pool(x3) + cond_proj)  # [B, base*8, 32, 32]

#         # Decoder
#         x_up2 = self.up2(x_b)                    # [B, base*4, 64, 64]
#         x_cat2 = torch.cat([x_up2, x3], dim=1)   # [B, base*8, 64, 64]
#         x_d2 = self.dec2(x_cat2)

#         x_up1 = self.up1(x_d2)                   # [B, base*2, 128, 128]
#         x_cat1 = torch.cat([x_up1, x2], dim=1)   # [B, base*4, 128, 128]
#         x_d1 = self.dec1(x_cat1)

#         x_up0 = self.up0(x_d1)                   # [B, base, 256, 256]
#         x_cat0 = torch.cat([x_up0, x1], dim=1)   # [B, base*2, 256, 256]
#         x_d0 = self.dec0(x_cat0)

#         return self.out_conv(x_d0)




# # this is the original cond unet:

# # class conditional_unet(nn.Module):
# #     def __init__(self, in_channels, base_channels=96, time_dim=128, context_dim=768):
# #         super().__init__()

# #         # Time + context conditioning
# #         self.time_mlp = nn.Sequential(
# #             sinusoidal_time_embedding(time_dim),
# #             nn.Linear(time_dim, time_dim),
# #             nn.SiLU(),
# #             nn.Linear(time_dim, time_dim)
# #         )
# #         self.context_proj = nn.Linear(context_dim, time_dim)
# #         self.cond_proj = nn.Linear(time_dim, base_channels * 4)

# #         # Encoder
# #         self.enc1 = residual_block(in_channels, base_channels)
# #         self.enc2 = residual_block(base_channels, base_channels * 2)
# #         self.enc3 = residual_block(base_channels * 2, base_channels * 4)
# #         self.pool = nn.MaxPool2d(2)

# #         # Bottleneck
# #         self.bottleneck = residual_block(base_channels * 4, base_channels * 4)

# #         # Decoder
# #         self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
# #         self.dec1 = residual_block(base_channels * 4, base_channels * 2)
# #         self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
# #         self.dec2 = residual_block(base_channels * 2, base_channels)
# #         self.out_conv = nn.Conv2d(base_channels, in_channels, 1)

# #     def forward(self, x, t, context):
# #         t_emb = self.time_mlp(t)
# #         c_emb = self.context_proj(context)
# #         cond = t_emb + c_emb
# #         cond = self.cond_proj(cond).unsqueeze(-1).unsqueeze(-1)

# #         x1 = self.enc1(x)        # 16x16
# #         x2 = self.enc2(self.pool(x1))  # 8x8
# #         x3 = self.enc3(self.pool(x2))  # 4x4

# #         b = self.bottleneck(x3 + cond)  # Add conditioning in bottleneck

# #         u1 = self.up1(b)
# #         d1 = self.dec1(torch.cat([u1, x2], dim=1))
# #         u2 = self.up2(d1)
# #         d2 = self.dec2(torch.cat([u2, x1], dim=1))

# #         return self.out_conv(d2)