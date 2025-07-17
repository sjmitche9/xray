
import torch.nn as nn

class LatentUNet(nn.Module):
    def __init__(self, latent_dim=256, cond_dim=768):
        super().__init__()
        self.fc_cond = nn.Linear(cond_dim, latent_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x, cond):
        cond_emb = self.fc_cond(cond)
        x = x + cond_emb
        return self.net(x)
