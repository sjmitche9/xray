import torch.nn as nn


class TextToLatent(nn.Module):
    def __init__(self, input_dim=768, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),  # after first activation

            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),  # after second activation

            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),  # optional

            nn.Linear(1024, latent_dim),
            # nn.LayerNorm(latent_dim)
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)