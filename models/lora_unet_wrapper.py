# models/lora_unet_wrapper.py
import torch.nn as nn

class LoRAUNetWrapper(nn.Module):
    def __init__(self, pretrained_unet, context_dim_in=768, context_dim_out=768):  # ← output = 768
        super().__init__()
        self.unet = pretrained_unet
        self.context_proj = nn.Linear(context_dim_in, context_dim_out)

    def forward(self, x, t, context):  # context: [B, L, 768]
        projected_context = self.context_proj(context)  # → [B, L, 768]
        return self.unet(x, t, encoder_hidden_states=projected_context)