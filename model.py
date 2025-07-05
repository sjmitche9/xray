from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision import models

train_dataset = load_from_disk("processed/train")
val_dataset = load_from_disk("processed/val")
test_dataset = load_from_disk("processed/test")

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)



class ImageToTextModel(nn.Module):
    
    def __init__(self, hidden_dim=768, vocab_size=30522, max_length=256):
        super().__init__()

        # Vision encoder: ResNet50 pretrained on ImageNet
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # adjust for grayscale
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool + FC

        self.linear = nn.Linear(2048, hidden_dim)  # project to match transformer dim

        # Text decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        self.max_length = max_length
        self.hidden_dim = hidden_dim

    def forward(self, images, input_ids):
        # Encode image
        features = self.cnn(images.unsqueeze(1))  # [B, 2048, H, W]
        features = torch.mean(features.flatten(2), dim=2)  # [B, 2048]
        features = self.linear(features).unsqueeze(0)      # [1, B, hidden_dim]

        # Embed token sequence
        tgt = self.embedding(input_ids).permute(1, 0, 2)  # [seq_len, B, dim]

        # Decode
        output = self.transformer_decoder(tgt=tgt, memory=features)
        output = output.permute(1, 0, 2)  # [B, seq_len, dim]

        return self.fc_out(output)