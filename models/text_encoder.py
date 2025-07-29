# models/text_encoder.py
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class text_encoder(nn.Module):

    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT"):
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input_text):
        tokens = self.tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.bert.device)

        output = self.bert(**tokens)

        return output.last_hidden_state.mean(dim=1)  # shape: (batch, hidden_dim)