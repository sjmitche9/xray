import torch.nn as nn
from transformers import AutoModel
import torch

class ImageToTextModel(nn.Module):

    def __init__(self, vocab_size, model_name="emilyalsentzer/Bio_ClinicalBERT"):
        super().__init__()
        
        # Efficient image encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),  # 256x256 -> 128x128
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),                         # smaller spatial
            nn.Flatten(),
            nn.Linear(8 * 4 * 4, 256),
            nn.ReLU(),
        )

        # Text decoder
        self.decoder = AutoModel.from_pretrained(model_name)

        # LM head
        self.lm_head = nn.Linear(self.decoder.config.hidden_size, vocab_size)

        # # Freeze decoder layers
        # for param in self.decoder.parameters():
        #     param.requires_grad = False
        
        # # Unfreeze last encoder layer (adjust name as needed)
        # for param in self.decoder.encoder.layer[-1].parameters():
        #     param.requires_grad = True

        # # Unfreeze lm_head
        # for param in self.lm_head.parameters():
        #     param.requires_grad = True

        # # Enable gradient checkpointing for memory savings
        # self.decoder.gradient_checkpointing_enable()


        

        
    def forward(self, image, input_ids=None, attention_mask=None, labels=None):
        # ✅ Debugging print statements (only print once)
        if not hasattr(self, "_has_logged_shapes"):
            print(f"🧾 Image shape: {image.shape}")
            print(f"🧾 Input IDs shape: {input_ids.shape if input_ids is not None else 'None'}")
            print(f"🧾 Labels shape: {labels.shape if labels is not None else 'None'}")
            print(f"🧠 CUDA allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
            print(f"🧠 CUDA reserved:  {torch.cuda.memory_reserved() / 1e6:.2f} MB")
            self._has_logged_shapes = True

        # 🖼️ Encode image
        if image.dim() == 3:
            image = image.unsqueeze(1)  # Ensure (B, 1, H, W)
        image_emb = self.encoder(image)  # (B, 768)

        # 🔤 Pass text through decoder
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = self.lm_head(decoder_outputs.last_hidden_state)

        output = {"logits": logits}

        # 🧮 If labels are provided, calculate loss
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            # Reshape logits and labels to (B * seq_len, vocab_size) and (B * seq_len)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            output["loss"] = loss

        return output