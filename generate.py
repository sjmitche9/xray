# generate.py
import torch
from model import ImageToTextModel
from datasets import load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "checkpoints/"  # or the specific checkpoint dir

def generate_report(model, image, tokenizer, max_len=256):
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(DEVICE)

        # Get image embedding
        features = model.cnn(image.unsqueeze(1))
        features = torch.mean(features.flatten(2), dim=2)
        memory = model.linear(features).unsqueeze(0)

        generated = [tokenizer.cls_token_id]
        for _ in range(max_len):
            input_ids = torch.tensor(generated).unsqueeze(0).to(DEVICE)
            tgt = model.embedding(input_ids).permute(1, 0, 2)
            out = model.decoder(tgt, memory)
            logits = model.output(out.permute(1, 0, 2))[:, -1, :]
            next_token = torch.argmax(logits, dim=-1).item()
            if next_token == tokenizer.sep_token_id:
                break
            generated.append(next_token)

        return tokenizer.decode(generated, skip_special_tokens=True)

def main():
    print("🔄 Loading model and tokenizer...")
    model = ImageToTextModel()
    model.load_state_dict(torch.load(f"{MODEL_PATH}/pytorch_model.bin", map_location=DEVICE))
    model.to(DEVICE)

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    test_dataset = load_from_disk("processed/test")

    for i in range(3):  # show first 3 examples
        image = torch.tensor(test_dataset[i]["image"], dtype=torch.float32)
        target = tokenizer.decode(test_dataset[i]["input_ids"], skip_special_tokens=True)
        generated = generate_report(model, image, tokenizer)

        print(f"\n📷 Example {i+1}")
        print(f"▶️ Generated: {generated}")
        print(f"✅ Target:    {target}")

if __name__ == "__main__":
    main()