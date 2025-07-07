import torch
import torch.nn as nn
import yaml
import gc
import wandb
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer, TrainingArguments, Trainer
import evaluate
from baseline.model import ImageToTextModel
from preprocess_mimic_jpg import DataCollatorImageText

def generate_text_debug(model, image_batch, tokenizer, max_length=128, device="cuda"):
    model.eval()
    batch_size = image_batch.size(0)

    start_token = tokenizer.cls_token_id or tokenizer.bos_token_id
    if start_token is None:
        raise ValueError("Tokenizer must have cls_token_id or bos_token_id defined.")

    generated = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
    attention_mask = torch.ones_like(generated, device=device)

    print("[Debug] Starting generation...")

    for step in range(max_length - 1):
        try:
            outputs = model(image=image_batch, input_ids=generated, attention_mask=attention_mask)
            logits = outputs["logits"]

            next_token_logits = logits[:, -1, :]
            next_tokens = next_token_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_tokens], dim=1)
            attention_mask = torch.ones_like(generated)

            if (next_tokens == tokenizer.sep_token_id).all():
                break

        except Exception as e:
            print(f"[Error during generation at step {step}]: {e}")
            break

    return generated

def evaluate_generation_debug(model, dataset, tokenizer, batch_size=4, device="cuda"):
    model.eval()
    preds, refs = [], []

    dataloader = DataLoader(dataset, batch_size=batch_size)

    for batch_idx, batch in enumerate(dataloader):
        try:
            images = batch["image"].to(device)
            labels = batch["labels"].to(device)

            print(f"\n[Batch {batch_idx}] Image shape: {images.shape}")
            print(f"[Batch {batch_idx}] Labels shape: {labels.shape}")

            generated_ids = generate_text_debug(model, images, tokenizer, max_length=128, device=device)
            decoded_preds = tokenizer.batch_decode(generated_ids.cpu(), skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels.cpu(), skip_special_tokens=True)

            print("[Decoded preds sample]:", decoded_preds[:2])
            print("[Decoded labels sample]:", decoded_labels[:2])

            preds.extend(decoded_preds)
            refs.extend(decoded_labels)

        except Exception as e:
            print(f"[Error during batch {batch_idx}]: {e}")

    metrics = {}
    if preds and refs:
        try:
            metrics["bleu"] = evaluate.load("bleu").compute(predictions=preds, references=[[r] for r in refs])["bleu"]
            metrics["rougeL"] = evaluate.load("rouge").compute(predictions=preds, references=refs)["rougeL"]
            metrics["meteor"] = evaluate.load("meteor").compute(predictions=preds, references=refs)["meteor"]
        except Exception as e:
            print(f"[Error computing metrics]: {e}")

    return metrics, preds, refs

def main():
    wandb.init(project="xray", name="debug_run")

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    dataset_dict = load_from_disk("processed/iu_xray_hf_split")
    val_dataset = dataset_dict["validation"].select(range(10))

    model = ImageToTextModel(vocab_size=config.get("vocab_size", tokenizer.vocab_size))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    metrics, preds, refs = evaluate_generation_debug(model, val_dataset, tokenizer, batch_size=2, device=device)

    print("\nFinal Metrics:", metrics)

    table = wandb.Table(columns=["Prediction", "Reference"])
    for p, r in zip(preds[:20], refs[:20]):
        table.add_data(p, r)
    wandb.log({"eval_examples": table})

if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()
    main()
