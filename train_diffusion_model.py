import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_from_disk
import wandb
import yaml
from tqdm import tqdm
from transformers import DataCollatorWithPadding

from models.vae import VAE
from models.diffusion import DiffusionModel
from torch.optim import Adam

# --- Load config ---
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

DATA_CFG = config["DATASET"]
MODEL_CFG = config["MODEL"]
TRAIN_CFG = config["TRAINING"]
WANDB_CFG = config["WANDB"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Init models ---
vae = VAE(in_channels=1, latent_dim=MODEL_CFG["LATENT_DIM"]).to(device)
if MODEL_CFG.get("VAE_RESUME") and os.path.exists(MODEL_CFG["VAE_CHECKPOINT"]):
    vae.load_state_dict(torch.load(MODEL_CFG["VAE_CHECKPOINT"], map_location=device))
vae.eval()  # VAE is frozen during diffusion training

diffusion_model = DiffusionModel(config).to(device)
if MODEL_CFG.get("DIFFUSION_RESUME") and os.path.exists(MODEL_CFG["DIFFUSION_CHECKPOINT"]):
    diffusion_model.unet.load_state_dict(torch.load(MODEL_CFG["DIFFUSION_CHECKPOINT"], map_location=device))

optimizer = Adam(diffusion_model.unet.parameters(), lr=float(TRAIN_CFG["LEARNING_RATE"]))

# --- Init W&B ---
wandb.init(project=WANDB_CFG["PROJECT"], name=WANDB_CFG["RUN_NAME_DIFFUSION"], config=config)

# --- Initialize tokenizer and data collator ---
tokenizer = diffusion_model.text_encoder.tokenizer
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# --- Collate function ---
def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    reports = [item["report"] for item in batch]
    tokenized = data_collator([{
        "input_ids": item["input_ids"],
        "attention_mask": item["attention_mask"]
    } for item in batch])
    tokenized["image"] = images
    tokenized["report"] = reports
    return tokenized

output_path = DATA_CFG["OUTPUT_PATH"]
chunk_dirs = sorted([d for d in os.listdir(output_path) if d.startswith("train_chunk_")])

# Load validation set
val_dataset = load_from_disk(os.path.join(output_path, "val"))
val_dataset.set_format(type="torch", columns=["image", "input_ids", "attention_mask", "report"])
val_loader = DataLoader(val_dataset, batch_size=TRAIN_CFG["BATCH_SIZE"], collate_fn=collate_fn)

diffusion_ckpt_path = MODEL_CFG["DIFFUSION_CHECKPOINT"]
vae_ckpt_path = MODEL_CFG["VAE_CHECKPOINT"]
best_val_loss = float("inf")
no_improvement_epochs = 0
early_stop_patience = TRAIN_CFG.get("EARLY_STOP_PATIENCE", 5)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=float(TRAIN_CFG["LR_SCHEDULER"]["FACTOR"]),
    patience=int(TRAIN_CFG["LR_SCHEDULER"]["PATIENCE"]),
    min_lr=float(TRAIN_CFG["LR_SCHEDULER"]["MIN_LR"])
)

for epoch in range(TRAIN_CFG["EPOCHS"]):
    diffusion_model.train()
    total_loss = 0.0
    step = 0

    for chunk_dir in chunk_dirs:
        chunk_path = os.path.join(output_path, chunk_dir)
        dataset = load_from_disk(chunk_path)
        dataset.set_format(type="torch", columns=["image", "input_ids", "attention_mask", "report"])
        train_loader = DataLoader(dataset, batch_size=TRAIN_CFG["BATCH_SIZE"], shuffle=True, collate_fn=collate_fn)

        for batch in tqdm(train_loader, desc=f"Training on {chunk_dir}"):
            images = batch["image"].to(device)
            reports = batch["report"]

            with torch.no_grad():
                latents = vae.encode(images)

            noise = torch.randn_like(latents)
            pred_noise, target = diffusion_model(latents, noise, reports)

            loss = F.mse_loss(pred_noise, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step += 1
            wandb.log({"train/loss": loss.item(), "train/epoch": epoch + 1, "train/step": step})

    avg_train_loss = total_loss / step

    # --- Validation loop ---
    diffusion_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            images = batch["image"].to(device)
            reports = batch["report"]
            latents = vae.encode(images)
            noise = torch.randn_like(latents)
            pred_noise, target = diffusion_model(latents, noise, reports)
            loss = F.mse_loss(pred_noise, target)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    wandb.log({
        "epoch": epoch + 1,
        "train/avg_loss": avg_train_loss,
        "val/loss": avg_val_loss,
        "best_val_loss": best_val_loss
    })

    # --- Sample image logging ---
    with torch.no_grad():
        sample_report = [batch["report"][0]]
        z = diffusion_model.sample(sample_report, (1, MODEL_CFG["LATENT_DIM"], MODEL_CFG["LATENT_H"], MODEL_CFG["LATENT_W"]), device)
        generated = vae.decode(z).cpu()
        image = (generated[0].squeeze(0).numpy() * 255).astype("uint8")
        wandb.log({"sample_image": [wandb.Image(image, caption=sample_report[0])]})

    # --- Checkpointing based on val loss ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        no_improvement_epochs = 0
        torch.save(diffusion_model.unet.state_dict(), diffusion_ckpt_path)
        torch.save(vae.state_dict(), vae_ckpt_path)
        wandb.log({
            "best/epoch": epoch + 1,
            "best/val_loss": best_val_loss,
            "best/diffusion_checkpoint": diffusion_ckpt_path,
            "best/vae_checkpoint": vae_ckpt_path
        })
    else:
        no_improvement_epochs += 1

    scheduler.step(avg_val_loss)

    print(f"Epoch {epoch + 1}: val_loss = {avg_val_loss:.4f} (best: {best_val_loss:.4f})")

    if no_improvement_epochs >= early_stop_patience:
        print(f"Early stopping at epoch {epoch + 1} after {no_improvement_epochs} epochs without improvement.")
        break
