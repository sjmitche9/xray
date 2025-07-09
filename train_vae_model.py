import os
import torch
import yaml
import wandb
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
from datasets import load_from_disk
from models.vae import VAE

# --- Load config ---
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

output_path = config["DATASET"]["OUTPUT_PATH"]
batch_size = config["TRAINING"]["BATCH_SIZE"]
epochs = config["TRAINING"]["EPOCHS"]
learning_rate = float(config["TRAINING"]["LEARNING_RATE"])
vae_ckpt_path = config["MODEL"]["VAE_CHECKPOINT"]
resume_vae = config["MODEL"].get("VAE_RESUME", False)
early_stop_patience = config["TRAINING"].get("EARLY_STOP_PATIENCE", 5)

# --- Init Weights & Biases ---
wandb.init(project=config["WANDB"]["PROJECT"], name=config["WANDB"]["RUN_NAME_VAE"], config=config)

# --- Collate Function ---
def collate_fn(batch):
    return {
        "image": torch.stack([torch.tensor(item["image"]) for item in batch]),
        "input_ids": torch.tensor([item["input_ids"] for item in batch]),
        "attention_mask": torch.tensor([item["attention_mask"] for item in batch]),
    }

# --- VAE loss function ---
def vae_loss(recon, x, mu, logvar):
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kl

# --- Model ---
model = VAE(in_channels=1, latent_dim=config["MODEL"]["LATENT_DIM"])
model = model.cuda() if torch.cuda.is_available() else model
device = next(model.parameters()).device

if resume_vae and os.path.exists(vae_ckpt_path):
    model.load_state_dict(torch.load(vae_ckpt_path, map_location=device))

optimizer = Adam(model.parameters(), lr=learning_rate)

# --- Learning Rate Scheduler ---
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=float(config["TRAINING"]["LR_SCHEDULER"]["FACTOR"]),
    patience=int(config["TRAINING"]["LR_SCHEDULER"]["PATIENCE"]),
    min_lr=float(config["TRAINING"]["LR_SCHEDULER"]["MIN_LR"])
)

# --- Load validation set once ---
val_dataset = load_from_disk(os.path.join(output_path, "val"))
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

# --- Training loop ---
chunk_dirs = sorted([d for d in os.listdir(output_path) if d.startswith("train_chunk_")])

best_val_loss = float("inf")
no_improvement_epochs = 0

for epoch in range(epochs):
    model.train()
    print(f"Epoch {epoch+1}/{epochs}")
    epoch_loss = 0.0
    step = 0

    for chunk_dir in chunk_dirs:
        chunk_path = os.path.join(output_path, chunk_dir)
        train_dataset = load_from_disk(chunk_path)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        for batch in tqdm(train_loader, desc=f"Training on {chunk_dir}"):
            pixel_values = batch["image"].to(device)

            recon, mu, logvar = model(pixel_values)
            loss = vae_loss(recon, pixel_values, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            step += 1
            wandb.log({"train/loss": loss.item()})

    avg_train_loss = epoch_loss / step

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            pixel_values = batch["image"].to(device)

            recon, mu, logvar = model(pixel_values)
            loss = vae_loss(recon, pixel_values, mu, logvar)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    wandb.log({
        "val/loss": avg_val_loss,
        "epoch": epoch + 1,
        "train/avg_loss": avg_train_loss,
        "best_val_loss": best_val_loss
    })

    # --- Checkpointing ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        no_improvement_epochs = 0
        torch.save(model.state_dict(), vae_ckpt_path)
        wandb.log({
            "best/epoch": epoch + 1,
            "best/checkpoint_path": vae_ckpt_path,
            "best/val_loss": best_val_loss
        })
    else:
        no_improvement_epochs += 1

    # Step scheduler on validation loss
    scheduler.step(avg_val_loss)

    print(f"Validation Loss after epoch {epoch+1}: {avg_val_loss:.4f} (best: {best_val_loss:.4f})")

    if no_improvement_epochs >= early_stop_patience:
        print(f"Early stopping at epoch {epoch+1} — no improvement in {no_improvement_epochs} epochs.")
        break
