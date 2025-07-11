import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_from_disk
from tqdm import tqdm
import wandb
from models.text_to_latent import TextToLatent
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.vae import VAE

# --- Load config ---
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

latent_path = os.path.join(config["DATASET"]["OUTPUT_PATH"], "latent")
batch_size = config["TRAINING"].get("TEXT_BATCH_SIZE", 64)
epochs = config["TRAINING"].get("EPOCHS", 20)
lr = float(config["TRAINING"].get("LEARNING_RATE", 1e-4))
latent_dim = config["MODEL"].get("LATENT_DIM", 64)
checkpoint_path = config["MODEL"].get("TEXT_TO_LATENT_CHECKPOINT", "checkpoints/text_to_latent.pt")
resume = config["TRAINING"].get("TEXT_TO_LATENT_RESUME", False)
early_stopping_patience = config["TRAINING"].get("EARLY_STOPPING_PATIENCE", 5)

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Validation set ---
val_dataset = load_from_disk(latent_path)["val"]
val_dataset.set_format(type="torch", columns=["text_embedding", "z_target"], output_all_columns=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

model = TextToLatent(input_dim=768, latent_dim=latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode=config["TRAINING"].get("LR_SCHEDULER_MODE", "min"),
    factor=config["TRAINING"].get("LR_SCHEDULER_FACTOR", 0.5),
    patience=config["TRAINING"].get("LR_SCHEDULER_PATIENCE", 2)
)


# --- Resume from checkpoint ---
start_epoch = 0
best_val_loss = float("inf")
patience_counter = 0
if resume and os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint.get("epoch", 0)
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    print(f"Resumed from checkpoint at epoch {start_epoch}")

# --- WandB ---
wandb.init(project="xray", name="text_to_latent", config=config)

# --- Training loop ---
for epoch in range(start_epoch, epochs):
    model.train()
    train_loss = 0
    train_batches = 0
    chunk_id = 0

    while True:
        chunk_path = os.path.join(latent_path, f"latent_train_chunk_{chunk_id}")
        if not os.path.exists(chunk_path):
            break

        chunk_dataset = load_from_disk(chunk_path)
        chunk_dataset.set_format(type="torch", columns=["text_embedding", "z_target"], output_all_columns=True)
        loader = DataLoader(chunk_dataset, batch_size=batch_size, shuffle=True)

        for batch in tqdm(loader, desc=f"Epoch {epoch+1} Chunk {chunk_id}"):
            x = batch["text_embedding"].to(device)
            y = batch["z_target"].to(device)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = F.mse_loss(y_pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        chunk_id += 1

    avg_train_loss = train_loss / train_batches

    # --- Validation ---
    model.eval()
    val_loss = 0
    images = []
    

    vae = VAE().to(device).eval()
    vae.load_state_dict(torch.load(config["MODEL"]["VAE_CHECKPOINT"], map_location=device))

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            x = batch["text_embedding"].to(device)
            y = batch["z_target"].to(device)
            y_pred = model(x)
            loss = F.mse_loss(y_pred, y)
            val_loss += loss.item()

            if i == 0:
                z_pred = y_pred[:8]
                z_true = y[:8]
                recon_pred = vae.decoder(z_pred).cpu()
                recon_true = vae.decoder(z_true).cpu()
                for j in range(len(recon_pred)):
                    pair = torch.cat([recon_true[j], recon_pred[j]], dim=2)

                    if "report" in batch:
                        caption = batch["report"][j]
                    else:
                        caption = f"Sample {j}"

                    images.append(wandb.Image(pair, caption=caption))

    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f} | Val Loss = {avg_val_loss:.6f}")
    scheduler.step(avg_val_loss)
    wandb.log({
        "lr": optimizer.param_groups[0]["lr"],
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "epoch": epoch + 1,
        "reconstructions": images
    })

    # --- Early stopping and checkpointing ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch + 1,
            "best_val_loss": best_val_loss
        }, checkpoint_path)
        print(f"Model improved. Saved checkpoint to {checkpoint_path}")
    else:
        patience_counter += 1
        print(f"No improvement. Patience: {patience_counter}/{early_stopping_patience}")
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break
