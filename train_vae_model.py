# --- train_vae_model.py ---
import os
import torch
import yaml
import wandb
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
from datasets import load_from_disk
from models.vae_enhanced import EnhancedVAE
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_msssim import ssim as ssim_loss_fn


def vae_loss_function(recon_x, x, mu, logvar, beta=1.0, ssim_weight=0.5):
    original_logvar = logvar.detach().clone()  # Save before clamp
    logvar = torch.clamp(logvar, min=-4.0)
    mse_loss = F.mse_loss(recon_x, x, reduction='mean')
    ssim_loss = 1 - ssim_loss_fn(recon_x, x, data_range=1.0, size_average=True)
    recon_loss = (1 - ssim_weight) * mse_loss + ssim_weight * ssim_loss
    # kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    raw_kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    free_nats = 1  # you can tune this
    kl_div = torch.mean(torch.maximum(raw_kl, torch.tensor(free_nats, device=logvar.device)))

    # Debug output
    print("raw logvar mean (before clamp):", original_logvar.mean().item())
    print("clamped logvar mean:", logvar.mean().item())
    print("KL divergence:", kl_div.item())

    return recon_loss + beta * kl_div, recon_loss, kl_div

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

output_path = config["DATASET"]["OUTPUT_PATH"]
batch_size = config["TRAINING"]["BATCH_SIZE"]
epochs = config["TRAINING"]["EPOCHS"]
learning_rate = float(config["TRAINING"]["LEARNING_RATE"])
beta_max = config["TRAINING"].get("BETA", 1.0)
warmup_epochs = config["TRAINING"].get("WARMUP_EPOCHS", 10)
early_stopping_patience = config["TRAINING"].get("EARLY_STOPPING_PATIENCE", 5)
checkpoint_path = config["MODEL"].get("VAE_CHECKPOINT", "checkpoints/vae.pt")
latent_dim = config["MODEL"]["LATENT_DIM"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vae = EnhancedVAE(latent_dim=latent_dim).to(device)
optimizer = Adam(vae.parameters(), lr=learning_rate)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode=config["TRAINING"].get("LR_SCHEDULER_MODE", "min"),
    factor=config["TRAINING"].get("LR_SCHEDULER_FACTOR", 0.5),
    patience=config["TRAINING"].get("LR_SCHEDULER_PATIENCE", 2)
)


val_dataset = load_from_disk(os.path.join(output_path, "val"))
val_dataset.set_format(type="torch", columns=["image"])
val_loader = DataLoader(val_dataset, batch_size=batch_size)

wandb.init(project="xray", name="train_vae", config=config)

best_val_loss = float("inf")
patience_counter = 0

for epoch in range(epochs):
    beta = beta_max * min(1.0, epoch / warmup_epochs)
    vae.train()
    train_loss = train_recon = train_kl = total_batches = 0

    chunk_id = 0

    while True:
        train_chunk_path = os.path.join(output_path, f"train_chunk_{chunk_id}")

        if not os.path.exists(train_chunk_path):
            break

        if chunk_id == 1: # use this to only load five chunks
            break

        train_dataset = load_from_disk(train_chunk_path)
        train_dataset.set_format(type="torch", columns=["image"])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} | Chunk {chunk_id}"):
            x = batch["image"].to(device)
            optimizer.zero_grad()
            recon, mu, logvar = vae(x)
            loss, recon_loss, kl_loss = vae_loss_function(recon, x, mu, logvar, beta)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_recon += recon_loss.item()
            train_kl += kl_loss.item()
            total_batches += 1

        chunk_id += 1

    train_loss /= total_batches
    train_recon /= total_batches
    train_kl /= total_batches

    vae.eval()
    val_loss = val_recon = val_kl = 0
    x_sample = recon_sample = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            x = batch["image"].to(device)
            recon, mu, logvar = vae(x)

            # if batch_idx == 0:
                # print("mu (mean of batch):", mu.mean().item())
                # print("logvar (mean of batch):", logvar.mean().item())

            loss, recon_loss, kl_loss = vae_loss_function(recon, x, mu, logvar, beta)
            val_loss += loss.item()
            val_recon += recon_loss.item()
            val_kl += kl_loss.item()

            if batch_idx == 0:
                x_sample = x[:8].cpu()
                recon_sample = recon[:8].clamp(0, 1).cpu()

    val_loss /= len(val_loader)
    val_recon /= len(val_loader)
    val_kl /= len(val_loader)

    images = []
    if x_sample is not None and recon_sample is not None:
        for i in range(len(x_sample)):
            combined = torch.cat([x_sample[i], recon_sample[i]], dim=2)
            images.append(wandb.Image(combined, caption=f"Sample {i}"))

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Recon: {train_recon:.4f} | KL: {train_kl:.4f}")
    print(f"Validation Loss: {val_loss:.4f} | Recon: {val_recon:.4f} | KL: {val_kl:.4f}")

    kl_ratio = train_kl / (train_recon + 1e-8)

    scheduler.step(val_loss)
    wandb.log({
        "lr": optimizer.param_groups[0]["lr"],
        "kl_ratio": kl_ratio,
        "reconstructions": images,
        "train_loss": train_loss,
        "train_recon": train_recon,
        "train_kl": train_kl,
        "val_loss": val_loss,
        "val_recon": val_recon,
        "val_kl": val_kl,
        "epoch": epoch + 1
    })

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(vae.state_dict(), checkpoint_path)
        print(f"Model improved. Saved checkpoint to {checkpoint_path}")
    else:
        patience_counter += 1
        print(f"No improvement. Patience: {patience_counter}/{early_stopping_patience}")
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break
