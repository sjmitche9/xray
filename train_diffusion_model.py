# --- train_diffusion_model.py ---
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_from_disk
from tqdm import tqdm
import yaml
import wandb
from transformers import AutoTokenizer, AutoModel
from models.vae_enhanced import EnhancedVAE
from models.diffusion import DiffusionModel
from torch.optim.lr_scheduler import ReduceLROnPlateau

# --- Load config.yaml ---
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

CONFIG = {
    "vae_checkpoint": config["MODEL"]["VAE_CHECKPOINT"],
    "val_dataset_path": os.path.join(config["DATASET"]["OUTPUT_PATH"], "val"),
    "batch_size": config["TRAINING"]["BATCH_SIZE"],
    "epochs": config["TRAINING"]["EPOCHS"],
    "lr": float(config["TRAINING"]["LEARNING_RATE"]),
    "latent_dim": config["MODEL"]["LATENT_DIM"],
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "run_name": config["WANDB"]["RUN_NAME_DIFFUSION"],
    "project": config["WANDB"]["PROJECT"],
    "checkpoint_path": config["MODEL"]["DIFFUSION_CHECKPOINT"],
    "lr_scheduler_mode": config["TRAINING"].get("LR_SCHEDULER_MODE", "min"),
    "lr_scheduler_factor": config["TRAINING"].get("LR_SCHEDULER_FACTOR", 0.5),
    "lr_scheduler_patience": config["TRAINING"].get("LR_SCHEDULER_PATIENCE", 2),
    "early_stopping_patience": config["TRAINING"].get("EARLY_STOPPING_PATIENCE", 5),
    "output_path": config["DATASET"]["OUTPUT_PATH"]
}

# --- Init Weights & Biases ---
wandb.init(project="xray", name="train_diffusion", config=config)

# --- Load tokenizer and text encoder ---
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
text_encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(CONFIG["device"])

# --- Collate function ---
def collate_fn(batch):
    return {
        "image": torch.stack([torch.tensor(item["image"], dtype=torch.float32) for item in batch]),
        "input_ids": torch.tensor([item["input_ids"] for item in batch]),
        "attention_mask": torch.tensor([item["attention_mask"] for item in batch]),
    }

# --- Load VAE ---
vae = EnhancedVAE(latent_dim=CONFIG["latent_dim"]).to(CONFIG["device"])
vae.load_state_dict(torch.load(CONFIG["vae_checkpoint"], map_location=CONFIG["device"]))
vae.eval()

# --- Load validation set ---
val_dataset = load_from_disk(CONFIG["val_dataset_path"])
val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=collate_fn)

# --- Initialize diffusion model ---
model = DiffusionModel(config).to(CONFIG["device"])
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])

scheduler = ReduceLROnPlateau(
    optimizer,
    mode=CONFIG["lr_scheduler_mode"],
    factor=CONFIG["lr_scheduler_factor"],
    patience=CONFIG["lr_scheduler_patience"]
)

# --- Training loop ---
best_val_loss = float("inf")
patience_counter = 0

for epoch in range(CONFIG["epochs"]):
    model.train()
    total_train_loss = 0.0

    chunk_id = 0
    while True:
        chunk_path = os.path.join(CONFIG["output_path"], f"train_chunk_{chunk_id}")

        if not os.path.exists(chunk_path):
            break

        if chunk_id == 1: # use this to train on one chunk for speed or debugging
            break

        train_dataset = load_from_disk(chunk_path)
        train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=collate_fn)

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} | Chunk {chunk_id}"):
            image = batch["image"].to(CONFIG["device"])
            input_ids = batch["input_ids"].to(CONFIG["device"])
            attention_mask = batch["attention_mask"].to(CONFIG["device"])

            with torch.no_grad():
                z = vae.encode(image).view(-1, CONFIG["latent_dim"], 16, 16)
                batch_size = z.shape[0]
                input_ids = input_ids[:batch_size]
                attention_mask = attention_mask[:batch_size]
                reports = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

            noise = torch.randn_like(z)
            pred_noise, target_noise = model(z, noise, reports)
            print("pred_noise std:", pred_noise.std().item(), "| noise std:", noise.std().item())
            loss = F.mse_loss(pred_noise, target_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        chunk_id += 1

    avg_train_loss = total_train_loss / max(1, chunk_id)
    wandb.log({"train/avg_loss": avg_train_loss, "epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"]})

    # --- Validation ---
    model.eval()
    total_val_loss = 0.0
    num_samples = 4  # log only a few to save time
    images = []

    with torch.no_grad():

        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
            image = batch["image"].to(CONFIG["device"])
            input_ids = batch["input_ids"].to(CONFIG["device"])
            attention_mask = batch["attention_mask"].to(CONFIG["device"])

            z = vae.encode(image)
            reports = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            noise = torch.randn_like(z)
            pred_noise, target_noise = model(z, noise, reports)

            val_loss = F.mse_loss(pred_noise, target_noise)
            total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        

        val_iter = iter(val_loader)
        batch = next(val_iter)

        input_ids = batch["input_ids"][:num_samples].to(CONFIG["device"])
        attention_mask = batch["attention_mask"][:num_samples].to(CONFIG["device"])

        # Sample latent z from diffusion process
        # report_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        # z_sample = model.sample(report_texts, latent_shape=(num_samples, CONFIG["latent_dim"], 16, 16), device=CONFIG["device"])
        # recon = vae.decode(z_sample).cpu().clamp(0, 1)

        # for i in range(num_samples):
        #     caption = report_texts[i].strip()
        #     images.append(wandb.Image(recon[i], caption=caption))

        # Sample latent z from diffusion process and decode intermediate steps
        report_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        z_sample, intermediate_recons = model.sample(
            report_texts,
            latent_shape=(num_samples, CONFIG["latent_dim"], 16, 16),
            device=CONFIG["device"],
            vae=vae,  # 🔥 Pass the VAE explicitly
            step_interval=50
        )

        # Log intermediate reconstructions
        for t_step, recon in intermediate_recons:
            for i in range(num_samples):
                caption = f"t={t_step} | {report_texts[i].strip()}"
                images.append(wandb.Image(recon[i], caption=caption))


    scheduler.step(avg_val_loss)

    wandb.log({
        "epoch": epoch + 1,
        "lr": optimizer.param_groups[0]["lr"],
        "train/avg_loss": avg_train_loss,
        "val/loss": avg_val_loss,
        "sampled_reconstructions": images
    })

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), CONFIG["checkpoint_path"])
        wandb.log({
            "best/epoch": epoch + 1,
            "best/checkpoint_path": CONFIG["checkpoint_path"]
        })
    else:
        patience_counter += 1
        print(f"No improvement. Patience: {patience_counter}/{CONFIG['early_stopping_patience']}")
        if patience_counter >= CONFIG["early_stopping_patience"]:
            print("Early stopping triggered.")
            break