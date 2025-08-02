# train_diffusion_model.py (updated for latent training + early stopping + wandb checkpoint + image logging)
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_from_disk
from tqdm import tqdm
import yaml
import wandb

from dataloading import collate_fn
from models.unet import conditional_unet
from scheduler.ddpm_scheduler import ddpm_scheduler
from transformers import AutoTokenizer, AutoModel

def composite_loss(pred, target, alpha=1.0, beta=0.5, gamma=0.1):
    l1 = F.smooth_l1_loss(pred, target)
    cos = 1 - F.cosine_similarity(pred.flatten(1), target.flatten(1)).mean()
    std_penalty = 1.0 / (pred.std() + 1e-6)
    return alpha * l1 + beta * cos + gamma * std_penalty

#--- Load config.yaml ---
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

CONFIG = {
    "train_on_one_batch": config["TRAINING"].get("TRAIN_ON_ONE_BATCH", False),
    "latent_path": config["DATASET"]["LATENT_OUTPUT_PATH"],
    "val_path": os.path.join(config["DATASET"]["LATENT_OUTPUT_PATH"], "latent_val"),
    "batch_size": config["TRAINING"]["BATCH_SIZE"],
    "epochs": config["TRAINING"]["EPOCHS"],
    "lr": float(config["TRAINING"]["LEARNING_RATE"]),
    "latent_dim": config["MODEL"]["LATENT_DIM"],
    "latent_h": config["MODEL"]["LATENT_H"],
    "latent_w": config["MODEL"]["LATENT_W"],
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "run_name": config["WANDB"]["RUN_NAME_DIFFUSION"],
    "project": config["WANDB"]["PROJECT"],
    "checkpoint_path": config["MODEL"]["DIFFUSION_CHECKPOINT"],
    "guidance_scale": config["TRAINING"]["GUIDANCE_SCALE"],
    "chunk_limit": config["TRAINING"].get("CHUNK_LIMIT", 1000),
    "early_stop_patience": config["TRAINING"].get("EARLY_STOP_PATIENCE", 5),
    "min_save_epoch": config["TRAINING"].get("MIN_SAVE_EPOCH", 5),
    "factor": config["TRAINING"]["LR_SCHEDULER"].get("FACTOR", .75),
    "patience": config["TRAINING"]["LR_SCHEDULER"].get("PATIENCE", 3),
    "min_lr": config["TRAINING"]["LR_SCHEDULER"].get("MIN_LR", 1e-7)
}

if __name__ == "__main__":
    wandb.init(project=CONFIG["project"], name=CONFIG["run_name"], config=CONFIG)

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    text_encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(CONFIG["device"]).eval()

    scheduler = ddpm_scheduler()
    model = conditional_unet(in_channels=CONFIG["latent_dim"], context_dim=768).to(CONFIG["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=CONFIG["factor"],
        patience=CONFIG["patience"],
        min_lr=CONFIG["min_lr"]
    )

    val_loader = DataLoader(
        load_from_disk(CONFIG["val_path"]),
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_train_loss = 0.0
        total_batches = 0

        chunk_id = 0
        while chunk_id < CONFIG["chunk_limit"]:
            chunk_path = os.path.join(CONFIG["latent_path"], f"latent_train_chunk_{chunk_id}")
            if not os.path.exists(chunk_path):
                break

            train_data = load_from_disk(chunk_path)
            train_loader = DataLoader(
                train_data,
                batch_size=CONFIG["batch_size"],
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=4,
                pin_memory=True
            )

            for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Chunk {chunk_id}")):
                if CONFIG["train_on_one_batch"] and i == 1:
                    break
                z = batch["z_target"].to(CONFIG["device"]).float()
                reports = batch["report"]
                tokens = tokenizer(reports, padding="max_length", truncation=True, max_length=128, return_tensors="pt").to(CONFIG["device"])
                ctx = text_encoder(**tokens).last_hidden_state.mean(dim=1)

                t = torch.randint(0, scheduler.num_timesteps, (z.size(0),), device=CONFIG["device"]).long()
                noise = torch.randn_like(z)
                z_noisy = scheduler.add_noise(z, noise, t)
                pred_noise = model(z_noisy, t, ctx)

                loss = composite_loss(pred_noise, noise)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                total_batches += 1

            chunk_id += 1

        avg_train_loss = total_train_loss / total_batches if total_batches > 0 else print("[warn] No training data found.") or 0

        model.eval()
        total_val_loss = 0.0
        val_pred_std_list = []
        val_cos_sim_list = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")):
                if CONFIG["train_on_one_batch"] and i == 1:
                    break
                z = batch["z_target"].to(CONFIG["device"])
                reports = batch["report"]
                tokens = tokenizer(reports, padding="max_length", truncation=True, max_length=128, return_tensors="pt").to(CONFIG["device"])
                ctx = text_encoder(**tokens).last_hidden_state.mean(dim=1)

                t = torch.randint(0, scheduler.num_timesteps, (z.size(0),), device=CONFIG["device"]).long()
                noise = torch.randn_like(z)
                z_noisy = scheduler.add_noise(z, noise, t)
                pred = model(z_noisy, t, ctx)

                val_loss = composite_loss(pred, noise)
                total_val_loss += val_loss.item()
                val_pred_std_list.append(pred.std().item())
                cos_sim = F.cosine_similarity(pred.flatten(1), noise.flatten(1)).mean().item()
                val_cos_sim_list.append(cos_sim)

        avg_val_loss = total_val_loss / len(val_loader)
        val_pred_std = sum(val_pred_std_list) / len(val_pred_std_list)
        val_cos_sim = sum(val_cos_sim_list) / len(val_cos_sim_list)

        # --- Sample and log reconstructions ---
        with torch.no_grad():
            sample_batch = next(iter(val_loader))
            reports = sample_batch["report"][:4]
            tokens = tokenizer(reports, padding="max_length", truncation=True, max_length=128, return_tensors="pt").to(CONFIG["device"])
            ctx = text_encoder(**tokens).last_hidden_state.mean(dim=1)
            null_ctx = torch.zeros_like(ctx)

            z_sampled = torch.randn((4, CONFIG["latent_dim"], CONFIG["latent_h"], CONFIG["latent_w"]), device=CONFIG["device"])
            for t_gen in reversed(range(scheduler.num_timesteps)):
                t_tensor = torch.full((z_sampled.size(0),), t_gen, device=CONFIG["device"], dtype=torch.long)
                pred_cond = model(z_sampled, t_tensor, ctx)
                pred_uncond = model(z_sampled, t_tensor, null_ctx)
                pred = pred_uncond + CONFIG["guidance_scale"] * (pred_cond - pred_uncond)
                alpha_t = scheduler.alpha[t_gen]
                alpha_hat_t = scheduler.alpha_hat[t_gen]
                beta_t = scheduler.beta[t_gen]
                z_sampled = (1 / torch.sqrt(alpha_t)) * (z_sampled - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * pred)
                if t_gen > 0:
                    z_sampled += torch.sqrt(beta_t) * torch.randn_like(z_sampled)

            images = (z_sampled.cpu().clamp(-5, 5) + 5) / 10
            wandb_images = []
            for img, rep, real_z in zip(images, reports, sample_batch["z_target"][:4]):
                recon = real_z.unsqueeze(0).unsqueeze(0).float()
                recon_resized = F.interpolate(recon, size=img.unsqueeze(0).shape[-2:], mode="bilinear", align_corners=False)[0, 0]
                comparison = torch.cat([recon_resized, img[0].cpu()], dim=1).numpy()
                wandb_images.append(wandb.Image(comparison, caption=rep))

        print(f"Epoch {epoch+1} → train_loss: {avg_train_loss:.4f}, val_loss: {avg_val_loss:.4f}, val_std: {val_pred_std:.4f}, val_cos_sim: {val_cos_sim:.4f}")

        current_lr = optimizer.param_groups[0]['lr']
        scheduler_plateau.step(avg_val_loss)

        wandb.log({
            "Images/sampled_latent_images": wandb_images,
            "Schedule/epoch": epoch + 1,
            "Loss/train_loss": avg_train_loss,
            "Loss/val_loss": avg_val_loss,
            "Score/val_pred_std": val_pred_std,
            "Score/val_cos_sim": val_cos_sim,
            "Schedule/lr": current_lr
        })

        if avg_val_loss < best_val_loss and (epoch + 1) >= CONFIG["min_save_epoch"]:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), CONFIG["checkpoint_path"])
            artifact = wandb.Artifact("best_model", type="model")
            artifact.add_file(CONFIG["checkpoint_path"])
            wandb.log_artifact(artifact)
            print(f"[info] Saved new best model at epoch {epoch+1}")
        elif (epoch + 1) >= CONFIG["min_save_epoch"]:
            patience_counter += 1
            print(f"[info] No improvement. Patience: {patience_counter}/{CONFIG['early_stop_patience']}")
            if patience_counter >= CONFIG["early_stop_patience"]:
                print("[info] Early stopping triggered.")
                break
