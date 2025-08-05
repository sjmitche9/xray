# train_diffusion_model.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_from_disk
from tqdm import tqdm
import yaml
import wandb
import os
from diffusers.models import AutoencoderKL
from dataloading import collate_fn
from models.unet import conditional_unet
from scheduler.ddpm_scheduler import ddpm_scheduler
from transformers import AutoTokenizer, AutoModel
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from pytorch_msssim import ssim as compute_ssim

def composite_loss(pred, target, alpha=1.0, beta=0.1, gamma=0.25):
    l1 = F.smooth_l1_loss(pred, target)
    cos = 1 - F.cosine_similarity(pred.flatten(1), target.flatten(1)).mean()
    std_penalty = 1.0 / (pred.std() + 1e-6)
    return alpha * l1 + beta * cos + gamma * std_penalty


def to_grayscale(img_tensor):
    return img_tensor.mean(dim=0, keepdim=True).repeat(3, 1, 1)

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
    "factor": float(config["TRAINING"]["LR_SCHEDULER"].get("FACTOR", .75)),
    "patience": float(config["TRAINING"]["LR_SCHEDULER"].get("PATIENCE", 3)),
    "min_lr": float(config["TRAINING"]["LR_SCHEDULER"].get("MIN_LR", 1e-7)),
    "sampling_steps": config["SCHEDULER"].get("SAMPLING_STEPS", 200),
    "context_dropout": float(config["TRAINING"].get("CONTEXT_DROPOUT_PROB", 0)),
}

if __name__ == "__main__":
    wandb.init(project=CONFIG["project"], name=CONFIG["run_name"], config=CONFIG)

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    text_encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(CONFIG["device"]).eval()

    scheduler = ddpm_scheduler(num_timesteps=CONFIG["sampling_steps"])
    model = conditional_unet(in_channels=CONFIG["latent_dim"], context_dim=768, context_dropout=CONFIG["context_dropout"]).to(CONFIG["device"])
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

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(CONFIG["device"])
    vae.eval()

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

        avg_train_loss = total_train_loss / total_batches if total_batches > 0 else 0.0

        model.eval()
        total_val_loss = 0.0
        val_pred_std_list = []
        val_cos_sim_list = []

        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")):
                if CONFIG["train_on_one_batch"] and i == 1:
                    break

                z = batch["z_target"].to(CONFIG["device"]).float()
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

            # z_clamped = ((z_sampled - z_sampled.mean()) / (z_sampled.std() + 1e-5)).clamp(-5, 5).float()
            z_clamped = z_sampled
            decoded = vae.decode(z_clamped).sample.clamp(0, 1).cpu()


            wandb_images = []
            print("[debug] z_sampled stats:")
            for i, z in enumerate(z_sampled[:4]):
                print(f"  z_sampled[{i}] mean: {z.mean():.5f}, std: {z.std():.5f}, min: {z.min():.5f}, max: {z.max():.5f}")

            ssim_scores, psnr_scores = [], []

            for i, (img, rep, real_z) in enumerate(zip(decoded, reports, sample_batch["z_target"][:4])):
                print(f"[debug] real_z[{i}] mean: {real_z.mean():.5f}, std: {real_z.std():.5f}, min: {real_z.min():.5f}, max: {real_z.max():.5f}")
                print(f"[debug] img[{i}] mean: {img.mean():.5f}, std: {img.std():.5f}, min: {img.min():.5f}, max: {img.max():.5f}")

                # Decode raw float32 z directly
                raw_decoded = vae.decode(real_z.unsqueeze(0).to(CONFIG["device"]).float()).sample[0].clamp(0, 1).cpu()
                print(f"[debug] raw_decoded[{i}] mean: {raw_decoded.mean():.5f}, std: {raw_decoded.std():.5f}, min: {raw_decoded.min():.5f}, max: {raw_decoded.max():.5f}")

                # Resize and format
                if raw_decoded.shape[0] != 1:
                    raw_decoded = raw_decoded.mean(dim=0, keepdim=True)  # [1, H, W]
                if img.shape[0] != 1:
                    img = img.mean(dim=0, keepdim=True)
                if raw_decoded.shape != img.shape:
                    raw_decoded = F.interpolate(raw_decoded.unsqueeze(0), size=img.shape[1:], mode="bilinear", align_corners=False)[0]

                # Compute SSIM and PSNR
                ssim_val = compute_ssim(raw_decoded.unsqueeze(0), img.unsqueeze(0), data_range=1.0).item()
                psnr_val = compute_psnr(img.squeeze().numpy(), raw_decoded.squeeze().numpy(), data_range=1.0)
                ssim_scores.append(ssim_val)
                psnr_scores.append(psnr_val)

                # Convert to grayscale for visualization
                raw_vis = to_grayscale(raw_decoded)
                img_vis = to_grayscale(img)
                side_by_side = torch.cat([raw_vis, img_vis], dim=2)
                wandb_images.append(wandb.Image(side_by_side.clamp(0, 1), caption=f"[recon | generated] {rep}"))

        print(f"Epoch {epoch+1} → train_loss: {avg_train_loss:.4f}, val_loss: {avg_val_loss:.4f}, val_std: {val_pred_std:.4f}, val_cos_sim: {val_cos_sim:.4f}")

        z_sampled_std = z_sampled.std().item()
        real_z_stds = [rz.std().item() for rz in sample_batch["z_target"][:4]]
        real_z_std_mean = sum(real_z_stds) / len(real_z_stds)
        z_std_ratio = z_sampled_std / (real_z_std_mean + 1e-8)

        wandb.log({
            "Debug/real_z_mean": real_z.mean().item(),
            "Debug/real_z_std": real_z.std().item(),
            "Debug/z_sampled_mean": z_sampled.mean().item(),
            "Debug/z_sampled_std": z_sampled_std,
            "Debug/real_z_std_mean": real_z_std_mean,
            "Debug/z_std_ratio": z_std_ratio,
            "Images/sampled_latent_images": wandb_images,
            "Schedule/epoch": epoch + 1,
            "Loss/train_loss": avg_train_loss,
            "Loss/val_loss": avg_val_loss,
            "Score/val_pred_std": val_pred_std,
            "Score/val_cos_sim": val_cos_sim,
            "Score/avg_ssim": sum(ssim_scores)/len(ssim_scores),
            "Score/avg_psnr": sum(psnr_scores)/len(psnr_scores),
            "Schedule/lr": optimizer.param_groups[0]['lr']
        })

        torch.cuda.empty_cache()

        scheduler_plateau.step(avg_val_loss)

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
