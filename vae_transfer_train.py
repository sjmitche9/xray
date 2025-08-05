import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_from_disk
from diffusers import AutoencoderKL
from torchvision.utils import make_grid
from pytorch_msssim import ssim as compute_ssim
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from tqdm import tqdm
import wandb
import yaml

def combined_loss(pred, target, alpha=1.0, beta=0.5):
    smooth_l1 = F.smooth_l1_loss(pred, target)
    ssim_loss = 1.0 - compute_ssim(pred, target, data_range=1.0, size_average=True)
    return alpha * smooth_l1 + beta * ssim_loss

def main():
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    CONFIG = {
        "data_path": config["DATASET"]["OUTPUT_PATH"],
        "checkpoint_path": config["MODEL"]["VAE_CHECKPOINT"],
        "batch_size": config["TRAINING"]["BATCH_SIZE"],
        "epochs": config["TRAINING"]["EPOCHS"],
        "lr": float(config["TRAINING"]["LEARNING_RATE"]),
        "project": config["WANDB"]["PROJECT"],
        "run_name": config["WANDB"]["RUN_NAME_VAE"],
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "train_on_one_batch": config["TRAINING"].get("TRAIN_ON_ONE_BATCH", False),
        "chunk_limit": config["TRAINING"].get("CHUNK_LIMIT", 36),
        "grad_accum_steps": config["TRAINING"].get("GRAD_ACCUM_STEPS", 1),
        "early_stop_patience": config["TRAINING"].get("EARLY_STOP_PATIENCE", 5),
        "min_save_epoch": config["TRAINING"].get("MIN_SAVE_EPOCH", 5),
        "kl_anneal_epochs": config["TRAINING"].get("KL_ANNEAL_EPOCHS", 10),
        "kl_max_weight": config["TRAINING"].get("KL_MAX_WEIGHT", 1.0),
        "scheduler_factor": float(config["TRAINING"]["LR_SCHEDULER"].get("FACTOR", 0.5)),
        "scheduler_patience": int(config["TRAINING"]["LR_SCHEDULER"].get("PATIENCE", 2)),
        "scheduler_min_lr": float(config["TRAINING"]["LR_SCHEDULER"].get("MIN_LR", 1e-6)),
    }

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(CONFIG["device"])
    vae.decoder.requires_grad_(True)
    vae.encoder.requires_grad_(True)
    vae.quant_conv.requires_grad_(False)
    vae.post_quant_conv.requires_grad_(True)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, vae.parameters()), lr=CONFIG["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=CONFIG["scheduler_factor"],
        patience=CONFIG["scheduler_patience"],
        min_lr=CONFIG["scheduler_min_lr"]
    )

    wandb.init(project=CONFIG["project"], name=CONFIG["run_name"], config=CONFIG)

    val_dataset = load_from_disk(os.path.join(CONFIG["data_path"], "val"))
    val_dataset.set_format("torch", columns=["image"])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(CONFIG["epochs"]):
        vae.train()
        total_loss = total_kl = total_recon = 0.0
        total_batches = 0
        ssim_scores, psnr_scores = [], []

        kl_weight = min(1.0, (epoch + 1) / CONFIG["kl_anneal_epochs"]) * CONFIG["kl_max_weight"]
        chunk_id = 0

        while chunk_id < CONFIG["chunk_limit"]:
            chunk_path = os.path.join(CONFIG["data_path"], f"train_chunk_{chunk_id}")
            if not os.path.exists(chunk_path):
                break

            dataset = load_from_disk(chunk_path)
            dataset.set_format("torch", columns=["image"])
            loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4)

            for i, batch in enumerate(tqdm(val_loader, desc=f"[Val] Epoch {epoch+1}")):
                if CONFIG["train_on_one_batch"] and i == 1:
                    break

                x = batch["image"].to(CONFIG["device"])
                if x.shape[1] == 1:
                    x = x.repeat(1, 3, 1, 1)

                latent_dist = vae.encode(x).latent_dist
                mu, logvar = latent_dist.mean, latent_dist.logvar
                z = latent_dist.sample()
                recon_x = vae.decode(z).sample

                recon_loss = combined_loss(recon_x, x)
                kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / (x.size(0) * x[0].numel())
                loss = recon_loss + kl_weight * kl

                loss.backward()
                if (i + 1) % CONFIG["grad_accum_steps"] == 0 or (i + 1) == len(loader):
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item()
                total_kl += kl.item()
                total_recon += recon_loss.item()
                total_batches += 1

                for j in range(min(4, x.size(0))):
                    ssim_scores.append(compute_ssim(recon_x[j:j+1], x[j:j+1], data_range=1.0).item())
                    psnr_scores.append(compute_psnr(x[j].detach().cpu().numpy(), recon_x[j].detach().cpu().numpy(), data_range=1.0))

            chunk_id += 1

        with torch.no_grad():
            z_std = z.std().item()
            z_mean = z.mean().item()
            grad_norm = sum(p.grad.norm(2).item()**2 for p in vae.parameters() if p.grad is not None)**0.5

        avg_loss = total_loss / total_batches
        avg_kl = total_kl / total_batches
        avg_recon = total_recon / total_batches
        avg_ssim = sum(ssim_scores) / len(ssim_scores)
        avg_psnr = sum(psnr_scores) / len(psnr_scores)

        vae.eval()
        val_loss = val_kl = val_recon = 0.0
        val_ssim, val_psnr = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[Val] Epoch {epoch+1}"):
                if CONFIG["train_on_one_batch"] and i == 1:
                    break
                x = batch["image"].to(CONFIG["device"])
                if x.shape[1] == 1:
                    x = x.repeat(1, 3, 1, 1)

                latent_dist = vae.encode(x).latent_dist
                mu, logvar = latent_dist.mean, latent_dist.logvar
                z = latent_dist.sample()
                recon_x = vae.decode(z).sample

                recon_loss = combined_loss(recon_x, x)
                kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / (x.size(0) * x[0].numel())
                loss = recon_loss + kl_weight * kl

                val_loss += loss.item()
                val_kl += kl.item()
                val_recon += recon_loss.item()

                for j in range(min(4, x.size(0))):
                    val_ssim.append(compute_ssim(recon_x[j:j+1], x[j:j+1], data_range=1.0).item())
                    val_psnr.append(compute_psnr(x[j].detach().cpu().numpy(), recon_x[j].detach().cpu().numpy(), data_range=1.0))

        val_loss /= len(val_loader)
        val_kl /= len(val_loader)
        val_recon /= len(val_loader)
        val_ssim = sum(val_ssim) / len(val_ssim) if val_ssim else 0.0
        val_psnr = sum(val_psnr) / len(val_psnr) if val_psnr else 0.0

        sample_x = next(iter(val_loader))["image"][:4].to(CONFIG["device"])
        if sample_x.shape[1] == 1:
            sample_x = sample_x.repeat(1, 3, 1, 1)
        with torch.no_grad():
            recon = vae(sample_x).sample.clamp(0, 1)
        panel = make_grid(torch.cat([sample_x, recon], dim=0), nrow=4)

        wandb.log({
            "Loss/train_total": avg_loss,
            "Loss/train_recon": avg_recon,
            "Loss/train_kl": avg_kl,
            "KL/weight": kl_weight,
            "KL/raw": avg_kl,
            "Recon/raw_loss": avg_recon,
            "Grad/norm": grad_norm,
            "Z/std": z_std,
            "Z/mean": z_mean,
            "Score/train_ssim": avg_ssim,
            "Score/train_psnr": avg_psnr,
            "Loss/val_total": val_loss,
            "Loss/val_recon": val_recon,
            "Loss/val_kl": val_kl,
            "Score/val_ssim": val_ssim,
            "Score/val_psnr": val_psnr,
            "Images/val_reconstructions": wandb.Image(panel)
        })

        print(f"\n[epoch {epoch+1}]")
        print(f"Train Loss:     {avg_loss:.4f}")
        print(f"Train Recon:    {avg_recon:.4f}")
        print(f"Train KL:       {avg_kl:.4f}")
        print(f"KL Weight:      {kl_weight:.4f}")
        print(f"Train SSIM:     {avg_ssim:.4f}")
        print(f"Train PSNR:     {avg_psnr:.4f}")
        print(f"Z Mean:         {z_mean:.4f}")
        print(f"Z Std:          {z_std:.4f}")
        print(f"Grad Norm:      {grad_norm:.4f}")
        print(f"Val Loss:       {val_loss:.4f}")
        print(f"Val Recon:      {val_recon:.4f}")
        print(f"Val KL:         {val_kl:.4f}")
        print(f"Val SSIM:       {val_ssim:.4f}")
        print(f"Val PSNR:       {val_psnr:.4f}\n")

        scheduler.step(val_loss)

        if val_loss < best_val_loss and (epoch + 1) >= CONFIG["min_save_epoch"]:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(vae.state_dict(), CONFIG["checkpoint_path"])
            artifact = wandb.Artifact("best_vae_model", type="model")
            artifact.add_file(CONFIG["checkpoint_path"])
            wandb.log_artifact(artifact)
            print(f"[info] Saved new best model at epoch {epoch+1}")
        elif (epoch + 1) >= CONFIG["min_save_epoch"]:
            patience_counter += 1
            print(f"[info] No improvement. Patience: {patience_counter}/{CONFIG['early_stop_patience']}")
            if patience_counter >= CONFIG["early_stop_patience"]:
                print("[info] Early stopping triggered.")
                break

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
