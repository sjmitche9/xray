# lora_unet_transfer_train.py
import os
import torch
import yaml
import wandb
from tqdm import tqdm
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from models.lora_unet_wrapper import LoRAUNetWrapper
from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from peft import get_peft_model, LoraConfig
from accelerate import Accelerator
from pytorch_msssim import ssim
import torch.nn.functional as F

# ----------------- helpers -----------------
def composite_loss(pred, target, ssim_weight=0.0, beta=1.0):
    sl1 = torch.nn.functional.smooth_l1_loss(pred, target, beta=beta)
    # normalize only for SSIM calc (detached mins/maxes to avoid autograd bloat)
    pmn, pmx = pred.detach().min(), pred.detach().max()
    tmn, tmx = target.detach().min(), target.detach().max()
    pred_n = (pred - pmn) / (pmx - pmn + 1e-5)
    targ_n = (target - tmn) / (tmx - tmn + 1e-5)
    ssim_loss = 1.0 - ssim(pred_n, targ_n, data_range=1.0, size_average=True)
    return sl1 + ssim_weight * ssim_loss, sl1.item(), ssim_loss.item()


def decode_latents(latents, vae, scale=0.18215):
    # Proper SD-style decode; DO NOT renorm/clamp latents themselves.
    with torch.no_grad():
        latents = latents.to(next(vae.parameters()).device)
        images = vae.decode(latents / scale).sample.clamp(0, 1)
    return images

def to_grayscale(img_tensor):
    img = img_tensor.mean(dim=0, keepdim=True).repeat(3, 1, 1)
    return (img * 255).clamp(0, 255).byte()

def apply_cfg_dropout(ctx, p_drop):
    if p_drop <= 0.0: return ctx, 0.0
    if p_drop >= 1.0: return torch.zeros_like(ctx), 1.0
    B = ctx.size(0)
    mask = (torch.rand(B, device=ctx.device) < p_drop).float().view(B, 1, 1)
    return ctx * (1.0 - mask), mask.mean().item()

def iter_lora_params(model):
    # LoRA params typically include "lora_" or ".lora"
    for n, p in model.named_parameters():
        if p.requires_grad and ("lora" in n.lower()):
            yield p

def grad_global_norm(params):
    total = torch.tensor(0.0, device="cpu")
    for p in params:
        if p.grad is not None:
            g = p.grad.detach().float()
            total += (g * g).sum().cpu()
    return float(total.sqrt().item())

def sample_from_model(unet, vae, tokenizer, text_encoder, scheduler, device, reports, guidance_scale=7.5, scale=0.18215):
    with torch.no_grad():
        tokens = tokenizer(list(reports), padding="max_length", truncation=True, max_length=128, return_tensors="pt").to(device)
        ctx_cond = text_encoder(**tokens).last_hidden_state
        ctx_uncond = torch.zeros_like(ctx_cond)
        B = len(reports)
        z = torch.randn((B, 4, 32, 32), device=device)
        timesteps = scheduler.timesteps.to(device)
        for t in timesteps:
            tt = torch.full((B,), t, device=device, dtype=torch.long)
            eps_c = unet(z, tt, ctx_cond).sample
            eps_u = unet(z, tt, ctx_uncond).sample
            eps = eps_u + guidance_scale * (eps_c - eps_u)
            z = scheduler.step(model_output=eps, timestep=t, sample=z).prev_sample
            del tt, eps_c, eps_u, eps
        decoded = vae.decode(z / scale).sample.clamp(0, 1)
        del z, tokens, ctx_cond, ctx_uncond
    return decoded

# ----------------- main -----------------
def main():
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    accelerator = Accelerator()
    device = accelerator.device

    # ---- config knobs ----
    SCALE            = float(config.get("MODEL", {}).get("LATENT_SCALE", 0.18215))
    latent_path      = config["DATASET"]["LATENT_OUTPUT_PATH"]
    batch_size       = config["TRAINING"]["BATCH_SIZE"]
    grad_accum_steps = config["TRAINING"].get("GRAD_ACCUM_STEPS", 1)
    max_grad_norm    = config["TRAINING"].get("MAX_GRAD_NORM", 1.0)
    train_on_three   = config["TRAINING"].get("TRAIN_ON_THREE_BATCHES", False)
    chunk_limit      = config["TRAINING"].get("CHUNK_LIMIT", 10)
    ssim_weight      = config["TRAINING"].get("SSIM_WEIGHT", 0.0)
    beta             = config["TRAINING"].get("BETA", 1.0)
    epochs           = config["TRAINING"]["EPOCHS"]
    cfg_dropout      = float(config["TRAINING"].get("CFG_DROPOUT", config["TRAINING"].get("CONTEXT_DROPOUT_PROB", 0.1)))
    guidance_scale   = float(config["TRAINING"].get("GUIDANCE_SCALE", 7.5))

    # ---- text encoder ----
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    text_encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)
    for p in text_encoder.parameters(): p.requires_grad = False

    # ---- UNet + LoRA ----
    unet_base = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="unet", cross_attention_dim=768
    )
    lora_config = LoraConfig(
        r=config["TRAINING"]["LORA_R"],
        lora_alpha=config["TRAINING"]["LORA_ALPHA"],
        target_modules=["to_q", "to_k", "to_v"],
        lora_dropout=config["TRAINING"]["LORA_DROPOUT"],
        bias="none",
    )
    unet_lora = get_peft_model(unet_base, lora_config)
    unet = LoRAUNetWrapper(unet_lora, context_dim_in=768, context_dim_out=768)
    unet = unet.to(device)

    # ---- VAE ----
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
    vae.load_state_dict(torch.load(config["MODEL"]["VAE_CHECKPOINT"], map_location=device))
    vae = vae.to(device).eval()

    # ---- noise scheduler ----
    scheduler = DDPMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    scheduler.config.num_train_timesteps = 1000
    scheduler.set_timesteps(100, device=device)  # used for val preview/sampling

    # ---- opt ----
    optimizer = torch.optim.AdamW(unet.parameters(), lr=float(config["TRAINING"]["LEARNING_RATE"]))
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor=float(config["TRAINING"]["LR_SCHEDULER"]["FACTOR"]),
        patience=int(config["TRAINING"]["LR_SCHEDULER"]["PATIENCE"]),
        min_lr=float(config["TRAINING"]["LR_SCHEDULER"]["MIN_LR"])
    )

    # accelerate prepare
    unet, optimizer = accelerator.prepare(unet, optimizer)

    wandb.init(project=config["WANDB"]["PROJECT"], name=config["WANDB"]["RUN_NAME_LORA_UNET"], config=config)

    best_val_loss, patience_counter = float('inf'), 0
    logged_scale_sanity = False  # log once

    for epoch in range(epochs):
        unet.train()
        epoch_loss = 0.0
        num_batches = 0
        observed_drop = []

        # ablation logs
        train_pred_std_sum, train_cos_sum = 0.0, 0.0
        lora_grad_norm_sum, lora_grad_norm_count = 0.0, 0

        # token lengths (train)
        train_tok_lens = []

        # -------- train over chunks --------
        chunk_id = 0
        while True:
            chunk_path = os.path.join(latent_path, f"latent_train_chunk_{chunk_id}")
            if not os.path.exists(chunk_path) or chunk_id >= chunk_limit: break

            dataset = load_from_disk(chunk_path).with_format("torch", columns=["z_target", "report"])
            loader = accelerator.prepare(DataLoader(dataset, batch_size=batch_size, shuffle=True))

            for i, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1} Chunk {chunk_id}")):
                if train_on_three and i == 3: break

                z = batch["z_target"].to(device).float()
                reports = batch["report"]
                noise = torch.randn_like(z)
                t = torch.randint(0, scheduler.config.num_train_timesteps, (z.size(0),), device=device).long()
                z_noisy = scheduler.add_noise(z, noise, t)

                tokens = tokenizer(list(reports), padding="max_length", truncation=True, max_length=128,
                                   return_tensors="pt", return_length=True).to(device)
                
                ctx = text_encoder(**tokens).last_hidden_state
                ctx, drop_ratio = apply_cfg_dropout(ctx, cfg_dropout)
                observed_drop.append(drop_ratio)

                # token length stats (true lengths pre-pad if available)
                if "length" in tokens:
                    train_tok_lens += [int(x) for x in tokens["length"]]
                else:
                    train_tok_lens += [int(m.sum().item()) for m in tokens["attention_mask"]]

                with accelerator.autocast():
                    pred = unet(z_noisy, t, ctx).sample
                    loss, sl1_val, ssim_val = composite_loss(pred, noise, ssim_weight=ssim_weight, beta=beta)
                    loss = loss / grad_accum_steps

                accelerator.backward(loss)

                # quick stats
                with torch.no_grad():
                    train_pred_std_sum += float(pred.float().std().item())
                    train_cos_sum += float(F.cosine_similarity(pred.flatten(1), noise.flatten(1)).mean().item())

                step_now = ((i + 1) % grad_accum_steps == 0) or ((i + 1) == len(loader))
                if step_now:
                    # LoRA-only grad norm (pre-clip)
                    gn = grad_global_norm(list(iter_lora_params(unet)))
                    lora_grad_norm_sum += gn
                    lora_grad_norm_count += 1

                    accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item() * grad_accum_steps
                num_batches += 1

                del z, reports, noise, t, z_noisy, tokens, ctx, pred, loss
                torch.cuda.empty_cache()

            chunk_id += 1

        avg_train_loss = epoch_loss / max(1, num_batches)
        avg_drop_ratio = float(sum(observed_drop) / max(1, len(observed_drop)))
        avg_train_pred_std = train_pred_std_sum / max(1, num_batches)
        avg_train_cos = train_cos_sum / max(1, num_batches)
        avg_lora_gn = (lora_grad_norm_sum / max(1, lora_grad_norm_count))

        # -------- validation --------
        val_path = os.path.join(latent_path, "latent_val")
        val_dataset = load_from_disk(val_path).with_format("torch", columns=["z_target", "report"])
        val_loader = accelerator.prepare(DataLoader(val_dataset, batch_size=batch_size))

        unet.eval()
        val_loss = val_l1_total = val_ssim_total = 0.0
        val_batches = 0
        val_images = []

        val_pred_std_sum, val_cos_sum = 0.0, 0.0
        t_vals = []
        val_tok_lens = []
        val_latent_means = []
        val_latent_stds = []

        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")):
                if train_on_three and i == 3: break

                z = batch["z_target"].to(device).float()
                reports = batch["report"]
                noise = torch.randn_like(z)

                # keep a record of latent stats for sanity
                val_latent_means.append(float(z.mean().item()))
                val_latent_stds.append(float(z.std().item()))

                # pick timesteps from inference grid
                ti = torch.randint(0, len(scheduler.timesteps), (z.size(0),), device=device).long()
                t_infer = scheduler.timesteps[ti]  # [B]
                z_noisy = scheduler.add_noise(z, noise, t_infer)

                tokens = tokenizer(list(reports), padding="max_length", truncation=True, max_length=128,
                                   return_tensors="pt", return_length=True).to(device)
                ctx = text_encoder(**tokens).last_hidden_state

                pred = unet(z_noisy, t_infer, ctx).sample
                loss, sl1_val_each, ssim_val_each = composite_loss(pred, noise, ssim_weight=ssim_weight, beta=beta)

                val_loss += loss.item()
                val_l1_total += sl1_val_each
                val_ssim_total += ssim_val_each
                val_batches += 1

                # quick stats
                val_pred_std_sum += float(pred.float().std().item())
                val_cos_sum += float(F.cosine_similarity(pred.flatten(1), noise.flatten(1)).mean().item())
                t_vals += [int(x) for x in t_infer.tolist()]

                # token lengths
                if "length" in tokens:
                    val_tok_lens += [int(x) for x in tokens["length"]]
                else:
                    val_tok_lens += [int(m.sum().item()) for m in tokens["attention_mask"]]

                # preview: (a) direct decode of z, (b) one-step recon (will look rough), caption shows t
                if len(val_images) < 3:
                    recon_latent = scheduler.step(
                        model_output=pred[:1], timestep=t_infer[:1], sample=z_noisy[:1]
                    ).prev_sample

                    # ---- scale sanity: decode two ways once (first epoch only) ----
                    if not logged_scale_sanity:
                        img_s1 = vae.decode((z[:1]) / 1.0).sample.clamp(0, 1)[0].detach().cpu()
                        img_sE = vae.decode((z[:1]) / SCALE).sample.clamp(0, 1)[0].detach().cpu()
                        val_images.append(wandb.Image(
                            torch.cat([to_grayscale(img_s1), 255*torch.ones((3, img_s1.shape[1], 1), dtype=torch.uint8),
                                       to_grayscale(img_sE)], dim=2),
                            caption=f"scale sanity: left=1.0, right={SCALE}"
                        ))
                        logged_scale_sanity = True

                    orig_img = decode_latents(z[:1], vae, scale=SCALE)[0].detach().cpu()
                    recon_img = decode_latents(recon_latent, vae, scale=SCALE)[0].detach().cpu()
                    H = min(orig_img.shape[1], recon_img.shape[1]); W = min(orig_img.shape[2], recon_img.shape[2])
                    orig_img, recon_img = to_grayscale(orig_img[:, :H, :W]), to_grayscale(recon_img[:, :H, :W])
                    panel = torch.cat([orig_img, 255*torch.ones_like(orig_img[:, :, :1]), recon_img], dim=2)
                    t_scalar = int(t_infer[0].item())
                    val_images.append(wandb.Image(panel, caption=f"t={t_scalar} | decoded | one-step recon — {reports[0]}"))

                del z, reports, noise, t_infer, z_noisy, tokens, ctx, pred, loss
                torch.cuda.empty_cache()

        avg_val_loss   = val_loss / max(1, val_batches)
        avg_val_l1     = val_l1_total / max(1, val_batches)
        avg_val_ssim   = val_ssim_total / max(1, val_batches)
        avg_val_pred_std = val_pred_std_sum / max(1, val_batches)
        avg_val_cos    = val_cos_sum / max(1, val_batches)

        # token length stats
        def _mean_p95(xs):
            if not xs: return 0.0, 0.0
            xs_sorted = sorted(xs)
            p95_idx = int(0.95 * (len(xs_sorted)-1))
            return float(sum(xs_sorted)/len(xs_sorted)), float(xs_sorted[p95_idx])

        train_len_mean, train_len_p95 = _mean_p95(train_tok_lens)
        val_len_mean,   val_len_p95   = _mean_p95(val_tok_lens)

        # t histogram (validation)
        t_hist = wandb.Histogram(np_histogram=torch.tensor(t_vals, dtype=torch.int32).cpu().numpy())

        # latent stats
        lat_mean = sum(val_latent_means)/max(1, len(val_latent_means))
        lat_std  = sum(val_latent_stds)/max(1, len(val_latent_stds))

        # ---- log ----
        log_data = {
            "model/epoch": epoch + 1,
            "model/lr": optimizer.param_groups[0]["lr"],
            "train/loss": avg_train_loss,
            "val/loss": avg_val_loss,
            "val/smooth_l1": avg_val_l1,
            "val/ssim_loss": avg_val_ssim,
            "train/pred_std": avg_train_pred_std,
            "val/pred_std": avg_val_pred_std,
            "train/cos_pred_noise": avg_train_cos,
            "val/cos_pred_noise": avg_val_cos,
            "train/cfg_dropout_observed": avg_drop_ratio,
            "train/cfg_dropout_target": cfg_dropout,
            "val/t_mean": (sum(t_vals)/max(1, len(t_vals))) if t_vals else 0.0,
            "val/t_hist": t_hist,
            "text/train_len_mean": train_len_mean,
            "text/train_len_p95":  train_len_p95,
            "text/val_len_mean":   val_len_mean,
            "text/val_len_p95":    val_len_p95,
            "grad/lora_global_norm": avg_lora_gn,
            "latents/val_mean": lat_mean,
            "latents/val_std":  lat_std,
            "images/val_samples": val_images,
            "inference/guidance_scale": guidance_scale,
        }

        wandb.log(log_data)
        lr_scheduler.step(avg_val_loss)

        if epoch == 0 or avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            save_path = os.path.join(config["MODEL"]["LORA_CHECKPOINT"], "best")
            os.makedirs(save_path, exist_ok=True)
            unet.unet.save_pretrained(save_path)
        else:
            patience_counter += 1
            if patience_counter >= config["TRAINING"]["EARLY_STOP_PATIENCE"]:
                print("[info] Early stopping triggered.")
                break

if __name__ == "__main__":
    # np_histogram helper for wandb.Histogram without importing numpy at top
    import numpy as _np
    def np_histogram(tensor_numpy):
        # tensor_numpy is a numpy array
        hist, bin_edges = _np.histogram(tensor_numpy, bins=10)
        return (hist, bin_edges)
    main()
















# import os
# import math
# import torch
# import yaml
# import wandb
# from tqdm import tqdm
# from datasets import load_from_disk
# from torch.utils.data import DataLoader
# from transformers import AutoTokenizer, AutoModel
# from models.lora_unet_wrapper import LoRAUNetWrapper
# from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL
# from peft import get_peft_model, LoraConfig
# from accelerate import Accelerator
# from pytorch_msssim import ssim



# def composite_loss(pred, target, ssim_weight=0.0, beta=1.0, return_per_sample=False):
#     """
#     If return_per_sample=True, returns (B,) tensor of per-sample losses,
#     else returns scalar + individual components for logging.
#     """
#     # Smooth L1 per-pixel -> per-sample mean
#     sl1_map = torch.nn.functional.smooth_l1_loss(pred, target, beta=beta, reduction="none")
#     sl1_per = sl1_map.view(sl1_map.size(0), -1).mean(dim=1)  # [B]

#     # Normalize to [0,1] per-sample for SSIM stability
#     def _minmax(x):
#         x_min = x.amin(dim=[1, 2, 3], keepdim=True)
#         x_max = x.amax(dim=[1, 2, 3], keepdim=True)
#         return (x - x_min) / (x_max - x_min + 1e-5)

#     pred_norm = _minmax(pred)
#     target_norm = _minmax(target)

#     # SSIM per-sample
#     ssim_per = 1.0 - ssim(pred_norm, target_norm, data_range=1.0, size_average=False)  # [B]

#     loss_per = sl1_per + ssim_weight * ssim_per  # [B]

#     if return_per_sample:
#         return loss_per, sl1_per, ssim_per  # all [B]

#     # scalar fallback (old behavior)
#     loss = loss_per.mean()
#     return loss, sl1_per.mean().item(), ssim_per.mean().item()


# def decode_latents(latents, vae, latents_are_scaled=True):
#     with torch.no_grad():
#         latents = latents.to(next(vae.parameters()).device, dtype=torch.float32)
#         scale = float(getattr(vae.config, "scaling_factor", 0.18215))
#         if latents_are_scaled:
#             latents = latents / scale
#         images = vae.decode(latents).sample.clamp(0, 1)
#     return images


# def to_grayscale(img_tensor):
#     img = img_tensor.mean(dim=0, keepdim=True).repeat(3, 1, 1)  # [3, H, W]
#     img = (img * 255).clamp(0, 255).byte()
#     return img


# def sample_timesteps(batch_size, num_steps, device, bias_cfg, epoch):
#     """
#     bias_cfg: dict with keys
#       ENABLED (bool), START_EPOCH (int), END_EPOCH (int), HIGH_T_MIN_FRAC (float)
#     If enabled and epoch in window, draws t from [floor(frac*num_steps), num_steps).
#     Otherwise draws uniformly from [0, num_steps).
#     """
#     if bias_cfg.get("ENABLED", False) and bias_cfg["START_EPOCH"] <= epoch <= bias_cfg["END_EPOCH"]:
#         lo = int(bias_cfg.get("HIGH_T_MIN_FRAC", 0.7) * num_steps)
#         lo = max(0, min(lo, num_steps - 1))
#         hi = num_steps
#         t = torch.randint(lo, hi, (batch_size,), device=device)
#     else:
#         t = torch.randint(0, num_steps, (batch_size,), device=device)
#     return t.long()


# def timestep_weights(t, num_steps, method):
#     """
#     Returns weights in [B] to upweight hard (high-noise) timesteps.
#     method in {"none","linear","quadratic","cosine"}
#     """
#     if method == "none":
#         return torch.ones_like(t, dtype=torch.float32)

#     x = t.float() / max(1, (num_steps - 1))  # 0..1
#     if method == "linear":
#         w = x  # more weight as t increases
#     elif method == "quadratic":
#         w = x ** 2
#     elif method == "cosine":
#         # low at t=0, high near t=T
#         w = 1.0 - torch.cos(0.5 * math.pi * x)
#     else:
#         w = torch.ones_like(x)
#     # Avoid zeros
#     return (w + 1e-3).to(torch.float32)


# def ohe_to_prompt(ohe_string: str) -> str:
#     """
#     Convert an OHE-style attribute string into a short, readable prompt.
#     Example input: "cardiomegaly_1,edema_0,view_ap"
#     Example output: "Cardiomegaly: present. Edema: absent. View: AP."
#     """
#     parts = ohe_string.split(",")
#     phrases = []
#     for p in parts:
#         p = p.strip()
#         if "_" not in p:
#             phrases.append(p.capitalize())
#             continue
#         name, val = p.rsplit("_", 1)
#         name = name.replace("-", " ").replace("_", " ").capitalize()
#         if val in {"1", "yes", "present"}:
#             phrases.append(f"{name}: present")
#         elif val in {"0", "no", "absent"}:
#             phrases.append(f"{name}: absent")
#         else:
#             phrases.append(f"{name}: {val.upper()}")
#     return ". ".join(phrases) + "."



# def main():
#     with open("config/config.yaml") as f:
#         config = yaml.safe_load(f)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Mixed precision + (if you later run multi-GPU) accumulation-aware
#     accelerator = Accelerator(mixed_precision="fp16")

#     # --- Config knobs (all optional) ---
#     timestep_weighting = config["TRAINING"].get("TIMESTEP_WEIGHTING", "none")  # "none"|"linear"|"quadratic"|"cosine"
#     hard_t_cfg = config["TRAINING"].get("HARD_T_BIAS", {
#         "ENABLED": False, "START_EPOCH": 0, "END_EPOCH": 0, "HIGH_T_MIN_FRAC": 0.7
#     })

#     tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
#     text_encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)
#     for p in text_encoder.parameters():
#         p.requires_grad = False

#     unet = UNet2DConditionModel.from_pretrained(
#         "CompVis/stable-diffusion-v1-4", subfolder="unet", cross_attention_dim=768
#     )
#     # Keep memory in check
#     if hasattr(unet, "enable_gradient_checkpointing"):
#         unet.enable_gradient_checkpointing()

#     lora_config = LoraConfig(
#         r=config["TRAINING"]["LORA_R"],
#         lora_alpha=config["TRAINING"]["LORA_ALPHA"],
#         target_modules=["to_q", "to_k", "to_v"],
#         lora_dropout=config["TRAINING"]["LORA_DROPOUT"],
#         bias="none",
#     )
#     unet = get_peft_model(unet, lora_config)
#     unet = LoRAUNetWrapper(unet, context_dim_in=768, context_dim_out=768).to(device)
#     # If wrapper exposes inner UNet (it does), checkpoint there too
#     if hasattr(unet, "unet") and hasattr(unet.unet, "enable_gradient_checkpointing"):
#         unet.unet.enable_gradient_checkpointing()

#     # VAE on CPU during training; move to GPU for val decode only
#     vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
#     # vae.load_state_dict(torch.load(config["MODEL"]["VAE_CHECKPOINT"], map_location="cpu"))
#     # vae = vae.eval()

#     scheduler = DDPMScheduler.from_pretrained(
#         "CompVis/stable-diffusion-v1-4", subfolder="scheduler"
#     )
#     # Keep SD's training horizon intact (1000). Use 100-step inference preview.
#     scheduler.config.num_train_timesteps = 1000
#     scheduler.set_timesteps(100, device=device)  # preview schedule

#     optimizer = torch.optim.AdamW(unet.parameters(), lr=float(config["TRAINING"]["LEARNING_RATE"]))
#     lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer,
#         mode="min",
#         factor=float(config["TRAINING"]["LR_SCHEDULER"]["FACTOR"]),
#         patience=int(config["TRAINING"]["LR_SCHEDULER"]["PATIENCE"]),
#         min_lr=float(config["TRAINING"]["LR_SCHEDULER"]["MIN_LR"]),
#     )

#     # Prepare with Accelerate
#     unet, optimizer = accelerator.prepare(unet, optimizer)

#     wandb.init(
#         project=config["WANDB"]["PROJECT"],
#         name=config["WANDB"]["RUN_NAME_LORA_UNET"],
#         config=config,
#     )

#     latent_path = config["DATASET"]["LATENT_OUTPUT_PATH"]
#     batch_size = int(config["TRAINING"]["BATCH_SIZE"])
#     grad_accum_steps = int(config["TRAINING"].get("GRAD_ACCUM_STEPS", 1))
#     max_grad_norm = float(config["TRAINING"].get("MAX_GRAD_NORM", 1.0))
#     train_on_three_batches = config["TRAINING"].get("TRAIN_ON_THREE_BATCHES", False)
#     chunk_limit = int(config["TRAINING"].get("CHUNK_LIMIT", 10))
#     ssim_weight = float(config["TRAINING"].get("SSIM_WEIGHT", 0.0))
#     beta = float(config["TRAINING"].get("BETA", 1.0))
#     epochs = int(config["TRAINING"]["EPOCHS"])
#     guidance_scale = float(config["TRAINING"].get("GUIDANCE_SCALE", 7.5))

#     # ---- Optional chunk controls (default noop) ----
#     chunk_start = int(config["TRAINING"].get("CHUNK_START", 0))           # start index
#     shuffle_chunks = bool(config["TRAINING"].get("SHUFFLE_CHUNKS", False))
#     explicit_chunk_ids = config["TRAINING"].get("CHUNK_IDS", None)        # e.g., [0,2,5]
#     debug_per_chunk = int(config["TRAINING"].get("DEBUG_PER_CHUNK_SAMPLES", 0))    # cap per-chunk rows
#     debug_max_global_steps = int(config["TRAINING"].get("DEBUG_MAX_GLOBAL_STEPS", 0))

#     def chunk_ids_for_epoch(epoch: int):
#         # Build the available chunk list starting at chunk_start, respecting CHUNK_LIMIT
#         if isinstance(explicit_chunk_ids, list) and len(explicit_chunk_ids) > 0:
#             ids = [int(x) for x in explicit_chunk_ids][:chunk_limit]
#         else:
#             ids = []
#             cid = chunk_start
#             while len(ids) < chunk_limit:
#                 p = os.path.join(latent_path, f"latent_train_chunk_{cid}")
#                 if not os.path.exists(p):
#                     break
#                 ids.append(cid)
#                 cid += 1
#         if shuffle_chunks and len(ids) > 1:
#             g = torch.Generator()
#             g.manual_seed(epoch)  # epoch‑deterministic shuffle
#             perm = torch.randperm(len(ids), generator=g)
#             ids = [ids[i] for i in perm.tolist()]
#         return ids

#     best_val_loss = float('inf')
#     patience_counter = 0
#     global_steps = 0

#     def log_ctx_stats(ctx, tag):
#         # ctx: [B, L, 768]
#         with torch.no_grad():
#             cm = ctx.mean().detach().float().item()
#             cs = ctx.std().detach().float().item()
#         return {
#             f"context/{tag}_mean": cm,
#             f"context/{tag}_std": cs
#         }

#     for epoch in range(epochs):
#         unet.train()
#         epoch_loss = 0.0
#         num_batches = 0
#         train_l1_sum = 0.0
#         train_ssim_sum = 0.0

#         # ---- iterate selected chunks, preserving CHUNK_LIMIT behavior ----
#         for chunk_id in chunk_ids_for_epoch(epoch):
#             chunk_path = os.path.join(latent_path, f"latent_train_chunk_{chunk_id}")
#             dataset = load_from_disk(chunk_path).with_format("torch", columns=["z_target", "report"])

#             # Optional tiny overfit/debug cap that won’t break normal runs
#             if debug_per_chunk > 0 and hasattr(dataset, "select"):
#                 n = min(debug_per_chunk, len(dataset))
#                 dataset = dataset.select(range(n))

#             loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#             loader = accelerator.prepare(loader)

#             for i, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1} Chunk {chunk_id}")):
#                 if train_on_three_batches and i == 3:
#                     break

#                 z = batch["z_target"].to(device, non_blocking=True)
#                 reports = batch["report"]

#                 # Text encoding: no-grad + autocast to cut VRAM
#                 with torch.no_grad(), accelerator.autocast():
#                     tokens = tokenizer(
#                         [ohe_to_prompt(r) for r in reports],
#                         padding="max_length",
#                         truncation=True,
#                         max_length=128,
#                         return_tensors="pt",
#                     ).to(device, non_blocking=True)
#                     ctx = text_encoder(**tokens).last_hidden_state  # [B, L, 768]
#                     assert ctx.size(-1) == 768, f"Unexpected context dim: {ctx.size(-1)}"

#                 with accelerator.autocast():
#                     # Draw noise once and keep it
#                     noise = torch.randn_like(z)
#                     # Biased/high‑t sampling (curriculum) + add noise
#                     t = sample_timesteps(z.size(0), scheduler.config.num_train_timesteps, device, hard_t_cfg, epoch)
#                     z_noisy = scheduler.add_noise(z, noise, t)

#                     # Forward
#                     pred = unet(z_noisy, t, ctx).sample  # predicted noise

#                     # Per-sample loss + weighting (against the exact noise we added)
#                     loss_per, sl1_per, ssim_per = composite_loss(
#                         pred, noise, ssim_weight=ssim_weight, beta=beta, return_per_sample=True
#                     )
#                     w = timestep_weights(t, scheduler.config.num_train_timesteps, timestep_weighting)  # [B]
#                     # Normalize weights (optional): keep average ~1
#                     w = w * (w.numel() / (w.sum() + 1e-8))
#                     loss = (loss_per * w).mean() / grad_accum_steps

#                 accelerator.backward(loss)

#                 if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(loader):
#                     accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
#                     optimizer.step()
#                     optimizer.zero_grad(set_to_none=True)

#                 epoch_loss += loss.item() * grad_accum_steps
#                 num_batches += 1
#                 train_l1_sum += sl1_per.mean().item()
#                 train_ssim_sum += ssim_per.mean().item()
#                 global_steps += 1

#                 # (optional) tiny, cheap context stats once in a while
#                 if (i == 0) and (chunk_id == 0):
#                     ctx_stats_train = log_ctx_stats(ctx, "train")

#                 # Debug global cap across chunks/epochs
#                 if debug_max_global_steps > 0 and global_steps >= debug_max_global_steps:
#                     break

#                 del z, reports, tokens, ctx, t, z_noisy, pred, loss, loss_per, sl1_per, ssim_per, w, noise

#             if debug_max_global_steps > 0 and global_steps >= debug_max_global_steps:
#                 break

#         avg_train_loss = epoch_loss / max(1, num_batches)
#         avg_train_l1 = train_l1_sum / max(1, num_batches)
#         avg_train_ssim = train_ssim_sum / max(1, num_batches)

#         # ----- Validation -----
#         val_path = os.path.join(latent_path, "latent_val")
#         val_dataset = load_from_disk(val_path).with_format("torch", columns=["z_target", "report"])
#         val_loader = DataLoader(
#             val_dataset,
#             batch_size=batch_size,
#             shuffle=False,
#         )
#         val_loader = accelerator.prepare(val_loader)

#         unet.eval()
#         val_loss = 0.0
#         val_batches = 0
#         val_l1_total = 0.0
#         val_ssim_total = 0.0
#         val_images = []
#         cfg_images = []

#         # Move VAE to GPU once for decode, then park back on CPU
#         vae = vae.to(device)

#         with torch.no_grad():
#             for i, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")):
#                 if train_on_three_batches and i == 3:
#                     break

#                 z = batch["z_target"].to(device, non_blocking=True)
#                 reports = batch["report"]

#                 with accelerator.autocast():
#                     # Random step from the inference schedule for preview
#                     ti = torch.randint(0, len(scheduler.timesteps), (z.size(0),), device=device).long()
#                     t_infer = scheduler.timesteps[ti]
#                     noise = torch.randn_like(z)
#                     z_noisy = scheduler.add_noise(z, noise, t_infer)

#                     tokens = tokenizer(
#                         [ohe_to_prompt(r) for r in reports],
#                         padding="max_length",
#                         truncation=True,
#                         max_length=128,
#                         return_tensors="pt",
#                     ).to(device, non_blocking=True)
#                     ctx = text_encoder(**tokens).last_hidden_state
#                     assert ctx.size(-1) == 768

#                     pred = unet(z_noisy, t_infer, ctx).sample
#                     loss, sl1_val, ssim_val = composite_loss(
#                         pred, noise, ssim_weight=ssim_weight, beta=beta, return_per_sample=False
#                     )
#                     val_loss += loss.item()
#                     val_l1_total += sl1_val
#                     val_ssim_total += ssim_val
#                     val_batches += 1

#                     # One reverse step panel
#                     recon_latent = scheduler.step(
#                         model_output=pred[:1],
#                         timestep=t_infer[:1],
#                         sample=z_noisy[:1],
#                     ).prev_sample

#                 # Decode panels in fp32
#                 orig_img = to_grayscale(decode_latents(z[:1], vae)[0].cpu())
#                 recon_img = to_grayscale(decode_latents(recon_latent, vae)[0].cpu())
#                 sep = 255 * torch.ones_like(orig_img[:, :, :1])
#                 panel = torch.cat([orig_img, sep, recon_img], dim=2)
#                 if len(val_images) < 3:
#                     val_images.append(
#                         wandb.Image(panel, caption=f"decoded | recon\n{reports[0]}\nt={t_infer[:1].item()}")
#                     )

#                 # CFG sample (single reverse step preview)
#                 if len(cfg_images) < 3:
#                     with accelerator.autocast():
#                         uncond_ctx = torch.zeros_like(ctx)
#                         pred_uncond = unet(z_noisy[:1], t_infer[:1], uncond_ctx[:1]).sample
#                         pred_text = pred[:1]
#                         pred_cfg = pred_uncond + guidance_scale * (pred_text - pred_uncond)
#                         recon_cfg_latent = scheduler.step(
#                             model_output=pred_cfg,
#                             timestep=t_infer[:1],
#                             sample=z_noisy[:1],
#                         ).prev_sample
#                     cfg_img = to_grayscale(decode_latents(recon_cfg_latent, vae)[0].cpu())
#                     cfg_panel = torch.cat([orig_img, sep, cfg_img], dim=2)
#                     cfg_images.append(
#                         wandb.Image(cfg_panel, caption=f"decoded | cfg\n{reports[0]}\nt={t_infer[:1].item()}")
#                     )

#                 if (i == 0):
#                     ctx_stats_val = log_ctx_stats(ctx, "val")

#                 del z, reports, ti, t_infer, noise, z_noisy, tokens, ctx, pred, recon_latent

#         # free VRAM for next epoch
#         vae = vae.cpu()

#         avg_val_loss = val_loss / max(1, val_batches)
#         avg_val_l1 = val_l1_total / max(1, val_batches)
#         avg_val_ssim = val_ssim_total / max(1, val_batches)

#         # --- Conditioning ablation (quick, one batch) ---
#         with torch.no_grad(), accelerator.autocast():
#             val_iter = iter(val_loader)
#             batch = next(val_iter)
#             z = batch["z_target"].to(device)
#             reports = batch["report"]
#             ti = torch.randint(0, len(scheduler.timesteps), (z.size(0),), device=device).long()
#             t_infer = scheduler.timesteps[ti]
#             noise = torch.randn_like(z)
#             z_noisy = scheduler.add_noise(z, noise, t_infer)

#             tokens = tokenizer(list(reports), padding="max_length", truncation=True, max_length=128, return_tensors="pt").to(device)
#             ctx = text_encoder(**tokens).last_hidden_state

#             pred_real = unet(z_noisy, t_infer, ctx).sample
#             loss_real, _, _ = composite_loss(pred_real, noise, ssim_weight=ssim_weight, beta=beta)

#             perm = torch.randperm(ctx.size(0), device=ctx.device)
#             pred_shuf = unet(z_noisy, t_infer, ctx[perm]).sample
#             loss_shuf, _, _ = composite_loss(pred_shuf, noise, ssim_weight=ssim_weight, beta=beta)

#             pred_zero = unet(z_noisy, t_infer, torch.zeros_like(ctx)).sample
#             loss_zero, _, _ = composite_loss(pred_zero, noise, ssim_weight=ssim_weight, beta=beta)

#         ablation = {
#             "ablation/loss_real_ctx": loss_real.item(),
#             "ablation/loss_shuf_ctx": loss_shuf.item(),
#             "ablation/loss_zero_ctx": loss_zero.item(),
#             "ablation/real_vs_shuf_rel_improve_%": 100.0 * (loss_shuf.item() - loss_real.item()) / (abs(loss_shuf.item()) + 1e-8)
#         }

#         # ---- Logging ----
#         log_data = {
#             "model/epoch": epoch + 1,
#             "train/loss": avg_train_loss,
#             "train/smooth_l1": avg_train_l1,
#             "train/ssim_loss": avg_train_ssim,
#             "val/loss": avg_val_loss,
#             "val/smooth_l1": avg_val_l1,
#             "val/ssim_loss": avg_val_ssim,
#             "model/lr": optimizer.param_groups[0]["lr"],
#             "images/val_samples": val_images,
#             "images/cfg_samples": cfg_images,
#             "timestep/weighting": {"none": 0, "linear": 1, "quadratic": 2, "cosine": 3}.get(timestep_weighting, 0),
#             "timestep/bias_enabled": int(bool(hard_t_cfg.get("ENABLED", False))),
#         }
#         log_data.update(ablation)

#         # Merge context stats if collected
#         if 'ctx_stats_train' in locals():
#             log_data.update(ctx_stats_train)
#         if 'ctx_stats_val' in locals():
#             log_data.update(ctx_stats_val)

#         wandb.log(log_data)

#         lr_scheduler.step(avg_val_loss)

#         # Save best
#         if epoch == 0 or avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             patience_counter = 0
#             save_path = os.path.join(config["MODEL"]["LORA_CHECKPOINT"], "best")
#             os.makedirs(save_path, exist_ok=True)
#             unet.unet.save_pretrained(save_path)
#         else:
#             patience_counter += 1
#             if patience_counter >= int(config["TRAINING"]["EARLY_STOP_PATIENCE"]):
#                 print("[info] Early stopping triggered.")
#                 break


# if __name__ == "__main__":
#     main()