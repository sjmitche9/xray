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


def read_scale_from_meta(latent_dir):
	try:
		meta_path = os.path.join(latent_dir, "meta.yaml")
		if not os.path.exists(meta_path):
			return None
		import yaml as _yaml
		with open(meta_path, "r") as f:
			meta = _yaml.safe_load(f)
		store_scaled = bool(meta.get("store_scaled", False))
		scaling = float(meta.get("scaling_factor", 0.18215))
		return scaling if store_scaled else 1.0
	except Exception:
		return None


def psnr01(a, b, eps=1e-8):
    # a,b in [0,1], same shape
    mse = torch.mean((a - b) ** 2)
    return 10.0 * torch.log10(1.0 / (mse + eps))


def decode_latents(latents, vae, scale=0.18215):
	with torch.no_grad():
		latents = latents.to(next(vae.parameters()).device)
		img = vae.decode(latents / scale).sample          # in [-1, 1]
		img = (img + 1) / 2                               # -> [0, 1]
		img = img.clamp(0, 1)
	return img


def to_grayscale(img_tensor, auto_contrast=True, eps=1e-6):
	# img_tensor: [3,H,W] in [0,1]
	img = img_tensor.mean(dim=0, keepdim=True)  # [1,H,W]
	if auto_contrast:
		mn = img.min()
		mx = img.max()
		img = (img - mn) / (mx - mn + eps)
	img = img.repeat(3, 1, 1)
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
	# NOTE: CPU text-encode + filtered inputs, then move only context to GPU
	with torch.no_grad():
		tokens = tokenizer(list(reports), padding="longest", truncation=True, max_length=64, return_tensors="pt")
		bert_inputs = {k: tokens[k] for k in ("input_ids", "attention_mask", "token_type_ids") if k in tokens}
		outputs = text_encoder(**bert_inputs)  # CPU
		ctx_cond = outputs.last_hidden_state.to(device=device, dtype=next(unet.parameters()).dtype, non_blocking=True)
		ctx_uncond = torch.zeros_like(ctx_cond)

		B = len(reports)
		z = torch.randn((B, 4, 32, 32), device=device, dtype=next(unet.parameters()).dtype)
		timesteps = scheduler.timesteps.to(device)

		for t in timesteps:
			tt = torch.full((B,), t, device=device, dtype=torch.long)
			eps_c = unet(z, tt, ctx_cond).sample
			eps_u = unet(z, tt, ctx_uncond).sample
			eps = eps_u + guidance_scale * (eps_c - eps_u)
			z = scheduler.step(model_output=eps, timestep=t, sample=z).prev_sample
			del tt, eps_c, eps_u, eps
		decoded = vae.decode(z / scale).sample.clamp(0, 1)
		del z, tokens, outputs, ctx_cond, ctx_uncond
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

	# ---- text encoder (CPU, no grad) ----
	tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
	text_encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to("cpu").eval()
	for p in text_encoder.parameters(): 
		p.requires_grad = False

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

				# ---- CPU tokenize + encode, filter inputs ----
				tokens = tokenizer(
					list(reports), padding="longest", truncation=True, max_length=64,
					return_tensors="pt", return_length=True
				)
				bert_inputs = {k: tokens[k] for k in ("input_ids", "attention_mask", "token_type_ids") if k in tokens}
				with torch.no_grad():
					outputs = text_encoder(**bert_inputs)  # CPU
				ctx = outputs.last_hidden_state.to(device=device, dtype=next(unet.parameters()).dtype, non_blocking=True)

				ctx, drop_ratio = apply_cfg_dropout(ctx, cfg_dropout)
				observed_drop.append(drop_ratio)

				# token length stats (true lengths from attention mask or 'length' key)
				if "length" in tokens:
					train_tok_lens += [int(x) for x in tokens["length"]]
				elif "attention_mask" in tokens:
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

				del z, reports, noise, t, z_noisy, tokens, outputs, ctx, pred, loss
				torch.cuda.empty_cache()

			chunk_id += 1

		avg_train_loss = epoch_loss / max(1, num_batches)
		avg_drop_ratio = float(sum(observed_drop) / max(1, len(observed_drop)))
		avg_train_pred_std = train_pred_std_sum / max(1, num_batches)
		avg_train_cos = train_cos_sum / max(1, num_batches)
		avg_lora_gn = (lora_grad_norm_sum / max(1, lora_grad_norm_count))

		# -------- validation --------
		val_path = os.path.join(latent_path, "latent_val")
		preview_scale = read_scale_from_meta(val_path) or SCALE
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
		pair_psnrs, pair_ssims = [], []


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

				# ---- CPU tokenize + encode, filter inputs ----
				tokens = tokenizer(
					list(reports), padding="longest", truncation=True, max_length=64,
					return_tensors="pt", return_length=True
				)
				bert_inputs = {k: tokens[k] for k in ("input_ids", "attention_mask", "token_type_ids") if k in tokens}
				outputs = text_encoder(**bert_inputs)  # CPU
				ctx = outputs.last_hidden_state.to(device=device, dtype=next(unet.parameters()).dtype, non_blocking=True)

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
				elif "attention_mask" in tokens:
					val_tok_lens += [int(m.sum().item()) for m in tokens["attention_mask"]]

				# === PREVIEW: 3 samples × [decoded(z) | full diffusion recon] ===
				PREVIEW_SAMPLES = 3  # hard-coded
				if len(val_images) < PREVIEW_SAMPLES:
					with torch.no_grad():
						B = z.shape[0]
						take = min(B, PREVIEW_SAMPLES)
						ts = scheduler.timesteps  # descending tensor (e.g., len=100)

						for k in range(take):
							# ----- select kth sample from current val batch -----
							z_k      = z[k:k+1]            # [1,4,H/8,W/8] (clean latent)
							zt_k     = z_noisy[k:k+1]      # [1,4,H/8,W/8] (noisy at t_k)
							ctx_k    = ctx[k:k+1]          # [1,L,768]
							rep_k    = reports[k]
							t_k_val  = t_infer[k]          # scalar tensor with actual timestep value
							t_k      = int(t_k_val.item())

							# ---------- decoded(z): true VAE reconstruction (reference) ----------
							dec_gpu = decode_latents(z_k, vae, scale=SCALE)   # [1,3,H,W] in [0,1]
							dec = dec_gpu[0].detach().cpu()                   # [3,H,W] CPU
							del dec_gpu

							# ---------- full diffusion recon from z_t down to t=0 ----------
							# find index of t_k in scheduler.timesteps
							idxs = (ts == t_k_val).nonzero(as_tuple=True)[0]
							start_idx = int(idxs[0].item()) if idxs.numel() > 0 else 0

							z_hat = zt_k.clone()
							for t_step in ts[start_idx:]:
								t_b = t_step[None].to(device)
								eps = unet(z_hat, t_b, ctx_k).sample
								z_hat = scheduler.step(model_output=eps, timestep=t_step, sample=z_hat).prev_sample
								del t_b, eps

							full_gpu = decode_latents(z_hat, vae, scale=SCALE)  # [1,3,H,W] in [0,1]
							full = full_gpu[0].detach().cpu()
							del z_hat, full_gpu

							# ---------- metrics (PSNR/SSIM) on grayscale [0,1] ----------
							dec_gf  = dec.mean(dim=0, keepdim=True).unsqueeze(0)    # [1,1,H,W]
							full_gf = full.mean(dim=0, keepdim=True).unsqueeze(0)   # [1,1,H,W]
							pair_psnrs.append(float(psnr01(dec_gf, full_gf).item()))
							pair_ssims.append(float(ssim(dec_gf, full_gf, data_range=1.0, size_average=True).item()))

							# ---------- build concatenated panel: decoded | full diffusion ----------
							H = min(dec.shape[1], full.shape[1]); W = min(dec.shape[2], full.shape[2])
							dec_u8  = to_grayscale(dec[:, :H, :W])     # [3,H,W] uint8
							full_u8 = to_grayscale(full[:, :H, :W])    # [3,H,W] uint8
							sep = 255 * torch.ones_like(dec_u8[:, :, :1])
							panel = torch.cat([dec_u8, sep, full_u8], dim=2)

							val_images.append(
								wandb.Image(panel, caption=f"[{k+1}/3] decoded(z) | full diffusion (from t={t_k}) — {rep_k}")
							)

							# tidy CPU tensors (panel kept inside W&B object)
							del dec, full, dec_gf, full_gf, dec_u8, full_u8, sep, panel

							# stop if we reached 3 panels
							if len(val_images) >= PREVIEW_SAMPLES:
								break

					# cap exactly at 3 panels (1 per sample)
					if len(val_images) > PREVIEW_SAMPLES:
						val_images[:] = val_images[:PREVIEW_SAMPLES]

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
			"text/train_len_mean": train_len_mean,
			"text/train_len_p95":  train_len_p95,
			"text/val_len_mean":   val_len_mean,
			"text/val_len_p95":    val_len_p95,
			"grad/lora_global_norm": avg_lora_gn,
			"latents/val_mean": lat_mean,
			"latents/val_std":  lat_std,
			"images/val_samples": val_images,
			"val/recon_psnr_mean": (sum(pair_psnrs)/len(pair_psnrs)) if pair_psnrs else 0.0,
			"val/recon_ssim_mean": (sum(pair_ssims)/len(pair_ssims)) if pair_ssims else 0.0,

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
	main()