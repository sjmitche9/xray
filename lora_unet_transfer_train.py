# lora_unet_transfer_train.py
import os
import glob
import math
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
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.environ.get("XRAY_PROJECT_ROOT", os.path.dirname(__file__)))


def composite_loss_tune(pred, target, ssim_weight=0.0, beta=1.0):
    sl1 = F.smooth_l1_loss(pred, target, beta=beta)

    if ssim_weight <= 0.0:
        # skip all SSIM math entirely
        return sl1, float(sl1.item()), 0.0

    # Normalize before SSIM to keep values in [0,1]
    pred_norm = (pred - pred.amin()) / (pred.amax() - pred.amin() + 1e-5)
    target_norm = (target - target.amin()) / (target.amax() - target.amin() + 1e-5)

    ssim_loss = 1.0 - ssim(pred_norm, target_norm, data_range=1.0, size_average=True)
    total = sl1 + ssim_weight * ssim_loss
    return total, float(sl1.item()), float(ssim_loss.item())


def _tune_report(
	*,
	val_score: float,
	recon_ssim_mean: float,
	recon_psnr_mean: float,
	val_loss: float,
	epoch: int,
):
	try:
		from ray import tune
		tune.report(
			{
				"val_score": float(val_score),
				"val_recon_ssim_mean": float(recon_ssim_mean),
				"val_recon_psnr_mean": float(recon_psnr_mean),
				"val_loss": float(val_loss),
				"epoch": int(epoch),
			}
		)
	except Exception:
		pass


# ----------------- helpers -----------------
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
	mse = torch.mean((a - b) ** 2)
	return 10.0 * torch.log10(1.0 / (mse + eps))


def compute_val_score(
	*,
	avg_val_loss: float,
	avg_val_cos: float,
	avg_val_pred_std: float,
	cfg_delta_norm: float,
	target_pred_std: float = 1.0,
) -> float:
	"""
	Composite objective for hyperparam tuning (higher is better).

	Components:
	  - loss_term: encourages denoiser learning (lower loss => higher score)
	  - cos_term: discourages degenerate "wrong direction" predictions
	  - std_term: discourages collapse/explosion of eps magnitude
	  - cfg_term: encourages conditioning to matter (delta_norm > 0)

	Notes:
	  * Keep weights modest to avoid overfitting the proxy.
	  * target_pred_std ~ 1.0 is reasonable for epsilon prediction with unit Gaussian noise.
	"""
	loss_term = -float(avg_val_loss)
	cos_term = float(avg_val_cos)

	# pred_std stabilization: penalize log-ratio distance from target
	eps = 1e-8
	ratio = (float(avg_val_pred_std) + eps) / (float(target_pred_std) + eps)
	std_term = -abs(math.log(ratio))

	# cfg delta norm: saturating reward in [0, 1]
	cap = 2.0
	cfg_term = max(0.0, min(float(cfg_delta_norm) / cap, 1.0))

	# weights (kept conservative)
	score = loss_term + 0.20 * cos_term + 0.10 * std_term + 0.20 * cfg_term
	return float(score)


def decode_latents(latents, vae, scale=0.18215):
	with torch.no_grad():
		latents = latents.to(next(vae.parameters()).device)
		img = vae.decode(latents / scale).sample
		img = (img + 1) / 2
		img = img.clamp(0, 1)
	return img


def to_grayscale(img_tensor, auto_contrast=True, eps=1e-6):
	img = img_tensor.mean(dim=0, keepdim=True)
	if auto_contrast:
		mn = img.min()
		mx = img.max()
		img = (img - mn) / (mx - mn + eps)
	img = img.repeat(3, 1, 1)
	return (img * 255).clamp(0, 255).byte()


def iter_lora_params(model):
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


def make_deterministic_noise(shape, device, dtype, seed=None):
	"""Create deterministic Gaussian noise that works across torch versions/platforms."""
	if seed is not None:
		g = torch.Generator(device="cpu").manual_seed(int(seed))
		n = torch.randn(shape, generator=g, dtype=torch.float32, device="cpu")
		return n.to(device=device, dtype=dtype, non_blocking=True)
	return torch.randn(shape, device=device, dtype=dtype)


# ---- token reducer (evenly spaced k tokens per sample) ----
def select_k_tokens(ctx_full, attn_mask=None, k=4):
	"""
	ctx_full: [B, L, D]; attn_mask: [B, L] optional. Returns [B, k, D].
	"""
	B, L, D = ctx_full.shape
	device = ctx_full.device
	out = torch.empty((B, k, D), device=device, dtype=ctx_full.dtype)
	if attn_mask is None:
		idx = torch.linspace(0, L - 1, steps=k, device=device).round().long().clamp(0, L - 1)
		out[:] = ctx_full[:, idx, :]
		return out
	for b in range(B):
		valid = attn_mask[b].bool()
		Lb = int(valid.sum().item())
		if Lb <= 0:
			pooled = ctx_full[b].mean(dim=0, keepdim=True)
			out[b] = pooled.expand(k, D)
		else:
			idx = torch.linspace(0, Lb - 1, steps=k, device=device).round().long()
			valid_idx = torch.nonzero(valid, as_tuple=True)[0]
			chosen = valid_idx[idx.clamp_max(valid_idx.numel() - 1)]
			out[b] = ctx_full[b, chosen, :]
	return out


def sample_from_model(
	unet,
	vae,
	tokenizer,
	text_encoder,
	scheduler,
	device,
	reports,
	guidance_scale=7.5,
	scale=0.18215,
	KTOK=16,
	batched_cfg=False,
):
	"""
	Stable CFG sampling:
	  • latents & scheduler math in float32
	  • UNet forward in model dtype (fp16/bf16/fp32)
	  • proper [-1,1] -> [0,1] decode
	"""
	with torch.no_grad():
		# ----- text encode on CPU -----
		tokens = tokenizer(list(reports), padding="longest", truncation=True, max_length=64, return_tensors="pt")
		bert_inputs = {k: tokens[k] for k in ("input_ids", "attention_mask", "token_type_ids") if k in tokens}
		outputs = text_encoder(**bert_inputs)
		ctx_full = outputs.last_hidden_state
		attn = bert_inputs.get("attention_mask", None)

		model_dtype = next(unet.parameters()).dtype
		ctx_cond = select_k_tokens(ctx_full, attn, k=KTOK).to(device=device, dtype=model_dtype, non_blocking=True)

		null_tokens = tokenizer([""], padding="longest", truncation=True, max_length=64, return_tensors="pt")
		null_inputs = {k: null_tokens[k] for k in ("input_ids", "attention_mask", "token_type_ids") if k in null_tokens}
		null_ctx = text_encoder(**null_inputs).last_hidden_state
		base_null = null_ctx.mean(dim=1, keepdim=True)
		ctx_uncond = base_null.expand(ctx_cond.size(0), KTOK, base_null.size(-1)).to(
			device=device, dtype=model_dtype, non_blocking=True
		)

		# ----- latents/scheduler in float32 -----
		B = len(reports)
		z = torch.randn((B, 4, 32, 32), device=device, dtype=torch.float32) * float(scheduler.init_noise_sigma)

		timesteps = scheduler.timesteps.to(device)
		if timesteps.numel() > 1 and timesteps[-1] > timesteps[0]:
			timesteps = timesteps.flip(0)

		if torch.is_tensor(guidance_scale):
			gs = guidance_scale.to(device=device, dtype=torch.float32).view(-1, *([1] * 3))
		else:
			gs = torch.tensor(float(guidance_scale), device=device, dtype=torch.float32)

		for t in timesteps:
			tt = torch.full((B,), t, device=device, dtype=torch.long)
			z_in = z.to(dtype=model_dtype)

			if batched_cfg:
				ctx_cat = torch.cat([ctx_uncond, ctx_cond], dim=0)
				z_cat = torch.cat([z_in, z_in], dim=0)
				tt_cat = torch.cat([tt, tt], dim=0)
				eps_cat = unet(z_cat, tt_cat, ctx_cat).sample.to(torch.float32)
				eps_u, eps_c = eps_cat.chunk(2, dim=0)
				del ctx_cat, z_cat, tt_cat, eps_cat
			else:
				eps_c = unet(z_in, tt, ctx_cond).sample.to(torch.float32)
				eps_u = unet(z_in, tt, ctx_uncond).sample.to(torch.float32)

			delta = eps_c - eps_u
			eps = eps_u + gs * delta
			z = scheduler.step(model_output=eps, timestep=t, sample=z).prev_sample

			del tt, z_in, eps_c, eps_u, delta, eps

		decoded = decode_latents(z, vae, scale=scale)
		del z, tokens, outputs, ctx_cond, ctx_uncond
	return decoded


# ----------------- LoRA-only EMA (robust to wrappers) -----------------
def _lora_keys(sd):
	return [k for k, v in sd.items() if "lora" in k.lower() and torch.is_floating_point(v)]


class LoRAEMA:
	def __init__(self, model: nn.Module, decay: float = 0.999):
		self.decay = decay
		self.keys, self.shadow = self._init_from_model(model)
		self.backup = {}

	def _unwrap(self, model: nn.Module):
		return getattr(model, "module", model)

	def _state_dict(self, model: nn.Module):
		m = self._unwrap(model)
		return m.state_dict()

	def _init_from_model(self, model: nn.Module):
		sd = self._state_dict(model)
		keys = _lora_keys(sd)
		shadow = {k: sd[k].detach().cpu().clone() for k in keys}
		return keys, shadow

	@torch.no_grad()
	def update(self, model: nn.Module):
		sd = self._state_dict(model)
		for k in self.keys:
			self.shadow[k].mul_(self.decay).add_(sd[k].detach().cpu(), alpha=1.0 - self.decay)

	@torch.no_grad()
	def store(self, model: nn.Module):
		sd = self._state_dict(model)
		self.backup = {k: sd[k].detach().clone() for k in self.keys}

	@torch.no_grad()
	def copy_to(self, model: nn.Module):
		m = self._unwrap(model)
		m.load_state_dict(self.shadow, strict=False)

	@torch.no_grad()
	def restore(self, model: nn.Module):
		if self.backup:
			m = self._unwrap(model)
			m.load_state_dict(self.backup, strict=False)
		self.backup = {}


# ----------------- checkpoint utils -----------------
def _preferred_ckpt_dir(root, name="", prefer_best=True):
	best = os.path.join(root, "best")
	if prefer_best and os.path.isdir(best):
		return best
	elif name != "":
		return os.path.join(root, name)
	else:
		cand = sorted(glob.glob(os.path.join(root, "epoch*")))
		return cand[-1] if cand else None


def _optim_hparams(pg):
	return {
		"lr": pg.get("lr", None),
		"betas": tuple(pg.get("betas", (0.9, 0.999))),
		"eps": pg.get("eps", 1e-8),
		"weight_decay": pg.get("weight_decay", 0.0),
		"amsgrad": pg.get("amsgrad", False),
	}


def _save_checkpoint(
	accelerator, unet_wrapped, optimizer, lr_scheduler, ema, ckpt_root, epoch, best_val_loss, best_val_score, patience_counter
):
	accelerator.wait_for_everyone()
	os.makedirs(ckpt_root, exist_ok=True)

	unwrapped = accelerator.unwrap_model(unet_wrapped).unet

	epoch_folder = os.path.join(ckpt_root, f"epoch{epoch+1:04d}")
	os.makedirs(epoch_folder, exist_ok=True)

	# Save the full LoRA adapter (all layers included)
	unwrapped.save_pretrained(epoch_folder)

	# also save the wrapper's context_proj weights
	wrapped = accelerator.unwrap_model(unet_wrapped)
	proj_sd = {k: v.detach().cpu() for k, v in wrapped.state_dict().items() if k.startswith("context_proj.")}

	state = {
		"epoch": epoch,
		"best_val_loss": best_val_loss,
		"best_val_score": float(best_val_score),
		"patience_counter": patience_counter,
		"optimizer": optimizer.state_dict(),
		"lr_scheduler": lr_scheduler.state_dict(),
		"ema_shadow": {k: v.clone() for k, v in ema.shadow.items()},
		"ema_keys": list(ema.keys),
		"opt_hparams": [_optim_hparams(pg) for pg in optimizer.param_groups],
		"context_proj": proj_sd,
	}
	torch.save(state, os.path.join(epoch_folder, "state.pt"))
	accelerator.print(f"[save] wrote {epoch_folder}")


def _save_best_checkpoint(
	accelerator, unet_wrapped, optimizer, lr_scheduler, ema, ckpt_root, epoch, best_val_loss, best_val_score, patience_counter
):
	accelerator.wait_for_everyone()
	best_path = os.path.join(ckpt_root, "best")
	os.makedirs(best_path, exist_ok=True)

	ema.store(unet_wrapped)
	ema.copy_to(unet_wrapped)

	accelerator.unwrap_model(unet_wrapped).unet.save_pretrained(best_path)

	wrapped = accelerator.unwrap_model(unet_wrapped)
	proj_sd = {k: v.detach().cpu() for k, v in wrapped.state_dict().items() if k.startswith("context_proj.")}

	state = {
		"epoch": epoch,
		"best_val_loss": best_val_loss,
		"best_val_score": float(best_val_score),
		"patience_counter": patience_counter,
		"optimizer": optimizer.state_dict(),
		"lr_scheduler": lr_scheduler.state_dict(),
		"ema_shadow": {k: v.clone() for k, v in ema.shadow.items()},
		"ema_keys": list(ema.keys),
		"opt_hparams": [_optim_hparams(pg) for pg in optimizer.param_groups],
		"context_proj": proj_sd,
		"is_best": True,
	}
	torch.save(state, os.path.join(best_path, "state.pt"))
	torch.save(proj_sd, os.path.join(best_path, "context_proj.pt"))

	ema.restore(unet_wrapped)
	accelerator.print(f"[save] updated best (EMA + state.pt) -> {best_path}")


def _identity_init_context_proj(accelerator, unet_wrapped):
	with torch.no_grad():
		wrapped = accelerator.unwrap_model(unet_wrapped)
		if hasattr(wrapped, "context_proj"):
			W = wrapped.context_proj
			nn.init.zeros_(W.weight)
			diag_len = min(W.weight.shape[0], W.weight.shape[1])
			for i in range(diag_len):
				W.weight[i, i] = 1.0
			if W.bias is not None:
				nn.init.zeros_(W.bias)
			accelerator.print("[resume] context_proj set to identity (fallback)")


def _try_resume(accelerator, unet_wrapped, optimizer, lr_scheduler, ema, ckpt_root, ckpt_name, load_opt_state):
	ckpt_dir = _preferred_ckpt_dir(root=ckpt_root, name=ckpt_name, prefer_best=False)
	if not ckpt_dir:
		return 0, float("inf"), -1e9, 0, False, None

	unwrapped_inner = accelerator.unwrap_model(unet_wrapped).unet
	unwrapped_inner.load_adapter(ckpt_dir, adapter_name="default", is_trainable=True)
	if hasattr(unwrapped_inner, "set_adapter"):
		try:
			unwrapped_inner.set_adapter("default")
		except Exception:
			pass
	if hasattr(unwrapped_inner, "set_active_adapters"):
		try:
			unwrapped_inner.set_active_adapters(["default"])
		except Exception:
			pass

	state_path = os.path.join(ckpt_dir, "state.pt")
	start_epoch = 0
	best_val_loss = float("inf")
	best_val_score = -1e9
	patience_counter = 0

	if os.path.exists(state_path):
		state = torch.load(state_path, map_location="cpu")

		if "context_proj" in state and state["context_proj"]:
			accelerator.unwrap_model(unet_wrapped).load_state_dict(state["context_proj"], strict=False)
			accelerator.print("[resume] restored context_proj from state.pt")
		else:
			_identity_init_context_proj(accelerator, unet_wrapped)

		start_epoch = int(state.get("epoch", 0) + 1)
		best_val_loss = float(state.get("best_val_loss", float("inf")))
		best_val_score = float(state.get("best_val_score", -best_val_loss if best_val_loss != float("inf") else -1e9))
		patience_counter = int(state.get("patience_counter", 0))

		prev_hparams = state.get("opt_hparams", None)
		curr_hparams = [_optim_hparams(pg) for pg in optimizer.param_groups]
		mismatch = (prev_hparams is not None) and (prev_hparams != curr_hparams)

		if load_opt_state and not mismatch:
			optimizer.load_state_dict(state["optimizer"])
			lr_scheduler.load_state_dict(state["lr_scheduler"])
		else:
			msg = "[resume] Skipping optimizer/lr state restore"
			if mismatch:
				msg += " (detected hyperparameter changes)"
			accelerator.print(msg)

		ema_loaded = False
		if "ema_shadow" in state and state["ema_shadow"]:
			ema.keys = state.get("ema_keys", ema.keys)
			with torch.no_grad():
				sd = accelerator.unwrap_model(unet_wrapped).state_dict()
				if ema.keys and not any(k.startswith("unet.") for k in ema.keys):
					ema.keys = [("unet." + k) if k in sd else k for k in ema.keys]
			ema.shadow = {}
			for k in ema.keys:
				if k in state["ema_shadow"]:
					ema.shadow[k] = state["ema_shadow"][k]
					ema_loaded = True
		if not ema_loaded:
			with torch.no_grad():
				wrapped = accelerator.unwrap_model(unet_wrapped)
				sd = wrapped.state_dict()
				ema.keys = [k for k, v in sd.items() if "lora" in k.lower() and torch.is_floating_point(v)]
				ema.shadow = {k: sd[k].detach().cpu().clone() for k in ema.keys}
			accelerator.print("[resume] No EMA in checkpoint; synced EMA shadow to current WRAPPER model.")
		ema_ready = True
	else:
		accelerator.print(f"[resume] Found {ckpt_dir} but no state.pt; resuming weights only")
		sidecar = os.path.join(ckpt_dir, "context_proj.pt")
		if os.path.exists(sidecar):
			proj_sd = torch.load(sidecar, map_location="cpu")
			accelerator.unwrap_model(unet_wrapped).load_state_dict(proj_sd, strict=False)
			accelerator.print("[resume] restored context_proj from context_proj.pt")
		else:
			_identity_init_context_proj(accelerator, unet_wrapped)

		with torch.no_grad():
			wrapped = accelerator.unwrap_model(unet_wrapped)
			sd = wrapped.state_dict()
			ema.keys = [k for k, v in sd.items() if "lora" in k.lower() and torch.is_floating_point(v)]
			ema.shadow = {k: sd[k].detach().cpu().clone() for k in ema.keys}
		ema_ready = True

	accelerator.print(f"[resume] from: {ckpt_dir} (start_epoch={start_epoch})")
	return start_epoch, best_val_loss, best_val_score, patience_counter, ema_ready, ckpt_dir


# ----------------- main -----------------
def main():
	with open("config/config.yaml") as f:
		config = yaml.safe_load(f)

	def _abs(p: str) -> str:
		if p is None:
			return p
		return p if os.path.isabs(p) else os.path.abspath(os.path.join(PROJECT_ROOT, p))

	# normalize important paths
	config["MODEL"]["VAE_CHECKPOINT"] = _abs(config["MODEL"]["VAE_CHECKPOINT"])
	config["MODEL"]["LORA_CHECKPOINT"] = _abs(config["MODEL"]["LORA_CHECKPOINT"])
	config["MODEL"]["DIFFUSION_CHECKPOINT"] = _abs(config["MODEL"]["DIFFUSION_CHECKPOINT"])
	config["DATASET"]["LATENT_OUTPUT_PATH"] = _abs(config["DATASET"]["LATENT_OUTPUT_PATH"])

	accelerator = Accelerator(mixed_precision="bf16")
	device = accelerator.device

	# ---- config knobs ----
	SCALE = float(config.get("MODEL", {}).get("LATENT_SCALE", 0.18215))
	latent_path = config["DATASET"]["LATENT_OUTPUT_PATH"]
	batch_size = config["TRAINING"]["BATCH_SIZE"]
	grad_accum_steps = config["TRAINING"].get("GRAD_ACCUM_STEPS", 1)
	max_grad_norm = config["TRAINING"].get("MAX_GRAD_NORM", 1.0)
	train_on_three = config["TRAINING"].get("TRAIN_ON_THREE_BATCHES", False)
	chunk_limit = config["TRAINING"].get("CHUNK_LIMIT", 10)
	ssim_weight = config["TRAINING"].get("SSIM_WEIGHT", 0.0)
	beta = config["TRAINING"].get("BETA", 1.0)
	epochs = int(config["TRAINING"]["EPOCHS"])
	guidance_scale = float(config["TRAINING"].get("GUIDANCE_SCALE", 7.5))  # inference-only
	KTOK = int(config["TRAINING"].get("TEXT_TOKENS", 16))
	p_uncond = float(config["TRAINING"].get("CFG_DROPOUT", 0.05))
	ckpt_root = config["MODEL"]["LORA_CHECKPOINT"]
	ckpt_name = config["MODEL"]["LORA_CHECKPOINT_NAME"]
	resume_enabled = bool(config["MODEL"].get("LORA_RESUME", False))
	resume_load_opt = bool(config["MODEL"].get("LORA_RESUME_LOAD_OPT_STATE", False))
	warmup_epochs = int(config["TRAINING"].get("WARMUP_EPOCHS", 5))

	fast_tune = bool(config["TRAINING"].get("FAST_TUNE", False))
	score_only = bool(config["TRAINING"].get("SCORE_ONLY", False))
	disable_preview_recon = bool(config["TRAINING"].get("DISABLE_PREVIEW_RECON", False))
	disable_preview_cfg = bool(config["TRAINING"].get("DISABLE_PREVIEW_CFG", False))

	# Optional: run heavy preview diffusion passes every N epochs (0 disables)
	RUN_PREVIEWS_EVERY = int(config.get("TRAINING", {}).get("RUN_PREVIEWS_EVERY", 10))

	# FAST_TUNE defaults: make tuning cheap unless you explicitly override
	if fast_tune:
		if "DISABLE_PREVIEW_RECON" not in config.get("TRAINING", {}):
			disable_preview_recon = True
		if "DISABLE_PREVIEW_CFG" not in config.get("TRAINING", {}):
			disable_preview_cfg = True
		if "RUN_PREVIEWS_EVERY" not in config.get("TRAINING", {}):
			RUN_PREVIEWS_EVERY = 0


	# Dataloader worker knobs (define ONCE; used for train + val)
	num_workers = int(config.get("DATASET", {}).get("NUM_WORKERS", 14))
	prefetch_factor = 2

	tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
	text_encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to("cpu").eval()
	for p in text_encoder.parameters():
		p.requires_grad = False

	# Precompute null context once on CPU
	ntoks = tokenizer([""], padding="longest", truncation=True, max_length=64, return_tensors="pt")
	ninputs = {k: ntoks[k] for k in ("input_ids", "attention_mask", "token_type_ids") if k in ntoks}
	with torch.no_grad():
		null_ctx_cpu = text_encoder(**ninputs).last_hidden_state  # [1, L0, 768]

	# ---- UNet + LoRA ----
	unet_base = UNet2DConditionModel.from_pretrained(
		"CompVis/stable-diffusion-v1-4", subfolder="unet", cross_attention_dim=768
	)

	# ---- Build "maximal" LoRA target list: leaf names for Linear + Conv2d ----
	def build_max_lora_targets(model: torch.nn.Module):
		leaf = []
		for name, mod in model.named_modules():
			if isinstance(mod, (torch.nn.Linear, torch.nn.Conv2d)):
				leaf.append(name.split(".")[-1])
		return list(OrderedDict.fromkeys(leaf))

	# target_modules = build_max_lora_targets(unet_base)
	target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
	print(f"[lora] target_modules (max leaf) = {len(target_modules)}")

	lora_config = LoraConfig(
		r=int(config["TRAINING"]["LORA_R"]),
		lora_alpha=int(config["TRAINING"]["LORA_ALPHA"]),
		target_modules=target_modules,
		lora_dropout=float(config["TRAINING"]["LORA_DROPOUT"]),
		bias="none",
	)

	unet_lora = get_peft_model(unet_base, lora_config)
	lora_names = [n for n, _ in unet_lora.named_parameters() if "lora" in n.lower()]
	print("Total LoRA tensors:", len(lora_names))


	unet = LoRAUNetWrapper(unet_lora, context_dim_in=768, context_dim_out=768).to(device)

	# ---- Freeze everything except LoRA adapters + context_proj ----
	base = accelerator.unwrap_model(unet)

	for _, p in base.named_parameters():
		p.requires_grad = False

	lora_trainables = 0
	for n, p in base.named_parameters():
		if "lora" in n.lower():
			p.requires_grad = True
			lora_trainables += p.numel()

	ctx_trainables = 0
	for n, p in base.named_parameters():
		if n.startswith("context_proj."):
			p.requires_grad = True
			ctx_trainables += p.numel()

	print(f"[trainables] lora={lora_trainables:,} params | ctx_proj={ctx_trainables:,} params")

	extras = [
		n
		for n, p in base.named_parameters()
		if p.requires_grad and ("lora" not in n.lower()) and (not n.startswith("context_proj."))
	]
	if extras:
		raise RuntimeError(f"Unexpected non-LoRA trainables (should be empty): {extras[:40]}")

	lora_names = [n for n, _ in unet_lora.named_parameters() if "lora" in n.lower()]
	print(f"[lora] num_lora_param_tensors={len(lora_names)}")
	print("[lora] sample names:")
	for n in lora_names[:30]:
		print(" ", n)

	expected = ["to_q", "to_k", "to_v", "to_out", "proj_in", "proj_out", "ff", "conv", "time_embedding"]
	hits = {k: any(k in n for n in lora_names) for k in expected}
	print("[lora] hits:", hits)

	trainable = sum(p.numel() for p in unet_lora.parameters() if p.requires_grad)
	total = sum(p.numel() for p in unet_lora.parameters())
	print(f"[lora] trainable={trainable:,} total={total:,} frac={trainable/total:.6f}")

	with torch.no_grad():
		W = accelerator.unwrap_model(unet).context_proj
		if W.weight.abs().mean() < 1e-3:
			if W.weight.shape[0] == W.weight.shape[1]:
				nn.init.eye_(W.weight)
			else:
				nn.init.zeros_(W.weight)
				diag_len = min(W.weight.shape[0], W.weight.shape[1])
				for i in range(diag_len):
					W.weight[i, i] = 1.0
			if W.bias is not None:
				nn.init.zeros_(W.bias)
			accelerator.print("[init] context_proj set to identity (temporary fallback)")

	# ---- memory savers ----
	try:
		unet.enable_gradient_checkpointing()
	except Exception:
		pass
	for fn in ("enable_xformers_memory_efficiency", "enable_xformers_memory_efficient_attention"):
		if hasattr(unet, fn):
			try:
				getattr(unet, fn)()
				break
			except Exception:
				pass
	try:
		unet.set_attention_slice("auto")
	except Exception:
		pass
	torch.backends.cuda.matmul.allow_tf32 = True

	# ---- VAE ----
	vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
	vae.load_state_dict(torch.load(config["MODEL"]["VAE_CHECKPOINT"], map_location=device))
	vae = vae.to(device).eval()

	# ---- noise scheduler ----
	scheduler = DDPMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
	scheduler.config.num_train_timesteps = config["SCHEDULER"]["SAMPLING_STEPS"]
	scheduler.config.prediction_type = "epsilon"
	scheduler.set_timesteps(config["SCHEDULER"]["INFERENCE_STEPS"], device=device)
	accelerator.print(
		f"[scheduler] beta_schedule={getattr(scheduler.config,'beta_schedule',None)} "
		f"num_train_timesteps={scheduler.config.num_train_timesteps} "
		f"prediction_type={scheduler.config.prediction_type}"
	)

	train_T = scheduler.config.num_train_timesteps
	val_steps = config["SCHEDULER"]["INFERENCE_STEPS"]
	fixed_t = torch.linspace(0, train_T - 1, steps=val_steps, device=device).long()

	base = accelerator.unwrap_model(unet)
	lora_params, ctx_params = [], []
	for n, p in base.named_parameters():
		if not p.requires_grad:
			continue
		if n.startswith("context_proj."):
			ctx_params.append(p)
		elif "lora" in n.lower():
			lora_params.append(p)

	if not lora_params:
		raise RuntimeError("No LoRA params found as trainable. LoRA injection likely failed.")

	optimizer = torch.optim.AdamW(
		[
			{"params": lora_params, "lr": float(config["TRAINING"]["LEARNING_RATE"]), "weight_decay": 0.0},
			{"params": ctx_params,  "lr": 1e-5, "weight_decay": 1e-4},
		],
		betas=(0.9, 0.999),
		eps=1e-8,
		weight_decay=0.0,
		fused=True,
	)

	lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer,
		mode="min",
		factor=float(config["TRAINING"]["LR_SCHEDULER"]["FACTOR"]),
		patience=int(config["TRAINING"]["LR_SCHEDULER"]["PATIENCE"]),
		min_lr=float(config["TRAINING"]["LR_SCHEDULER"]["MIN_LR"]),
	)

	# ---- EMA (LoRA-only, CPU) ----
	ema = LoRAEMA(unet, decay=0.999)

	# ---- accelerate prepare ----
	unet, optimizer = accelerator.prepare(unet, optimizer)

	# ---- wandb ----
	# wandb.init(project=config["WANDB"]["PROJECT"], name=config["WANDB"]["RUN_NAME_LORA_UNET"], config=config)

	# ---- wandb (main process only) ----
	use_wandb = accelerator.is_main_process and (os.environ.get("WANDB_MODE", "").lower() != "disabled")
	wandb_run = None
	if use_wandb:
		if fast_tune:
			wandb_run = wandb.init(
				project=config["WANDB"]["PROJECT"],
				name=os.environ["WANDB_NAME"],
				config=config,
			)
		else:
			wandb_run = wandb.init(
				project=config["WANDB"]["PROJECT"],
				name=config["WANDB"]["RUN_NAME_LORA_UNET"],
				config=config,
			)

	# ---- resume (optional) ----
	if resume_enabled:
		start_epoch, best_val_loss, best_val_score, patience_counter, ema_ready, ckpt_dir = _try_resume(
			accelerator, unet, optimizer, lr_scheduler, ema, ckpt_root, ckpt_name, load_opt_state=resume_load_opt
		)
		if not resume_load_opt:
			start_epoch = 0
	else:
		start_epoch, best_val_loss, best_val_score, patience_counter, ckpt_dir = 0, float("inf"), -1e9, 0, None
		with torch.no_grad():
			sd = accelerator.unwrap_model(unet).state_dict()
			ema.keys = [k for k, v in sd.items() if "lora" in k.lower() and torch.is_floating_point(v)]
			ema.shadow = {k: sd[k].detach().cpu().clone() for k in ema.keys}
		ema_ready = True

	accelerator.print("=== run config ===")
	accelerator.print(f"resume={resume_enabled}  load_opt={resume_load_opt}  ckpt_root={ckpt_root} ckpt_name={ckpt_name}")
	if ckpt_dir:
		accelerator.print(f"[resume] dir={ckpt_dir}")
	accelerator.print(f"epochs={epochs}  batch_size={batch_size}  lr={optimizer.param_groups[0]['lr']}")

	val_gen = torch.Generator(device=device).manual_seed(42)

	p_noise_max = 0.1
	ramp_frac = 0.6

	# -------- training loop --------
	for epoch in range(start_epoch, epochs):
		accelerator.print(f"[train] epoch {epoch+1} starting...")
		unet.train()
		epoch_loss = 0.0
		num_batches = 0

		train_pred_std_sum, train_cos_sum = 0.0, 0.0
		lora_grad_norm_sum, lora_grad_norm_count = 0.0, 0
		train_tok_lens = []
		observed_drop = []

		chunk_id = 0
		while True:
			chunk_path = os.path.join(latent_path, f"latent_train_chunk_{chunk_id}")
			if not os.path.exists(chunk_path) or chunk_id >= chunk_limit:
				break

			dataset = load_from_disk(chunk_path).with_format("torch", columns=["z_target", "report", "ctx16"])

			loader = DataLoader(
				dataset,
				batch_size=batch_size,
				shuffle=True,
				pin_memory=True,
				num_workers=num_workers,
				persistent_workers=(num_workers > 0),
				prefetch_factor=prefetch_factor if num_workers > 0 else None,
			)
			loader = accelerator.prepare(loader)

			for i, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1} Chunk {chunk_id}", ascii=True)):
				if train_on_three and i == 3:
					break

				reports = batch["report"]
				if i == 0 and chunk_id == 0 and epoch == start_epoch:
					accelerator.print(f"[debug] batch keys: {list(batch.keys())}")
					accelerator.print(f"[debug] using ctx16 fast path = {'ctx16' in batch}")

				z_dataset = batch["z_target"].to(device).float().contiguous(memory_format=torch.channels_last)

				ramp_epochs = max(1, int(ramp_frac * epochs))
				progress = min(1.0, epoch / max(1, ramp_epochs - 1))
				p_noise = p_noise_max * progress

				B = z_dataset.size(0)
				mask = (torch.rand(B, device=device) < p_noise).view(B, *([1] * (z_dataset.ndim - 1)))
				z_random = torch.randn_like(z_dataset)
				z = torch.where(mask, z_random, z_dataset)

				T = scheduler.config.num_train_timesteps
				t = torch.randint(0, T, (B,), device=device)
				noise = torch.randn_like(z)
				z_noisy = scheduler.add_noise(z, noise, t).contiguous(memory_format=torch.channels_last)

				tokens = None
				outputs = None
				ctx_full = None
				unet_dtype = next(unet.parameters()).dtype

				if "ctx16" in batch:
					ctx16 = batch["ctx16"].to(device=device, dtype=unet_dtype, non_blocking=True)
					idx = torch.linspace(0, 15, steps=KTOK, device=device).round().long()
					ctx_cond = ctx16[:, idx, :]
				else:
					tokens = tokenizer(
						list(reports),
						padding="longest",
						truncation=True,
						max_length=64,
						return_tensors="pt",
						return_length=True,
					)
					bert_inputs = {
						k: tokens[k] for k in ("input_ids", "attention_mask", "token_type_ids", "length") if k in tokens
					}
					with torch.no_grad():
						outputs = text_encoder(**{k: bert_inputs[k] for k in bert_inputs if k != "length"})
					ctx_full = outputs.last_hidden_state
					attn = bert_inputs.get("attention_mask", None)
					ctx_cond = select_k_tokens(ctx_full, attn, k=KTOK).to(
						device=device, dtype=next(unet.parameters()).dtype, non_blocking=True
					)

				B = ctx_cond.size(0)
				base_null = null_ctx_cpu.mean(dim=1, keepdim=True)
				ctx_uncond = (
					base_null.to(device=device, dtype=ctx_cond.dtype, non_blocking=True)
					.expand(B, KTOK, base_null.size(-1))
				)

				with accelerator.autocast():
					if p_uncond > 0.0:
						mask = torch.rand(B, device=device) < p_uncond
						ctx_train = ctx_cond.clone()
						if mask.any():
							ctx_train[mask] = ctx_uncond[mask]
						observed_drop.extend(mask.float().tolist())
					else:
						ctx_train = ctx_cond
						observed_drop.extend([0.0] * B)

					pred = unet(z_noisy, t, ctx_train).sample
					loss, _, _ = composite_loss(pred, noise, ssim_weight=ssim_weight, beta=beta)

					loss /= grad_accum_steps

				accelerator.backward(loss)

				with torch.no_grad():
					train_pred_std_sum += float(pred.float().std().item())
					train_cos_sum += float(
						F.cosine_similarity(pred.float().flatten(1), noise.float().flatten(1)).mean().item()
					)

				step_now = ((i + 1) % grad_accum_steps == 0) or ((i + 1) == len(loader))
				if step_now:
					gn = grad_global_norm(list(iter_lora_params(unet)))
					lora_grad_norm_sum += gn
					lora_grad_norm_count += 1

					accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
					optimizer.step()
					optimizer.zero_grad()
					ema.update(unet)

				epoch_loss += loss.item() * grad_accum_steps
				num_batches += 1

				if isinstance(tokens, dict):
					if "length" in tokens:
						train_tok_lens += [int(x) for x in tokens["length"]]
					elif "attention_mask" in tokens:
						train_tok_lens += [int(m.sum().item()) for m in tokens["attention_mask"]]

				del z, reports, noise, t, z_noisy, tokens, outputs, ctx_full, ctx_cond, ctx_uncond, pred, loss

			chunk_id += 1

		avg_train_loss = epoch_loss / max(1, num_batches)
		avg_train_pred_std = train_pred_std_sum / max(1, num_batches)
		avg_train_cos = train_cos_sum / max(1, num_batches)
		avg_lora_gn = (lora_grad_norm_sum / max(1, lora_grad_norm_count)) if lora_grad_norm_count else 0.0
		avg_drop_ratio = float(sum(observed_drop) / max(1, len(observed_drop))) if observed_drop else 0.0
		accelerator.print(f"[train] epoch {epoch+1} done: avg_train_loss={avg_train_loss:.5f}")

		# -------- validation --------
		val_path = os.path.join(latent_path, "latent_val")
		val_decode_scale = read_scale_from_meta(val_path) or SCALE
		preview_scale = read_scale_from_meta(val_path) or SCALE
		accelerator.print(f"[val] decode scales: preview={preview_scale}  cfg={val_decode_scale}")

		val_dataset = load_from_disk(val_path).with_format("torch", columns=["z_target", "report", "ctx16"])
		val_loader = DataLoader(
			val_dataset,
			batch_size=batch_size,
			shuffle=False,
			pin_memory=True,
			num_workers=num_workers,
			persistent_workers=(num_workers > 0),
			prefetch_factor=prefetch_factor if num_workers > 0 else None,
		)
		val_loader = accelerator.prepare(val_loader)

		ema.store(unet)
		ema.copy_to(unet)
		used_ema = True

		unet.eval()
		val_loss_total, val_sl1_loss_total, val_ssim_loss_total = 0.0, 0.0, 0.0
		val_batches = 0
		val_images = [] if not score_only else []
		cfg_images = [] if not score_only else []

		val_pred_std_sum, val_cos_sum = 0.0, 0.0
		t_vals = []
		val_tok_lens = []
		val_latent_means = []
		val_latent_stds = []
		pair_psnrs, pair_ssims = [], []
		cfg_log = {}

		run_previews = (RUN_PREVIEWS_EVERY > 0) and (((epoch + 1) % RUN_PREVIEWS_EVERY) == 0)

		if fast_tune or score_only:
			run_previews = False

		with torch.no_grad():
			for i, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")):
				if train_on_three and i == 3:
					break

				z = batch["z_target"].to(device).float()
				reports = batch["report"]

				val_latent_means.append(float(z.mean().item()))
				val_latent_stds.append(float(z.std().item()))

				ti = torch.randint(0, len(fixed_t), (z.size(0),), generator=val_gen, device=device).long()
				t_infer = fixed_t[ti]
				noise = make_deterministic_noise(
					shape=z.shape, device=z.device, dtype=z.dtype, seed=12345 + epoch * 100_000 + i
				)
				z_noisy = scheduler.add_noise(z, noise, t_infer)

				tokens = None
				unet_dtype = next(unet.parameters()).dtype

				if "ctx16" in batch:
					ctx16 = batch["ctx16"].to(device=device, dtype=unet_dtype, non_blocking=True)
					idx = torch.linspace(0, 15, steps=KTOK, device=device).round().long()
					ctx = ctx16[:, idx, :]
				else:
					tokens = tokenizer(
						list(reports),
						padding="longest",
						truncation=True,
						max_length=64,
						return_tensors="pt",
						return_length=True,
					)
					bert_inputs = {
						k: tokens[k] for k in ("input_ids", "attention_mask", "token_type_ids", "length") if k in tokens
					}
					outputs = text_encoder(**{k: bert_inputs[k] for k in bert_inputs if k != "length"})
					ctx_full = outputs.last_hidden_state
					attn = bert_inputs.get("attention_mask", None)
					ctx = select_k_tokens(ctx_full, attn, k=KTOK).to(
						device=device, dtype=next(unet.parameters()).dtype, non_blocking=True
					)

				pred = unet(z_noisy, t_infer, ctx).sample
				val_loss, val_sl1_loss, val_ssim_loss = composite_loss(pred, noise, ssim_weight=ssim_weight, beta=beta)

				val_loss_total += val_loss.item()
				val_sl1_loss_total += val_sl1_loss
				val_ssim_loss_total += val_ssim_loss
				val_batches += 1

				val_pred_std_sum += float(pred.float().std().item())
				val_cos_sum += float(F.cosine_similarity(pred.float().flatten(1), noise.float().flatten(1)).mean().item())
				t_vals += [int(x) for x in t_infer.tolist()]

				if isinstance(tokens, dict):
					if "length" in tokens:
						val_tok_lens += [int(x) for x in tokens["length"]]
					elif "attention_mask" in tokens:
						val_tok_lens += [int(m.sum().item()) for m in tokens["attention_mask"]]

				# --- CFG diagnostics (first val batch only) ---
				if i == 0:
					N = min(4, z.size(0))
					z_s = z_noisy[:N]
					t_s = t_infer[:N]
					noise_s = noise[:N]
					ctx_s = ctx[:N]

					base_null = null_ctx_cpu.mean(dim=1, keepdim=True)
					ctx_u_s = base_null.to(device=device, dtype=ctx.dtype).expand(
						N, KTOK, base_null.size(-1)
					)

					eps_c = unet(z_s, t_s, ctx_s).sample
					eps_u = unet(z_s, t_s, ctx_u_s).sample

					delta_norm = (eps_c - eps_u).flatten(1).norm(dim=1).mean().item()

					cfg_log["cfg/delta_norm"] = delta_norm
					cfg_log["cfg/uncond_prob_target"] = p_uncond

					# Only compute extra diagnostics if NOT score-only
					if not score_only:
						cond_loss = F.smooth_l1_loss(eps_c, noise_s, beta=beta).item()
						uncond_loss = F.smooth_l1_loss(eps_u, noise_s, beta=beta).item()
						eps_cos = F.cosine_similarity(
							eps_c.flatten(1), eps_u.flatten(1)
						).mean().item()

						guided_losses = {}
						for s in (3.0, 7.5):
							eps_g = eps_u + s * (eps_c - eps_u)
							guided_losses[s] = F.smooth_l1_loss(
								eps_g, noise_s, beta=beta
							).item()

						cfg_log.update(
							{
								"cfg/cond_loss_val": cond_loss,
								"cfg/uncond_loss_val": uncond_loss,
								"cfg/eps_cos": eps_cos,
								"cfg/guided_loss_s3": guided_losses[3.0],
								"cfg/guided_loss_s7p5": guided_losses[7.5],
							}
						)

				# === PREVIEW A: reconstruction (no CFG) ===
				PREVIEW_SAMPLES = 3
				if run_previews and (not disable_preview_recon) and (len(val_images) < PREVIEW_SAMPLES):
					B = z.shape[0]
					take = min(B, PREVIEW_SAMPLES)
					ts = scheduler.timesteps
					if ts.numel() > 1 and ts[-1] > ts[0]:
						ts = ts.flip(0)
					for k in range(take):
						z_k = z[k : k + 1]
						zt_k = z_noisy[k : k + 1]
						ctx_k = ctx[k : k + 1]
						rep_k = reports[k]
						t_k_val = t_infer[k]

						dec_gpu = decode_latents(z_k, vae, scale=preview_scale)
						dec = dec_gpu[0].detach().cpu()
						del dec_gpu

						idxs = (ts == t_k_val).nonzero(as_tuple=True)[0]
						start_idx = int(idxs[0].item()) if idxs.numel() > 0 else int((ts - t_k_val).abs().argmin().item())

						z_hat = zt_k.clone().to(torch.float32)
						for t_step in ts[start_idx:]:
							t_b = t_step[None].to(device)
							z_hat_in = z_hat.to(dtype=next(unet.parameters()).dtype)
							eps_step = unet(z_hat_in, t_b, ctx_k).sample.to(torch.float32)
							z_hat = scheduler.step(model_output=eps_step, timestep=t_step, sample=z_hat).prev_sample
							del t_b, z_hat_in, eps_step

						full_gpu = decode_latents(z_hat, vae, scale=preview_scale)
						full = full_gpu[0].detach().cpu()
						del z_hat, full_gpu

						dec_gf = dec.mean(dim=0, keepdim=True).unsqueeze(0)
						full_gf = full.mean(dim=0, keepdim=True).unsqueeze(0)
						pair_psnrs.append(float(psnr01(dec_gf, full_gf).item()))
						pair_ssims.append(float(ssim(dec_gf, full_gf, data_range=1.0, size_average=True).item()))

						H = min(dec.shape[1], full.shape[1])
						W = min(dec.shape[2], full.shape[2])
						dec_u8 = to_grayscale(dec[:, :H, :W])
						full_u8 = to_grayscale(full[:, :H, :W])
						sep = 255 * torch.ones_like(dec_u8[:, :, :1])
						panel = torch.cat([dec_u8, sep, full_u8], dim=2)

						val_images.append(
							wandb.Image(panel, caption=f"decoded(z) full diff (from t={int(t_k_val.item())})\n{rep_k}")
						)

						del dec, full, dec_gf, full_gf, dec_u8, full_u8, sep, panel

						if len(val_images) >= PREVIEW_SAMPLES:
							break

				# === PREVIEW B: guided samples (uses config GUIDANCE_SCALE) ===
				if run_previews and (not disable_preview_cfg) and (len(cfg_images) < PREVIEW_SAMPLES):
					N = min(PREVIEW_SAMPLES - len(cfg_images), z.size(0))
					sample_imgs = sample_from_model(
						unet=unet,
						vae=vae,
						tokenizer=tokenizer,
						text_encoder=text_encoder,
						scheduler=scheduler,
						device=device,
						reports=[reports[j] for j in range(N)],
						guidance_scale=guidance_scale,
						scale=val_decode_scale,
						KTOK=KTOK,
						batched_cfg=False,
					)

					for j in range(N):
						img = to_grayscale(sample_imgs[j].detach().cpu())
						cfg_images.append(wandb.Image(img, caption=f"CFG s={guidance_scale} | {reports[j]}"))
					del sample_imgs

				del z, reports, noise, t_infer, z_noisy, tokens, ctx, pred, val_loss, val_sl1_loss, val_ssim_loss

		avg_val_loss = val_loss_total / max(1, val_batches)
		avg_val_sl1_loss = val_sl1_loss_total / max(1, val_batches)
		avg_val_ssim_loss = val_ssim_loss_total / max(1, val_batches)
		avg_val_pred_std = val_pred_std_sum / max(1, val_batches) if val_batches else 0.0
		avg_val_cos = val_cos_sum / max(1, val_batches) if val_batches else 0.0

		if score_only:
			recon_ssim_mean = 0.0
			recon_psnr_mean = 0.0
		else:
			recon_ssim_mean = (sum(pair_ssims) / len(pair_ssims)) if pair_ssims else 0.0
			recon_psnr_mean = (sum(pair_psnrs) / len(pair_psnrs)) if pair_psnrs else 0.0

		cfg_delta_norm = float(cfg_log.get("cfg/delta_norm", 0.0))
		val_score = compute_val_score(
			avg_val_loss=avg_val_loss,
			avg_val_cos=avg_val_cos,
			avg_val_pred_std=avg_val_pred_std,
			cfg_delta_norm=cfg_delta_norm,
		)


		def _mean_p95(xs):
			if not xs:
				return 0.0, 0.0
			xs_sorted = sorted(xs)
			p95_idx = int(0.95 * (len(xs_sorted) - 1))
			return float(sum(xs_sorted) / len(xs_sorted)), float(xs_sorted[p95_idx])

		train_len_mean, train_len_p95 = _mean_p95(train_tok_lens)
		val_len_mean, val_len_p95 = _mean_p95(val_tok_lens)
		lat_mean = sum(val_latent_means) / max(1, len(val_latent_means)) if val_latent_means else 0.0
		lat_std = sum(val_latent_stds) / max(1, len(val_latent_stds)) if val_latent_stds else 0.0


		log_data = {
			"model/epoch": epoch + 1,
			"train/cfg_drop_ratio": avg_drop_ratio,
			"train/loss": avg_train_loss,
			"train/p_noise": float(p_noise),
			"val/loss": avg_val_loss,
			"val/sl1_loss": avg_val_sl1_loss,
			"val/ssim_loss": avg_val_ssim_loss,
			"val/score": float(val_score),
			"train/pred_std": avg_train_pred_std,
			"val/pred_std": avg_val_pred_std,
			"train/cos_pred_noise": avg_train_cos,
			"val/cos_pred_noise": avg_val_cos,
			"train/cfg_uncond_prob": p_uncond,
			"val/t_mean": (sum(t_vals) / max(1, len(t_vals))) if t_vals else 0.0,
			"text/train_len_mean": train_len_mean,
			"text/train_len_p95": train_len_p95,
			"text/val_len_mean": val_len_mean,
			"text/val_len_p95": val_len_p95,
			"grad/lora_global_norm": avg_lora_gn,
			"latents/val_mean": lat_mean,
			"latents/val_std": lat_std,
			"images/val_samples": val_images,
			"images/cfg_samples": cfg_images,
			"val/used_ema": int(used_ema),
		}
		log_data["val/recon_psnr_mean"] = float(recon_psnr_mean)
		log_data["val/recon_ssim_mean"] = float(recon_ssim_mean)
		log_data["train/lr_ctx"] = optimizer.param_groups[1]["lr"]
		log_data["train/lr_lora"] = optimizer.param_groups[0]["lr"]
		log_data["val/run_previews"] = int(run_previews)
		log_data.update(cfg_log)

		if use_wandb:
			wandb.log(log_data)
		accelerator.print(
			f"[val]   epoch {epoch+1}: avg_val_loss={avg_val_loss:.5f} val_score={val_score:.5f} used_ema={used_ema}"
		)

		# Keep LR scheduler on true val loss (stable), regardless of tuning objective
		if epoch >= warmup_epochs:
			lr_scheduler.step(avg_val_loss)

		# IMPORTANT: always restore raw (non-EMA) weights after val
		if used_ema:
			ema.restore(unet)

		# save best by val_score (maximize) + rolling checkpoints
		if epoch == start_epoch or val_score > best_val_score:
			best_val_score = val_score
			best_val_loss = avg_val_loss
			patience_counter = 0
			_save_best_checkpoint(
				accelerator, unet, optimizer, lr_scheduler, ema, ckpt_root, epoch, best_val_loss, best_val_score, patience_counter
			)
		else:
			patience_counter += 1

		_save_checkpoint(
			accelerator, unet, optimizer, lr_scheduler, ema, ckpt_root, epoch, best_val_loss, best_val_score, patience_counter
		)

		if patience_counter >= config["TRAINING"]["EARLY_STOP_PATIENCE"]:
			accelerator.print("[info] Early stopping triggered.")
			break

		# after training loop ends

		_tune_report(
			val_score=val_score,
			recon_ssim_mean=recon_ssim_mean,
			recon_psnr_mean=recon_psnr_mean,
			val_loss=avg_val_loss,
			epoch=epoch + 1,
		)

	if use_wandb and wandb_run is not None:
		wandb.finish()


if __name__ == "__main__":
	main()