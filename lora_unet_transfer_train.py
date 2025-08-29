# lora_unet_transfer_train.py
import os
import glob
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
import torch.nn as nn
import torch.nn.functional as F


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
	KTOK=4,
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

		model_dtype = next(unet.parameters()).dtype  # e.g., torch.float16 on GPU
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

		# scalar or per-sample CFG, but keep in float32
		if torch.is_tensor(guidance_scale):
			gs = guidance_scale.to(device=device, dtype=torch.float32).view(-1, *([1] * 3))  # [B,1,1,1]
		else:
			gs = torch.tensor(float(guidance_scale), device=device, dtype=torch.float32)

		for t in timesteps:
			tt = torch.full((B,), t, device=device, dtype=torch.long)

			# UNet forward at model dtype; keep scheduler math in float32
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
			eps = eps_u + gs * delta  # works for scalar or per-sample gs

			z = scheduler.step(model_output=eps, timestep=t, sample=z).prev_sample  # float32 path

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
		# handle accelerate/DDP wrappers
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
def _preferred_ckpt_dir(root, prefer_best=True):
	best = os.path.join(root, "best")
	if prefer_best and os.path.isdir(best):
		return best
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


def _save_checkpoint(accelerator, unet_wrapped, optimizer, lr_scheduler, ema, ckpt_root, epoch, best_val_loss, patience_counter):
	accelerator.wait_for_everyone()
	os.makedirs(ckpt_root, exist_ok=True)
	save_dir = os.path.join(ckpt_root, f"epoch{epoch+1:04d}")
	os.makedirs(save_dir, exist_ok=True)

	unwrapped = accelerator.unwrap_model(unet_wrapped).unet
	unwrapped.save_pretrained(save_dir)

	# also save the wrapper's context_proj weights
	wrapped = accelerator.unwrap_model(unet_wrapped)
	proj_sd = {k: v.detach().cpu() for k, v in wrapped.state_dict().items() if k.startswith("context_proj.")}

	state = {
		"epoch": epoch,
		"best_val_loss": best_val_loss,
		"patience_counter": patience_counter,
		"optimizer": optimizer.state_dict(),
		"lr_scheduler": lr_scheduler.state_dict(),
		"ema_shadow": {k: v.clone() for k, v in ema.shadow.items()},
		"ema_keys": list(ema.keys),
		"opt_hparams": [_optim_hparams(pg) for pg in optimizer.param_groups],
		"context_proj": proj_sd,
	}
	torch.save(state, os.path.join(save_dir, "state.pt"))
	accelerator.print(f"[save] wrote {save_dir}")


def _save_best_checkpoint(
	accelerator, unet_wrapped, optimizer, lr_scheduler, ema, ckpt_root, epoch, best_val_loss, patience_counter
):
	accelerator.wait_for_everyone()
	best_path = os.path.join(ckpt_root, "best")
	os.makedirs(best_path, exist_ok=True)

	# Save EMA weights into best/
	ema.store(unet_wrapped)   # backup RAW train weights
	ema.copy_to(unet_wrapped) # swap EMA into the model

	# LoRA adapter (inner UNet) to best/
	accelerator.unwrap_model(unet_wrapped).unet.save_pretrained(best_path)

	# Persist wrapper extras (e.g., context_proj) + full state for true resume
	wrapped = accelerator.unwrap_model(unet_wrapped)
	proj_sd = {k: v.detach().cpu() for k, v in wrapped.state_dict().items() if k.startswith("context_proj.")}

	state = {
		"epoch": epoch,
		"best_val_loss": best_val_loss,
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
	torch.save(proj_sd, os.path.join(best_path, "context_proj.pt"))  # optional sidecar

	# Restore RAW training weights so training continues correctly
	ema.restore(unet_wrapped)

	accelerator.print(f"[save] updated best (EMA + state.pt) -> {best_path}")


def _identity_init_context_proj(accelerator, unet_wrapped):
	with torch.no_grad():
		wrapped = accelerator.unwrap_model(unet_wrapped)
		if hasattr(wrapped, "context_proj"):
			W = wrapped.context_proj
			nn.init.zeros_(W.weight)
			# handle rectangular
			diag_len = min(W.weight.shape[0], W.weight.shape[1])
			for i in range(diag_len):
				W.weight[i, i] = 1.0
			if W.bias is not None:
				nn.init.zeros_(W.bias)
			accelerator.print("[resume] context_proj set to identity (fallback)")


def _try_resume(accelerator, unet_wrapped, optimizer, lr_scheduler, ema, ckpt_root, load_opt_state):
	ckpt_dir = _preferred_ckpt_dir(ckpt_root, prefer_best=True)
	if not ckpt_dir:
		return 0, float("inf"), 0, False, None

	# Load LoRA adapter weights into INNER UNet
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

		# EMA
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
	return start_epoch, best_val_loss, patience_counter, ema_ready, ckpt_dir


# ----------------- main -----------------
def main():
	with open("config/config.yaml") as f:
		config = yaml.safe_load(f)

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
	KTOK = int(config["TRAINING"].get("TEXT_TOKENS", 4))
	p_uncond = float(config["TRAINING"].get("CFG_DROPOUT", .05))
	ckpt_root = config["MODEL"]["LORA_CHECKPOINT"]
	resume_enabled = bool(config["MODEL"].get("LORA_RESUME", False))
	resume_load_opt = bool(config["MODEL"].get("LORA_RESUME_LOAD_OPT_STATE", False))

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
	lora_config = LoraConfig(
		r=config["TRAINING"]["LORA_R"],
		lora_alpha=config["TRAINING"]["LORA_ALPHA"],
		target_modules=[
			# attention
			"to_q","to_k","to_v","to_out.0",
			# spatial transformer 1x1
			"proj_in","proj_out",
			# spatial transformer FFN
			"ff.net.0.proj","ff.net.2",
			# UNet resnet convs
			"conv1","conv2","conv_shortcut",
			# time embedding MLP
			"time_embedding.linear_1","time_embedding.linear_2",
		],
		lora_dropout=config["TRAINING"]["LORA_DROPOUT"],
		bias="none",
	)
	unet_lora = get_peft_model(unet_base, lora_config)
	unet = LoRAUNetWrapper(unet_lora, context_dim_in=768, context_dim_out=768).to(device)

	# --- identity context_proj if effectively uninitialized ---
	with torch.no_grad():
		W = accelerator.unwrap_model(unet).context_proj
		if W.weight.abs().mean() < 1e-3:
			nn.init.eye_(W.weight)
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
	scheduler.config.num_train_timesteps = 1000
	scheduler.config.prediction_type = "epsilon"
	scheduler.set_timesteps(500, device=device)
	accelerator.print(
		f"[scheduler] beta_schedule={getattr(scheduler.config,'beta_schedule',None)} "
		f"num_train_timesteps={scheduler.config.num_train_timesteps} "
		f"prediction_type={scheduler.config.prediction_type}"
	)

	# --- allow context_proj to learn
	for p in accelerator.unwrap_model(unet).context_proj.parameters():
		p.requires_grad = True

	# --- split parameter groups: LoRA vs context_proj
	base = accelerator.unwrap_model(unet)
	lora_params, ctx_params = [], []
	for n, p in base.named_parameters():
		if not p.requires_grad:
			continue
		if n.startswith("context_proj."):
			ctx_params.append(p)
		else:
			# trainable LoRA adapters have "lora" in their names (peft)
			if "lora" in n.lower():
				lora_params.append(p)

	optimizer = torch.optim.AdamW(
		[
			{"params": lora_params, "lr": float(config["TRAINING"]["LEARNING_RATE"]), "weight_decay": 0.0},
			{"params": ctx_params,  "lr": 1e-5, "weight_decay": 1e-4},
		],
		betas=(0.9, 0.999),
		eps=1e-8,
		weight_decay=0.0,
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
	wandb.init(project=config["WANDB"]["PROJECT"], name=config["WANDB"]["RUN_NAME_LORA_UNET"], config=config)

	# ---- resume (optional) ----
	if resume_enabled:
		start_epoch, best_val_loss, patience_counter, ema_ready, ckpt_dir = _try_resume(
			accelerator, unet, optimizer, lr_scheduler, ema, ckpt_root, load_opt_state=resume_load_opt
		)
	else:
		start_epoch, best_val_loss, patience_counter, ckpt_dir = 0, float("inf"), 0, None
		with torch.no_grad():
			sd = accelerator.unwrap_model(unet).state_dict()
			ema.keys = [k for k, v in sd.items() if "lora" in k.lower() and torch.is_floating_point(v)]
			ema.shadow = {k: sd[k].detach().cpu().clone() for k in ema.keys}
		ema_ready = True

	accelerator.print("=== run config ===")
	accelerator.print(f"resume={resume_enabled}  load_opt={resume_load_opt}  ckpt_root={ckpt_root}")
	if ckpt_dir:
		accelerator.print(f"[resume] dir={ckpt_dir}")
	accelerator.print(f"epochs={epochs}  batch_size={batch_size}  lr={optimizer.param_groups[0]['lr']}")

	# ---- deterministic validation setup ----
	val_gen = torch.Generator(device=device).manual_seed(42)
	fixed_t_idx = torch.linspace(0, len(scheduler.timesteps) - 1, steps=50, device=device).long()
	fixed_t = scheduler.timesteps[fixed_t_idx]

	# -------- training loop --------
	for epoch in range(start_epoch, epochs):
		accelerator.print(f"[train] epoch {epoch+1} starting...")
		unet.train()
		epoch_loss = 0.0
		num_batches = 0

		# train stats
		train_pred_std_sum, train_cos_sum = 0.0, 0.0
		lora_grad_norm_sum, lora_grad_norm_count = 0.0, 0
		train_tok_lens = []
		observed_drop = []

		# iterate chunks
		chunk_id = 0
		while True:
			chunk_path = os.path.join(latent_path, f"latent_train_chunk_{chunk_id}")
			if not os.path.exists(chunk_path) or chunk_id >= chunk_limit:
				break

			dataset = load_from_disk(chunk_path).with_format("torch", columns=["z_target", "report"])
			loader = accelerator.prepare(DataLoader(dataset, batch_size=batch_size, shuffle=True))

			for i, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1} Chunk {chunk_id}")):
				if train_on_three and i == 3:
					break

				z = batch["z_target"].to(device).float().contiguous(memory_format=torch.channels_last)
				reports = batch["report"]
				
				T = scheduler.config.num_train_timesteps
				B = z.size(0)
				t = torch.randint(0, T, (B,), device=device)        # uniform over [0, T)
				noise = torch.randn_like(z)
				z_noisy = scheduler.add_noise(z, noise, t).contiguous(memory_format=torch.channels_last)

				# ---- CPU tokenize + encode, reduce to KTOK ----
				tokens = tokenizer(
					list(reports),
					padding="longest",
					truncation=True,
					max_length=64,
					return_tensors="pt",
					return_length=True,
				)
				bert_inputs = {k: tokens[k] for k in ("input_ids", "attention_mask", "token_type_ids", "length") if k in tokens}
				with torch.no_grad():
					outputs = text_encoder(**{k: bert_inputs[k] for k in bert_inputs if k != "length"})
				ctx_full = outputs.last_hidden_state
				attn = bert_inputs.get("attention_mask", None)
				ctx_cond = select_k_tokens(ctx_full, attn, k=KTOK).to(
					device=device, dtype=next(unet.parameters()).dtype, non_blocking=True
				)

				# ---- unconditional context from precomputed null -> mean then repeat to KTOK
				B = ctx_cond.size(0)
				base_null = null_ctx_cpu.mean(dim=1, keepdim=True)  # [1,1,D] on CPU
				ctx_uncond = (
					base_null.to(device=device, dtype=ctx_cond.dtype, non_blocking=True)
					.expand(B, KTOK, base_null.size(-1))
				)

				# ---- CFG DROPOUT (single forward)
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
					loss = F.smooth_l1_loss(pred, noise, beta=beta) / grad_accum_steps

				accelerator.backward(loss)

				with torch.no_grad():
					train_pred_std_sum += float(pred.float().std().item())
					train_cos_sum += float(F.cosine_similarity(pred.float().flatten(1), noise.float().flatten(1)).mean().item())

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

				if "length" in tokens:
					train_tok_lens += [int(x) for x in tokens["length"]]
				elif "attention_mask" in tokens:
					train_tok_lens += [int(m.sum().item()) for m in tokens["attention_mask"]]

				del z, reports, noise, t, z_noisy, tokens, outputs, ctx_full, ctx_cond, ctx_uncond, pred, loss
				torch.cuda.empty_cache()

			chunk_id += 1

		# train epoch aggregates
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
		val_dataset = load_from_disk(val_path).with_format("torch", columns=["z_target", "report"])
		val_loader = accelerator.prepare(DataLoader(val_dataset, batch_size=batch_size))

		# Always evaluate/sampling with EMA weights
		ema.store(unet)
		ema.copy_to(unet)
		used_ema = True

		unet.eval()
		val_loss_total = 0.0
		val_batches = 0
		val_images = []
		cfg_images = []

		val_pred_std_sum, val_cos_sum = 0.0, 0.0
		t_vals = []
		val_tok_lens = []
		val_latent_means = []
		val_latent_stds = []
		pair_psnrs, pair_ssims = [], []
		cfg_log = {}

		with torch.no_grad():
			for i, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")):
				if train_on_three and i == 3:
					break

				z = batch["z_target"].to(device).float()
				reports = batch["report"]

				val_latent_means.append(float(z.mean().item()))
				val_latent_stds.append(float(z.std().item()))

				# deterministic noise/timestep
				ti = torch.randint(0, len(fixed_t), (z.size(0),), generator=val_gen, device=device).long()
				t_infer = fixed_t[ti]
				noise = make_deterministic_noise(shape=z.shape, device=z.device, dtype=z.dtype, seed=12345 + epoch * 100_000 + i)
				z_noisy = scheduler.add_noise(z, noise, t_infer)

				# encode (CPU) and select tokens
				tokens = tokenizer(
					list(reports),
					padding="longest",
					truncation=True,
					max_length=64,
					return_tensors="pt",
					return_length=True,
				)
				bert_inputs = {k: tokens[k] for k in ("input_ids", "attention_mask", "token_type_ids", "length") if k in tokens}
				outputs = text_encoder(**{k: bert_inputs[k] for k in bert_inputs if k != "length"})
				ctx_full = outputs.last_hidden_state
				attn = bert_inputs.get("attention_mask", None)
				ctx = select_k_tokens(ctx_full, attn, k=KTOK).to(
					device=device, dtype=next(unet.parameters()).dtype, non_blocking=True
				)

				pred = unet(z_noisy, t_infer, ctx).sample
				val_sl1 = F.smooth_l1_loss(pred, noise, beta=beta)

				val_loss_total += val_sl1.item()
				val_batches += 1

				val_pred_std_sum += float(pred.float().std().item())
				val_cos_sum += float(F.cosine_similarity(pred.float().flatten(1), noise.float().flatten(1)).mean().item())
				t_vals += [int(x) for x in t_infer.tolist()]

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
					ctx_u_s = base_null.to(device=device, dtype=ctx.dtype, non_blocking=True).expand(N, KTOK, base_null.size(-1))

					eps_c = unet(z_s, t_s, ctx_s).sample
					eps_u = unet(z_s, t_s, ctx_u_s).sample

					cond_loss = F.smooth_l1_loss(eps_c, noise_s, beta=beta).item()
					uncond_loss = F.smooth_l1_loss(eps_u, noise_s, beta=beta).item()
					eps_cos = F.cosine_similarity(eps_c.flatten(1), eps_u.flatten(1)).mean().item()
					delta_norm = (eps_c - eps_u).flatten(1).norm(dim=1).mean().item()

					guided_losses = {}
					for s in (3.0, 7.5):
						eps_g = eps_u + s * (eps_c - eps_u)
						guided_losses[s] = F.smooth_l1_loss(eps_g, noise_s, beta=beta).item()

					cfg_log.update(
						{
							"cfg/cond_loss_val": cond_loss,
							"cfg/uncond_loss_val": uncond_loss,
							"cfg/eps_cos": eps_cos,
							"cfg/delta_norm": delta_norm,
							"cfg/guided_loss_s3": guided_losses[3.0],
							"cfg/guided_loss_s7p5": guided_losses[7.5],
							"cfg/uncond_prob_target": p_uncond,
						}
					)

				# === PREVIEW A: reconstruction (no CFG)
				PREVIEW_SAMPLES = 3
				if len(val_images) < PREVIEW_SAMPLES:
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
							wandb.Image(
								panel,
								caption=f"decoded(z) full diff (from t={int(t_k_val.item())})\n{rep_k}",
							)
						)

						del dec, full, dec_gf, full_gf, dec_u8, full_u8, sep, panel

						if len(val_images) >= PREVIEW_SAMPLES:
							break

				# === PREVIEW B: guided samples (uses config GUIDANCE_SCALE) ===
				if len(cfg_images) < PREVIEW_SAMPLES:
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

				del z, reports, noise, t_infer, z_noisy, tokens, ctx, pred, val_sl1
				torch.cuda.empty_cache()

		avg_val_loss = val_loss_total / max(1, val_batches)
		avg_val_pred_std = val_pred_std_sum / max(1, val_batches) if val_batches else 0.0
		avg_val_cos = val_cos_sum / max(1, val_batches) if val_batches else 0.0

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
			"val/loss": avg_val_loss,
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
			"val/recon_psnr_mean": (sum(pair_psnrs) / len(pair_psnrs)) if pair_psnrs else 0.0,
			"val/recon_ssim_mean": (sum(pair_ssims) / len(pair_ssims)) if pair_ssims else 0.0,
			"val/used_ema": int(used_ema),
		}
		log_data["train/lr_ctx"]  = optimizer.param_groups[1]["lr"]
		log_data["train/lr_lora"] = optimizer.param_groups[0]["lr"]
		log_data.update(cfg_log)
		wandb.log(log_data)
		accelerator.print(f"[val]   epoch {epoch+1}: avg_val_loss={avg_val_loss:.5f} used_ema={used_ema}")

		lr_scheduler.step(avg_val_loss)
		if used_ema:
			ema.restore(unet)

		# save best (EMA + state) + rolling checkpoints
		if epoch == start_epoch or avg_val_loss < best_val_loss:
			best_val_loss = avg_val_loss
			patience_counter = 0
			_save_best_checkpoint(accelerator, unet, optimizer, lr_scheduler, ema, ckpt_root, epoch, best_val_loss, patience_counter)
		else:
			patience_counter += 1

		_save_checkpoint(accelerator, unet, optimizer, lr_scheduler, ema, ckpt_root, epoch, best_val_loss, patience_counter)

		if patience_counter >= config["TRAINING"]["EARLY_STOP_PATIENCE"]:
			accelerator.print("[info] Early stopping triggered.")
			break


if __name__ == "__main__":
	main()
