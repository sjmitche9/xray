# app.py
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from peft import get_peft_model, LoraConfig
from models.lora_unet_wrapper import LoRAUNetWrapper
import yaml
import os

latent_stats_file = "data/processed_dataset/latent/dataset_latent_stats.pt"
latent_stats = torch.load(latent_stats_file, map_location="cuda")
mean_latent = latent_stats["mean"].to("cuda")
std_latent = latent_stats["std"].to("cuda")


def load_config(path="config/config.yaml"):
	"""
	Load YAML config and return as dict.
	Raises an error if file is missing or malformed.
	"""
	if not os.path.exists(path):
		raise FileNotFoundError(f"Config file not found: {path}")
	
	with open(path, "r") as f:
		config = yaml.safe_load(f)
	
	if not isinstance(config, dict):
		raise ValueError("Config file did not parse as a dictionary.")
	
	return config


# Example usage:
config = load_config("config/config.yaml")
vae_ckpt = config["MODEL"]["VAE_CHECKPOINT"]
lora_ckpt = os.path.join(config["MODEL"]["LORA_CHECKPOINT"], config["MODEL"]["LORA_CHECKPOINT_NAME"])
latent_scale = float(config.get("MODEL", {}).get("LATENT_SCALE", 0.18215))
KTOK = int(config["TRAINING"].get("TEXT_TOKENS", 16))
infer_timesteps = config["SCHEDULER"]["INFERENCE_STEPS"]
lora_r = config["TRAINING"]["LORA_R"]
lora_alpha = config["TRAINING"]["LORA_ALPHA"]
lora_dropout = config["TRAINING"]["LORA_DROPOUT"]


# ----------------- Utilities -----------------
def make_deterministic_noise(shape, device, dtype, seed=None):
	if seed is not None:
		g = torch.Generator(device="cpu").manual_seed(int(seed))
		n = torch.randn(shape, generator=g, dtype=torch.float32, device="cpu")
		return n.to(device=device, dtype=dtype, non_blocking=True)
	return torch.randn(shape, device=device, dtype=dtype)

def select_k_tokens(ctx_full, attn_mask=None, k=4):
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

def decode_latents(latents, vae, scale=0.18215):
	with torch.no_grad():
		latents = latents.to(next(vae.parameters()).device)
		img = vae.decode(latents / scale).sample
		img = (img + 1) / 2
		return img.clamp(0, 1)

def to_grayscale(img_tensor, auto_contrast=True, eps=1e-6):
	img = img_tensor.mean(dim=0, keepdim=True)
	if auto_contrast:
		mn = img.min()
		mx = img.max()
		img = (img - mn) / (mx - mn + eps)
	img = img.repeat(3, 1, 1)
	return (img * 255).clamp(0, 255).byte()

# ----------------- Image Generation -----------------
@st.cache_resource(show_spinner=False)
def load_models(KTOK=16, device="cuda"):
	# Tokenizer + Text encoder
	tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
	text_encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to("cpu").eval()
	for p in text_encoder.parameters():
		p.requires_grad = False

	# Precompute null context
	ntoks = tokenizer([""], padding="longest", truncation=True, max_length=64, return_tensors="pt")
	ninputs = {k: ntoks[k] for k in ("input_ids", "attention_mask", "token_type_ids") if k in ntoks}
	with torch.no_grad():
		null_ctx_cpu = text_encoder(**ninputs).last_hidden_state

	# UNet + LoRA
	unet_base = UNet2DConditionModel.from_pretrained(
		"CompVis/stable-diffusion-v1-4", subfolder="unet", cross_attention_dim=768
	)
	lora_config = LoraConfig(
		r=lora_r,
		lora_alpha=lora_alpha,
		target_modules=[
			"to_q","to_k","to_v","to_out.0",
			"proj_in","proj_out",
			"ff.net.0.proj","ff.net.2",
			"conv1","conv2","conv_shortcut",
			"time_embedding.linear_1","time_embedding.linear_2",
		],
		lora_dropout=lora_dropout,
		bias="none",
	)
	unet_lora = get_peft_model(unet_base, lora_config)
	unet = LoRAUNetWrapper(unet_lora, context_dim_in=768, context_dim_out=768).to(device)
	unet.unet.load_adapter(os.path.abspath(lora_ckpt), adapter_name="default", is_trainable=False)

	# VAE
	vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
	vae.load_state_dict(torch.load(vae_ckpt, map_location=device))
	vae = vae.to(device).eval()

	# Scheduler
	scheduler = DDPMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
	scheduler.set_timesteps(infer_timesteps, device=device)

	return tokenizer, text_encoder, unet, vae, scheduler, null_ctx_cpu


@torch.no_grad()
def sample_single(
    unet,
    vae,
    tokenizer,
    text_encoder,
    scheduler,
    device,
    prompt: str,
    mean_latent,
    std_latent,
    alpha: float = 0.3,      # 0 = mostly dataset latent, 1 = fully random
    guidance_scale: float = 7.5,
    KTOK: int = 16,
    scale: float = 0.18215,
):
    # --- text encoding ---
    tokens = tokenizer([prompt], padding="longest", truncation=True, max_length=64, return_tensors="pt")
    bert_inputs = {k: tokens[k] for k in ("input_ids", "attention_mask", "token_type_ids") if k in tokens}
    outputs = text_encoder(**bert_inputs)
    ctx_full = outputs.last_hidden_state
    attn = bert_inputs.get("attention_mask", None)
    model_dtype = next(unet.parameters()).dtype
    ctx_cond = select_k_tokens(ctx_full, attn, k=KTOK).to(device=device, dtype=model_dtype, non_blocking=True)

    # --- unconditional context ---
    null_tokens = tokenizer([""], padding="longest", truncation=True, max_length=64, return_tensors="pt")
    null_inputs = {k: null_tokens[k] for k in ("input_ids", "attention_mask", "token_type_ids") if k in null_tokens}
    null_ctx = text_encoder(**null_inputs).last_hidden_state
    base_null = null_ctx.mean(dim=1, keepdim=True)
    ctx_uncond = base_null.expand(ctx_cond.size(0), KTOK, base_null.size(-1)).to(device=device, dtype=model_dtype)

    # --- hybrid latent ---
    B = ctx_cond.size(0)
    z_random = torch.randn((B, 4, 32, 32), device=device, dtype=torch.float32) * float(scheduler.init_noise_sigma)
    z_dataset = mean_latent + std_latent * torch.randn_like(mean_latent)
    z = alpha * z_random + (1 - alpha) * z_dataset

    # --- scheduler timesteps ---
    timesteps = scheduler.timesteps.to(device)
    if timesteps.numel() > 1 and timesteps[-1] > timesteps[0]:
        timesteps = timesteps.flip(0)
    gs = torch.tensor(float(guidance_scale), device=device, dtype=torch.float32)

    # --- diffusion loop ---
    for t in timesteps:
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        z_in = z.to(dtype=model_dtype)

        eps_c = unet(z_in, t_tensor, ctx_cond).sample.to(torch.float32)
        eps_u = unet(z_in, t_tensor, ctx_uncond).sample.to(torch.float32)
        eps = eps_u + gs * (eps_c - eps_u)

        z = scheduler.step(model_output=eps, timestep=t, sample=z).prev_sample

    # --- decode ---
    decoded = decode_latents(z, vae, scale=scale)
    img_gray = to_grayscale(decoded[0], auto_contrast=True)
    return img_gray



# ----------------- Streamlit UI -----------------
st.title("Clinical Report → X-ray Image")

# Device & hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
KTOK = 16
SCALE = 0.18215

# Load models
tokenizer, text_encoder, unet, vae, scheduler, null_ctx_cpu = load_models(KTOK=KTOK, device=device)


guidance_scale = st.slider("CFG Guidance Scale", min_value=0, max_value=10, value=2, step=1)
alpha = st.slider("Random vs Dataset Similar Latent", min_value=0.0, max_value=1.0, value=1.0, step=0.1)

# Define all possible conditions and views
conditions = ["atelectasis", "cardiomegaly", "consolidation", "edema", "enlarged cardiomediastinum", "fracture", "lung lesion", "lung opacity", "no finding", "pleural effusion", "pleural other", "pneumonia", "pneumothorax"]

views = ["ap", "ap axial", "ap lld", "ap rld", "lao", "lateral", "ll", "lpo", "nan", "pa", "pa lld", "pa rld", "rao", "swimmers", "xtable lateral", "support devices"]

if "selected_conditions" not in st.session_state:
	st.session_state.selected_conditions = set()

def toggle_condition(cond):
	if cond in st.session_state.selected_conditions:
		st.session_state.selected_conditions.remove(cond)
	else:
		st.session_state.selected_conditions.add(cond)

st.markdown("### Conditions")
# for cond in conditions:
# 	active = cond in st.session_state.selected_conditions
# 	if st.button(("✅ " if active else "➕ ") + cond, key=f"cond_{cond}"):
# 		toggle_condition(cond)

cols_per_row = 4
for i in range(0, len(conditions), cols_per_row):
    cols = st.columns(cols_per_row)
    for j, cond in enumerate(conditions[i:i+cols_per_row]):
        active = cond in st.session_state.selected_conditions
        if cols[j].button(("✅ " if active else "➕ ") + cond, key=f"cond_{cond}"):
            toggle_condition(cond)

selected_view = st.radio("### View", views, index=None, horizontal=True)

# Build report string
report_parts = []
if st.session_state.selected_conditions:
	report_parts.extend(sorted(st.session_state.selected_conditions))
if selected_view:
	report_parts.append(f"view: {selected_view.lower()}")

report = ", ".join(report_parts) if report_parts else "no finding"

# baseline report (no conditions, but same view)
baseline_report = None
if selected_view:
	baseline_report = f"no finding, view: {selected_view.lower()}"

# --- Show reports ---
if report:
	st.markdown(f"**Conditioned report:** `{report}`")
if baseline_report:
	st.markdown(f"**Baseline report:** `{baseline_report}`")

# --- Generate images ---
if report and baseline_report:
	col1, col2 = st.columns(2)
	with col1:
		st.write("Conditioned")
		conditioned_img = sample_single(
			unet=unet,
			vae=vae,
			tokenizer=tokenizer,
			text_encoder=text_encoder,
			scheduler=scheduler,
			device=device,
			prompt=report,
			mean_latent=mean_latent,
			std_latent=std_latent,
			alpha=alpha,
			guidance_scale=guidance_scale,
			KTOK=KTOK,
			scale=SCALE
		)
		st.image(conditioned_img.permute(1, 2, 0).cpu().numpy(), caption=report, width=330)

	with col2:
		st.write("Baseline")
		baseline_img = sample_single(
			unet=unet,
			vae=vae,
			tokenizer=tokenizer,
			text_encoder=text_encoder,
			scheduler=scheduler,
			device=device,
			prompt=baseline_report,
			mean_latent=mean_latent,
			std_latent=std_latent,
			alpha=alpha,
			guidance_scale=guidance_scale,
			KTOK=KTOK,
			scale=SCALE
		)
		st.image(baseline_img.permute(1, 2, 0).cpu().numpy(), caption=baseline_report, width=330)