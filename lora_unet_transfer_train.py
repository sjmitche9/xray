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

def composite_loss(pred, target, ssim_weight=0.0, beta=1.0):
	sl1 = torch.nn.functional.smooth_l1_loss(pred, target, beta=beta)
	pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-5)
	target_norm = (target - target.min()) / (target.max() - target.min() + 1e-5)
	ssim_loss = 1.0 - ssim(pred_norm, target_norm, data_range=1.0, size_average=True)
	return sl1 + ssim_weight * ssim_loss, sl1.item(), ssim_loss.item()

def decode_latents(latents, vae):
	with torch.no_grad():
		latents = latents.to(next(vae.parameters()).device)
		latents = (latents - latents.mean()) / (latents.std() + 1e-5)
		latents = latents.clamp(-5, 5)
		images = vae.decode(latents).sample.clamp(0, 1)
	return images

def to_grayscale(img_tensor):
	img = img_tensor.mean(dim=0, keepdim=True).repeat(3, 1, 1)  # [3, H, W]
	img = (img * 255).clamp(0, 255).byte()
	return img

def sample_from_model(unet, vae, tokenizer, text_encoder, scheduler, device, reports):
	with torch.no_grad():
		tokens = tokenizer(list(reports), padding="max_length", truncation=True, max_length=128, return_tensors="pt").to(device)
		context = text_encoder(**tokens).last_hidden_state
		z = torch.randn((len(reports), 4, 32, 32), device=device)

		for t in reversed(range(scheduler.config.num_train_timesteps)):
			t_tensor = torch.full((z.size(0),), t, device=device, dtype=torch.long)
			pred_noise = unet(z, t_tensor, context).sample
			beta_t = scheduler.betas[t].to(device)
			alpha_t = scheduler.alphas[t].to(device)
			alpha_hat_t = scheduler.alphas_cumprod[t].to(device)
			z = (1 / alpha_t.sqrt()) * (z - ((1 - alpha_t) / (1 - alpha_hat_t).sqrt()) * pred_noise)
			if t > 0:
				z += beta_t.sqrt() * torch.randn_like(z)

			del t_tensor, pred_noise, beta_t, alpha_t, alpha_hat_t
			torch.cuda.empty_cache()

		decoded = vae.decode(z).sample.clamp(0, 1)
		del z, tokens, context
		torch.cuda.empty_cache()
	return decoded

def main():
	with open("config/config.yaml") as f:
		config = yaml.safe_load(f)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	accelerator = Accelerator()

	tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
	text_encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)
	for param in text_encoder.parameters():
		param.requires_grad = False

	unet = UNet2DConditionModel.from_pretrained(
		"CompVis/stable-diffusion-v1-4", subfolder="unet", cross_attention_dim=768
	)

	lora_config = LoraConfig(
		r=config["TRAINING"]["LORA_R"],
		lora_alpha=config["TRAINING"]["LORA_ALPHA"],
		target_modules=["to_q", "to_k", "to_v"],
		lora_dropout=config["TRAINING"]["LORA_DROPOUT"],
		bias="none",
	)
	unet = get_peft_model(unet, lora_config)
	unet = LoRAUNetWrapper(unet, context_dim_in=768, context_dim_out=768).to(device)

	vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
	vae.load_state_dict(torch.load(config["MODEL"]["VAE_CHECKPOINT"]))
	vae = vae.to(device).eval()

	scheduler = DDPMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
	# keep SD's training horizon intact (1000). use 100 inference steps, but make sure they're on the right device
	scheduler.config.num_train_timesteps = 1000
	scheduler.set_timesteps(100, device=device)  # <-- important: device=device



	optimizer = torch.optim.AdamW(unet.parameters(), lr=float(config["TRAINING"]["LEARNING_RATE"]))
	lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer,
		mode='min',
		factor=float(config["TRAINING"]["LR_SCHEDULER"]["FACTOR"]),
		patience=float(config["TRAINING"]["LR_SCHEDULER"]["PATIENCE"]),
		min_lr=float(config["TRAINING"]["LR_SCHEDULER"]["MIN_LR"])
	)

	unet, optimizer = accelerator.prepare(unet, optimizer)

	wandb.init(project=config["WANDB"]["PROJECT"], name=config["WANDB"]["RUN_NAME_LORA_UNET"], config=config)

	latent_path = config["DATASET"]["LATENT_OUTPUT_PATH"]
	batch_size = config["TRAINING"]["BATCH_SIZE"]
	grad_accum_steps = config["TRAINING"].get("GRAD_ACCUM_STEPS", 1)
	max_grad_norm = config["TRAINING"].get("MAX_GRAD_NORM", 1.0)
	train_on_three_batches = config["TRAINING"].get("TRAIN_ON_THREE_BATCHES", False)
	chunk_limit = config["TRAINING"].get("CHUNK_LIMIT", 10)
	ssim_weight = config["TRAINING"].get("SSIM_WEIGHT", 0.0)
	beta = config["TRAINING"].get("BETA", 1.0)
	epochs = config["TRAINING"]["EPOCHS"]

	best_val_loss = float('inf')
	patience_counter = 0

	for epoch in range(epochs):
		unet.train()
		epoch_loss = 0
		num_batches = 0

		chunk_id = 0
		while True:
			chunk_path = os.path.join(latent_path, f"latent_train_chunk_{chunk_id}")
			if not os.path.exists(chunk_path) or chunk_id >= chunk_limit:
				break

			dataset = load_from_disk(chunk_path)
			dataset = dataset.with_format("torch", columns=["z_target", "report"])
			loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
			loader = accelerator.prepare(loader)

			for i, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1} Chunk {chunk_id}")):
				if train_on_three_batches and i == 3:
					break

				z = batch["z_target"].to(device)
				reports = batch["report"]
				noise = torch.randn_like(z)
				t = torch.randint(0, scheduler.config.num_train_timesteps, (z.size(0),), device=device).long()
				z_noisy = scheduler.add_noise(z, noise, t)

				tokens = tokenizer(list(reports), padding="max_length", truncation=True, max_length=128, return_tensors="pt").to(device)
				ctx = text_encoder(**tokens).last_hidden_state

				pred = unet(z_noisy, t, ctx).sample
				loss, sl1_val, ssim_val = composite_loss(pred, noise, ssim_weight=ssim_weight, beta=beta)
				loss = loss / grad_accum_steps
				accelerator.backward(loss)

				if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(loader):
					accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
					optimizer.step()
					optimizer.zero_grad()

				epoch_loss += loss.item() * grad_accum_steps
				num_batches += 1

				del z, reports, noise, t, z_noisy, tokens, ctx, pred, loss
				torch.cuda.empty_cache()

			chunk_id += 1

		avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0

		val_path = os.path.join(latent_path, "latent_val")
		val_dataset = load_from_disk(val_path)
		val_dataset = val_dataset.with_format("torch", columns=["z_target", "report"])
		val_loader = DataLoader(val_dataset, batch_size=batch_size)
		val_loader = accelerator.prepare(val_loader)

		unet.eval()
		val_loss = 0
		val_batches = 0
		val_l1_total = 0
		val_ssim_total = 0
		val_images = []

		with torch.no_grad():
			for i, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")):
				if train_on_three_batches and i == 3:
					break

				z = batch["z_target"].to(device)
				reports = batch["report"]
				noise = torch.randn_like(z)
				ti = torch.randint(0, len(scheduler.timesteps), (z.size(0),), device=device).long()
				t_infer = scheduler.timesteps[ti]
				z_noisy = scheduler.add_noise(z, noise, t_infer)

				tokens = tokenizer(list(reports), padding="max_length", truncation=True, max_length=128, return_tensors="pt").to(device)
				ctx = text_encoder(**tokens).last_hidden_state

				pred = unet(z_noisy, t_infer, ctx).sample
				loss, sl1_val, ssim_val = composite_loss(pred, noise, ssim_weight=ssim_weight, beta=beta)

				val_loss += loss.item()
				val_l1_total += sl1_val
				val_ssim_total += ssim_val
				val_batches += 1

				recon_latent = scheduler.step(
					model_output=pred[:1],
					timestep=t_infer[:1],
					sample=z_noisy[:1]
				).prev_sample

				if len(val_images) < 3:
					# decode GT latent and the one-step reconstructed latent
					orig_img = decode_latents(z[:1], vae)[0].detach().cpu()          # [3, H, W] in [0,1]
					recon_img = decode_latents(recon_latent, vae)[0].detach().cpu()  # [3, H, W] in [0,1]

					# (optional) grayscale so panels look uniform with CXRs
					orig_img = to_grayscale(orig_img)    # -> [3, H, W], uint8
					recon_img = to_grayscale(recon_img)  # -> [3, H, W], uint8

					# make sure shapes match (defensive in case of tiny rounding diffs)
					H = min(orig_img.shape[1], recon_img.shape[1])
					W = min(orig_img.shape[2], recon_img.shape[2])
					orig_img = orig_img[:, :H, :W]
					recon_img = recon_img[:, :H, :W]

					# thin separator
					sep = 255 * torch.ones_like(orig_img[:, :, :1])  # white 1‑px bar

					# side‑by‑side: decoded | recon
					panel = torch.cat([orig_img, sep, recon_img], dim=2)  # concat width-wise

					val_images.append(
						wandb.Image(panel, caption=f"decoded | recon  —  {reports[0]}")
					)

				del z, reports, noise, t_infer, z_noisy, tokens, ctx, pred, loss
				torch.cuda.empty_cache()

		avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
		avg_val_l1 = val_l1_total / val_batches if val_batches > 0 else 0
		avg_val_ssim = val_ssim_total / val_batches if val_batches > 0 else 0

		# sample_reports = ["cardiomegaly, edema, view: ap", "no finding, view: ap", "pleural effusion, consolidation, view: ap"]
		# samples = sample_from_model(unet, vae, tokenizer, text_encoder, scheduler, device, sample_reports)
		# sample_images = [wandb.Image(to_grayscale(img.cpu()), caption=rep) for img, rep in zip(samples, sample_reports)]
		# del samples
		torch.cuda.empty_cache()

		log_data = {
			"model/epoch": epoch + 1,
			"train/loss": avg_train_loss,
			"val/loss": avg_val_loss,
			"val/smooth_l1": avg_val_l1,
			"val/ssim_loss": avg_val_ssim,
			"train/smooth_l1": sl1_val,
			"train/ssim_loss": ssim_val,
			"model/lr": optimizer.param_groups[0]["lr"],
			"images/val_samples": val_images,
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