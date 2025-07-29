# train_pixel_diffusion_model.py
import os
import yaml
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_from_disk
from skimage.metrics import peak_signal_noise_ratio as psnr
from models.diffusion import DiffusionModel
from pytorch_msssim import ssim


def local_variance_penalty(img, eps=1e-6):
	# [B, 1, H, W] → local variance per image
	var = torch.var(img, dim=[2, 3])  # over H and W
	return torch.mean(1.0 / (var + eps))  # higher penalty for lower variance


def combined_loss(pred, target, alpha=1.0, beta=0.2, gamma=0.1):
	mse_loss = F.smooth_l1_loss(pred, target)
	ssim_loss = 1.0 - ssim(pred, target, data_range=1.0, size_average=True)
	blob_penalty = local_variance_penalty(pred)
	return alpha * mse_loss + beta * ssim_loss + gamma * blob_penalty


def main():
	with open("config/config.yaml") as f:
		config = yaml.safe_load(f)

	output_path = config["DATASET"]["OUTPUT_PATH"]
	val_path = os.path.join(output_path, "val")

	batch_size = config["TRAINING"]["BATCH_SIZE"]
	epochs = config["TRAINING"]["EPOCHS"]
	lr = float(config["TRAINING"]["LEARNING_RATE"])
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	min_save_epoch = config["TRAINING"].get("MIN_SAVE_EPOCH", 5)
	early_stop_patience = config["TRAINING"].get("EARLY_STOP_PATIENCE", 5)
	checkpoint_path = config["MODEL"]["DIFFUSION_CHECKPOINT"]
	resume = config["TRAINING"].get("DIFFUSION_RESUME", False)
	num_workers = config["TRAINING"].get("NUM_WORKERS", 4)
	train_on_one_batch = config["TRAINING"].get("TRAIN_ON_ONE_BATCH", False)
	chunk_limit = config["TRAINING"].get("CHUNK_LIMIT", 36)

	# --- Gradient Accumulation + Clipping Settings ---
	grad_accum_steps = config["TRAINING"].get("GRAD_ACCUM_STEPS", 1)
	max_grad_norm = config["TRAINING"].get("MAX_GRAD_NORM", 1.0)

	val_data = load_from_disk(val_path)
	val_data.set_format("torch", columns=["image", "report"])
	val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers)

	model = DiffusionModel(config).to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
	scaler = torch.amp.GradScaler("cuda")
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer,
		mode='min',
		factor=float(config["TRAINING"]["LR_SCHEDULER"].get("FACTOR", 0.5)),
		patience=float(config["TRAINING"]["LR_SCHEDULER"].get("PATIENCE", 2)),
		min_lr=float(config["TRAINING"]["LR_SCHEDULER"].get("MIN_LR", 1e-6)),
	)

	start_epoch = 0
	best_val_loss = float("inf")
	patience_counter = 0

	if resume and os.path.exists(checkpoint_path):
		checkpoint = torch.load(checkpoint_path)
		model.load_state_dict(checkpoint["model_state"])
		optimizer.load_state_dict(checkpoint["optimizer_state"])
		start_epoch = checkpoint.get("epoch", 0)
		best_val_loss = checkpoint.get("best_val_loss", float("inf"))
		print(f"[info] Resumed from checkpoint at epoch {start_epoch}")

	wandb.init(project=config["WANDB"]["PROJECT"], name=config["WANDB"]["RUN_NAME_DIFFUSION"], config=config)

	timesteps = list(range(model.scheduler.num_timesteps))
	wandb.log({
		"schedule/beta": wandb.plot.line_series(xs=timesteps, ys=[model.scheduler.beta.cpu()], keys=["beta"], title="Beta Schedule", xname="t"),
		"schedule/alpha": wandb.plot.line_series(xs=timesteps, ys=[model.scheduler.alpha.cpu()], keys=["alpha"], title="Alpha Schedule", xname="t"),
		"schedule/alpha_hat": wandb.plot.line_series(xs=timesteps, ys=[model.scheduler.alpha_hat.cpu()], keys=["alpha_hat"], title="Alpha Hat Schedule", xname="t"),
		"model/guidance_scale": model.guidance_scale
	})

	for epoch in range(start_epoch, epochs):
		model.train()
		train_loss = 0
		total_batches = 0
		chunk_id = 0

		while True:
			chunk_path = os.path.join(output_path, f"train_chunk_{chunk_id}")
			if not os.path.exists(chunk_path) or chunk_id == chunk_limit:
				break

			train_data = load_from_disk(chunk_path).shuffle(seed=42)
			train_data.set_format("torch", columns=["image", "report"])
			train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

			optimizer.zero_grad(set_to_none=True)

			for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Chunk {chunk_id}")):
				if train_on_one_batch and i == 1:
					break

				x = batch["image"].to(device)
				noise = torch.randn_like(x)
				reports = batch["report"]

				with torch.amp.autocast(device_type='cuda'):
					pred, target = model(x, noise, reports)
					loss = combined_loss(pred, target) / grad_accum_steps

				scaler.scale(loss).backward()

				if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_loader):
					scaler.unscale_(optimizer)
					torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
					scaler.step(optimizer)
					scaler.update()
					optimizer.zero_grad(set_to_none=True)

				train_loss += loss.item() * grad_accum_steps
				total_batches += 1

			chunk_id += 1

		avg_train_loss = train_loss / total_batches if total_batches > 0 else 0

		# --- Validation ---
		model.eval()
		val_loss = psnr_score = ssim_score = blob_loss = 0

		with torch.no_grad():
			for i, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")):
				if train_on_one_batch and i == 1:
					break

				x = batch["image"].to(device)
				noise = torch.randn_like(x)
				reports = batch["report"]

				pred, target = model(x, noise, reports)
				loss = combined_loss(pred, target)
				val_loss += loss.item()

				blob_loss += local_variance_penalty(pred).item()
				recon = (x - noise + pred).clamp(0, 1)

				for j in range(recon.size(0)):
					psnr_score += psnr(x[j].cpu().numpy(), recon[j].cpu().numpy(), data_range=1)
					ssim_score += ssim(x[j].unsqueeze(0), recon[j].unsqueeze(0), data_range=1.0).item()

		avg_val_loss = val_loss / len(val_loader)
		avg_psnr = psnr_score / len(val_data)
		avg_ssim = ssim_score / len(val_data)
		avg_blob_loss = blob_loss / len(val_loader)

		# context = model.text_encoder(reports).to(device)
		_, images_by_step = model.sample(reports[:3], x[:3].shape, device, raw=True)
		final_images = [img for (t, img) in images_by_step if t == 0]
		gen_stack = torch.cat(final_images, dim=0)

		log_data = {
			"model/epoch": epoch + 1,
			"train/avg_loss": avg_train_loss,
			"val/avg_loss": avg_val_loss,
			"val/psnr": avg_psnr,
			"val/ssim": avg_ssim,
			"val/blob_loss": avg_blob_loss,
			# "val/context_mean": context.mean().item(),
			# "val/context_std": context.std().item(),
		}

		if epoch % 3 == 0:
			with torch.no_grad():
				report_sample = reports[:3]
				shape_sample = x[:3].shape
				_, images_by_step = model.sample( 
					report_texts=report_sample,
					latent_shape=shape_sample,
					device=device,
					raw=True,
					return_intermediates=[499, 199, 0]
				)

				intermediate_dict = {t: img for t, img in images_by_step}
				intermediate_images = []

				for j in range(3):
					for t in [499, 199, 0]:
						img = intermediate_dict[t][j].cpu().clamp(0, 1)
						caption = f"sample {j} | timestep {t} | report: {reports[j]}"
						intermediate_images.append(wandb.Image(img, caption=caption))

				log_data["val/intermediate_strip"] = intermediate_images
				final_images = intermediate_dict[0]
				gen_stack = final_images

				log_data["val/generated_samples"] = [
					wandb.Image(img.cpu().clamp(0, 1), caption=str(rep))
					for img, rep in zip(final_images, reports[:3])
				]
				log_data["val/generated_std"] = gen_stack.std().item()

				recon_images = []
				for j in range(3):
					real = x[j].cpu()
					fake = final_images[j].cpu()
					pair = torch.cat([real, fake], dim=2)
					recon_images.append(wandb.Image(pair.clamp(0, 1), caption=f"val sample {j}"))
				log_data["val/reconstructions"] = recon_images

				guidance_images = []
				for scale in [5, 7.5, 10.0, 12.5, 15.0, 17.5]:
					model.guidance_scale = scale
					_, g_steps = model.sample(reports[:1], x[:1].shape, device, raw=True, return_intermediates=[0])
					g_img = g_steps[0][1][0]
					guidance_images.append(wandb.Image(g_img.cpu().clamp(0, 1), caption=f"guidance_scale={scale}"))

				log_data["val/guidance_sweep"] = guidance_images

		current_lr = optimizer.param_groups[0]["lr"]
		log_data["train/lr"] = current_lr
		wandb.log(log_data)
		scheduler.step(avg_val_loss)

		if avg_val_loss < best_val_loss and (epoch + 1) >= min_save_epoch:
			best_val_loss = avg_val_loss
			patience_counter = 0
			torch.save({
				"model_state": model.state_dict(),
				"optimizer_state": optimizer.state_dict(),
				"epoch": epoch + 1,
				"best_val_loss": best_val_loss
			}, checkpoint_path)
			print(f"[info] Model improved. Saved checkpoint to {checkpoint_path}")
		elif (epoch + 1) >= min_save_epoch:
			patience_counter += 1
			print(f"[info] No improvement. Patience: {patience_counter}/{early_stop_patience}")
			if patience_counter >= early_stop_patience:
				print("[info] Early stopping triggered.")
				break

if __name__ == "__main__":
	main()



# import os
# import yaml
# import torch
# import torch.nn.functional as F
# import wandb
# from tqdm import tqdm
# from torch.utils.data import DataLoader
# from datasets import load_from_disk
# from skimage.metrics import peak_signal_noise_ratio as psnr
# from models.diffusion import DiffusionModel
# from pytorch_msssim import ssim


# def local_variance_penalty(img, eps=1e-6):
# 	# [B, 1, H, W] → local variance per image
# 	var = torch.var(img, dim=[2, 3])  # over H and W
# 	return torch.mean(1.0 / (var + eps))  # higher penalty for lower variance


# def combined_loss(pred, target, alpha=1.0, beta=0.5, gamma=0.1):
# 	"""
# 	Combines MSE loss and SSIM loss.

# 	Args:
# 		pred: predicted image tensor [B, 1, H, W]
# 		target: ground-truth image tensor [B, 1, H, W]
# 		alpha: weight for MSE
# 		beta: weight for SSIM

# 	Returns:
# 		scalar loss
# 	"""
# 	mse_loss = F.mse_loss(pred, target)
# 	ssim_loss = 1.0 - ssim(pred, target, data_range=1.0, size_average=True)
# 	blob_penalty = local_variance_penalty(pred)

# 	return alpha * mse_loss + beta * ssim_loss + gamma * blob_penalty


# def main():
# 	# --- Load Config ---
# 	with open("config/config.yaml") as f:
# 		config = yaml.safe_load(f)

# 	output_path = config["DATASET"]["OUTPUT_PATH"]
# 	val_path = os.path.join(output_path, "val")

# 	batch_size = config["TRAINING"]["BATCH_SIZE"]
# 	epochs = config["TRAINING"]["EPOCHS"]
# 	lr = float(config["TRAINING"]["LEARNING_RATE"])
# 	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 	min_save_epoch = config["TRAINING"].get("MIN_SAVE_EPOCH", 5)
# 	early_stop_patience = config["TRAINING"].get("EARLY_STOP_PATIENCE", 5)
# 	checkpoint_path = config["MODEL"]["DIFFUSION_CHECKPOINT"]
# 	resume = config["TRAINING"].get("DIFFUSION_RESUME", False)
# 	num_workers = config["TRAINING"].get("NUM_WORKERS", 4)
# 	train_on_one_batch = config["TRAINING"].get("TRAIN_ON_ONE_BATCH", False)
# 	chunk_limit = config["TRAINING"].get("CHUNK_LIMIT", 36)

# 	val_data = load_from_disk(val_path)
# 	val_data.set_format("torch", columns=["image", "report"])
# 	val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers)

# 	model = DiffusionModel(config).to(device)
# 	optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
# 	scaler = torch.amp.GradScaler("cuda")
# 	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
# 		optimizer,
# 		mode='min',
# 		factor=float(config["TRAINING"]["LR_SCHEDULER"].get("FACTOR", 0.5)),
# 		patience=float(config["TRAINING"]["LR_SCHEDULER"].get("PATIENCE", 2)),
# 		min_lr=float(config["TRAINING"]["LR_SCHEDULER"].get("MIN_LR", 1e-6)),
# 	)

# 	start_epoch = 0
# 	best_val_loss = float("inf")
# 	patience_counter = 0

# 	if resume and os.path.exists(checkpoint_path):
# 		checkpoint = torch.load(checkpoint_path)
# 		model.load_state_dict(checkpoint["model_state"])
# 		optimizer.load_state_dict(checkpoint["optimizer_state"])
# 		start_epoch = checkpoint.get("epoch", 0)
# 		best_val_loss = checkpoint.get("best_val_loss", float("inf"))
# 		print(f"[info] Resumed from checkpoint at epoch {start_epoch}")

# 	wandb.init(project=config["WANDB"]["PROJECT"], name=config["WANDB"]["RUN_NAME_DIFFUSION"], config=config)

# 	# --- Log noise schedule ---
# 	timesteps = list(range(model.scheduler.num_timesteps))
# 	wandb.log({
# 		"schedule/beta": wandb.plot.line_series(xs=timesteps, ys=[model.scheduler.beta.cpu()], keys=["beta"], title="Beta Schedule", xname="t"),
# 		"schedule/alpha": wandb.plot.line_series(xs=timesteps, ys=[model.scheduler.alpha.cpu()], keys=["alpha"], title="Alpha Schedule", xname="t"),
# 		"schedule/alpha_hat": wandb.plot.line_series(xs=timesteps, ys=[model.scheduler.alpha_hat.cpu()], keys=["alpha_hat"], title="Alpha Hat Schedule", xname="t"),
# 		"model/guidance_scale": model.guidance_scale
# 	})

# 	for epoch in range(start_epoch, epochs):
# 		model.train()
# 		train_loss = 0
# 		total_batches = 0
# 		chunk_id = 0

# 		while True:
# 			chunk_path = os.path.join(output_path, f"train_chunk_{chunk_id}")

# 			if not os.path.exists(chunk_path):
# 				break

# 			if chunk_id == chunk_limit: # use this to train on one chunk only for debugging (should be 10,000 images)
# 				break

# 			train_data = load_from_disk(chunk_path)
# 			train_data.set_format("torch", columns=["image", "report"])
# 			train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# 			for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Chunk {chunk_id}")):

# 				if train_on_one_batch and i == 1:
# 					break

# 				x = batch["image"].to(device)
# 				noise = torch.randn_like(x)
# 				reports = batch["report"]

# 				optimizer.zero_grad()
				
# 				with torch.amp.autocast(device_type='cuda'):
# 					pred, target = model(x, noise, reports)
# 					loss = combined_loss(pred, target)


# 				scaler.scale(loss).backward()
# 				scaler.step(optimizer)
# 				scaler.update()

# 				train_loss += loss.item()
# 				total_batches += 1

# 			chunk_id += 1

# 		avg_train_loss = train_loss / total_batches if total_batches > 0 else 0

# 		model.eval()
# 		val_loss = 0
# 		psnr_score = 0
# 		ssim_score = 0
# 		blob_loss = 0

# 		with torch.no_grad():
# 			for i, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")):

# 				if train_on_one_batch and i == 1:
# 					break

# 				x = batch["image"].to(device)
# 				noise = torch.randn_like(x)
# 				reports = batch["report"]

# 				pred, target = model(x, noise, reports)
# 				loss = combined_loss(pred, target)
# 				val_loss += loss.item()
# 				blob_loss_batch = local_variance_penalty(pred)
# 				blob_loss += blob_loss_batch.item()

# 				recon = (x - noise + pred).clamp(0, 1)
# 				for j in range(recon.size(0)):
# 					psnr_score += psnr(x[j].cpu().numpy(), recon[j].cpu().numpy(), data_range=1)
# 					ssim_score += ssim(
# 						x[j].unsqueeze(0),       # Add batch dimension → [1, 1, H, W]
# 						recon[j].unsqueeze(0),   # Same here
# 						data_range=1.0
# 					).item()

# 		avg_val_loss = val_loss / len(val_loader)
# 		avg_psnr = psnr_score / len(val_data) # this is the number of images
# 		avg_ssim = ssim_score / len(val_data)
# 		avg_blob_loss = blob_loss / len(val_loader)

# 		context = model.text_encoder(reports).to(device)
# 		_, images_by_step = model.sample(reports[:3], x[:3].shape, device, raw=True)

# 		# Find the image from timestep 0 (final sample)
# 		final_images = [img for (t, img) in images_by_step if t == 0]

# 		# final_images is a list of [B, 1, 256, 256] tensors — stack and compute std
# 		gen_stack = torch.cat(final_images, dim=0)

# 		log_data = {
# 			"model/epoch": epoch + 1,
# 			"train/avg_loss": avg_train_loss,
# 			"val/avg_loss": avg_val_loss,
# 			"val/psnr": avg_psnr,
# 			"val/ssim": avg_ssim,
# 			"val/blob_loss": avg_blob_loss,
# 			"val/context_mean": context.mean().item(),
# 			"val/context_std": context.std().item(),
# 		}

# 		if epoch % 5 == 0:
# 			with torch.no_grad():
# 				report_sample = reports[:3]
# 				shape_sample = x[:3].shape

# 				# Sampling with intermediates
# 				_, images_by_step = model.sample(
# 					report_texts=report_sample,
# 					latent_shape=shape_sample,
# 					device=device,
# 					raw=True,
# 					return_intermediates=[499, 199, 0]
# 				)

# 				intermediate_dict = {t: img for t, img in images_by_step}

# 				intermediate_images = []

# 				for j in range(3):  # sample index
# 					for t in [499, 199, 0]:  # timestep
# 						img = intermediate_dict[t][j].cpu().clamp(0, 1)
# 						caption = f"sample {j} | timestep {t} | report: {reports[j]}"
# 						intermediate_images.append(wandb.Image(img, caption=caption))

# 				log_data["val/intermediate_strip"] = intermediate_images


# 				# Final timestep images
# 				final_images = intermediate_dict[0]  # [3, 1, H, W]
# 				gen_stack = final_images

# 				log_data["val/generated_samples"] = [
# 					wandb.Image(img.cpu().clamp(0, 1), caption=str(rep))
# 					for img, rep in zip(final_images, reports[:3])
# 					]

# 				# Log generated sample std
# 				log_data["val/generated_std"] = gen_stack.std().item()

# 				# Align reconstructions with same sample inputs
# 				recon_images = []
# 				sample_inputs = x[:3]
# 				for j in range(3):
# 					real = sample_inputs[j].cpu()
# 					fake = final_images[j].cpu()
# 					pair = torch.cat([real, fake], dim=2)
# 					recon_images.append(wandb.Image(pair.clamp(0, 1), caption=f"val sample {j}"))
# 				log_data["val/reconstructions"] = recon_images

# 				# Guidance scale sweep
# 				guidance_scales = [5, 7.5, 10.0, 12.5, 15.0, 17.5]
# 				guidance_images = []

# 				for scale in guidance_scales:
# 					model.guidance_scale = scale
# 					_, g_steps = model.sample(
# 						report_texts=reports[:1],
# 						latent_shape=x[:1].shape,
# 						device=device,
# 						raw=True,
# 						return_intermediates=[0]
# 					)
# 					g_img = g_steps[0][1][0]
# 					guidance_images.append(
# 						wandb.Image(g_img.cpu().clamp(0, 1), caption=f"guidance_scale={scale}")
# 					)

# 				log_data["val/guidance_sweep"] = guidance_images

# 		wandb.log(log_data)

# 		scheduler.step(avg_val_loss)

# 		if avg_val_loss < best_val_loss and (epoch + 1) >= min_save_epoch:
# 			best_val_loss = avg_val_loss
# 			patience_counter = 0
# 			torch.save({
# 				"model_state": model.state_dict(),
# 				"optimizer_state": optimizer.state_dict(),
# 				"epoch": epoch + 1,
# 				"best_val_loss": best_val_loss
# 			}, checkpoint_path)
# 			print(f"[info] Model improved. Saved checkpoint to {checkpoint_path}")
# 		elif (epoch + 1) >= min_save_epoch:
# 			patience_counter += 1
# 			print(f"[info] No improvement. Patience: {patience_counter}/{early_stop_patience}")
# 			if patience_counter >= early_stop_patience:
# 				print("[info] Early stopping triggered.")
# 				break

# if __name__ == "__main__":
# 	main()