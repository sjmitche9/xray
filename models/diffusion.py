# --- models/diffusion.py ---
import torch
import torch.nn as nn
from models.text_encoder import text_encoder
from models.unet import conditional_unet
from scheduler.ddpm_scheduler import ddpm_scheduler


class DiffusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_cfg = config["MODEL"]

        self.unet = conditional_unet(
            in_channels=model_cfg["LATENT_DIM"],
            context_dim=768  # Match BERT output
        )
        self.text_encoder = text_encoder(model_cfg["TOKENIZER_NAME"])
        self.scheduler = ddpm_scheduler()
        self.guidance_scale = config["TRAINING"]["GUIDANCE_SCALE"]
        self.context_dropout = config["TRAINING"]["CONTEXT_DROPOUT_PROB"]

    def forward(self, latents, noise, reports):
        context = self.text_encoder(reports).to(latents.device)
        if torch.rand(1).item() < self.context_dropout:
            context = torch.zeros_like(context)

        t = torch.randint(0, self.scheduler.num_timesteps, (latents.size(0),), device=latents.device).long()
        noisy_latents = self.scheduler.add_noise(latents, noise, t)

        pred_noise = self.unet(noisy_latents, t, context)
        return pred_noise, noise
    
    def sample(self, report_texts, latent_shape, device, vae=None, step_interval=50):
        assert vae is not None, "You must pass a VAE instance to decode intermediate steps."

        with torch.no_grad():
            ctx = self.text_encoder(report_texts).to(device)
            null_ctx = torch.zeros_like(ctx)
            z = torch.randn(latent_shape, device=device)

            images_by_step = []

            for t_gen in reversed(range(self.scheduler.num_timesteps)):
                t_tensor = torch.full((latent_shape[0],), t_gen, device=device, dtype=torch.long)

                pred_cond = self.unet(z, t_tensor, ctx)
                pred_uncond = self.unet(z, t_tensor, null_ctx)
                pred = pred_uncond + self.guidance_scale * (pred_cond - pred_uncond)

                alpha_t = self.scheduler.alpha[t_gen]
                alpha_hat_t = self.scheduler.alpha_hat[t_gen]
                beta_t = self.scheduler.beta[t_gen]

                z = (1 / torch.sqrt(alpha_t)) * (z - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * pred)
                if t_gen > 0:
                    z += torch.sqrt(beta_t) * torch.randn_like(z)

                # Save intermediate reconstructions
                if t_gen % step_interval == 0 or t_gen == 0:
                    # Optional: normalize z before decoding
                    z_norm = (z - z.mean()) / (z.std() + 1e-5)
                    z_clamped = z_norm.clamp(-5, 5)
                    recon = vae.decode(z_clamped).cpu().clamp(0, 1)
                    images_by_step.append((t_gen, recon))

            return z, images_by_step