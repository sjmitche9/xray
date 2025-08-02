# models/diffusion.py
import torch
import torch.nn as nn
from models.text_encoder import text_encoder
from models.unet import conditional_unet
from scheduler.ddpm_scheduler import ddpm_scheduler


class DiffusionModel(nn.Module):
    def __init__(self, config, context_dropout, guidance_scale):
        super().__init__()
        
        model_cfg = config["MODEL"]
        base_channels = model_cfg.get("BASE_CHANNELS", 96)
        reconstruction_timesteps = config["SCHEDULER"].get("TIMESTEPS", 250)
        latent_dim = model_cfg["LATENT_DIM"]

        self.unet = conditional_unet(
            in_channels=latent_dim,
            base_channels=base_channels,
            context_dim=768
        )
        self.text_encoder = text_encoder(model_cfg["TOKENIZER_NAME"])
        self.scheduler = ddpm_scheduler(num_timesteps=reconstruction_timesteps)
        self.guidance_scale = guidance_scale
        self.context_dropout = context_dropout


    def forward(self, z_target, noise, reports):
        context = self.text_encoder(reports).to(z_target.device)

        if self.training and torch.rand(1).item() < self.context_dropout:
            context = torch.zeros_like(context)

        t = torch.randint(0, self.scheduler.num_timesteps, (z_target.size(0),), device=z_target.device).long()
        noisy_latents = self.scheduler.add_noise(z_target, noise, t)

        if not self.training and self.guidance_scale > 0:
            pred_cond = self.unet(noisy_latents, t, context)
            pred_uncond = self.unet(noisy_latents, t, torch.zeros_like(context))
            pred_noise = pred_uncond + self.guidance_scale * (pred_cond - pred_uncond)
        else:
            pred_noise = self.unet(noisy_latents, t, context)

        return pred_noise, noise


    def sample(self, report_texts, latent_shape, device, step_interval=50, raw=False, return_intermediates=None):

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

                # 💡 correct denoising step
                z = self.scheduler.denoise(z, pred, t_tensor)

                if t_gen > 0:
                    beta_t = self.scheduler.beta[t_gen]
                    z += torch.sqrt(beta_t) * torch.randn_like(z)

                should_log = (
                    (return_intermediates and t_gen in return_intermediates)
                    or (return_intermediates is None and (t_gen % step_interval == 0 or t_gen == 0))
                )
                if should_log:
                    recon = z.cpu().clamp(0, 1) if raw else z.cpu().clamp(0, 1)
                    images_by_step.append((t_gen, recon))

            return z, images_by_step