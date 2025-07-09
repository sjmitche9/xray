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

    def sample(self, report_texts, latent_shape, device):
        with torch.no_grad():
            ctx = self.text_encoder(report_texts).to(device)
            null_ctx = torch.zeros_like(ctx)
            z = torch.randn(latent_shape).to(device)

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

        return z