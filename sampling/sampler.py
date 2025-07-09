# --- sampling/sampler.py ---
import torch


def guided_sample(unet, scheduler, context, null_context, latent_shape, guidance_scale, device):
    z = torch.randn(*latent_shape, device=device)

    for t_gen in reversed(range(scheduler.num_timesteps)):
        t_tensor = torch.full((latent_shape[0],), t_gen, device=device, dtype=torch.long)

        pred_cond = unet(z, t_tensor, context)
        pred_uncond = unet(z, t_tensor, null_context)
        pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

        alpha_t = scheduler.alpha[t_gen]
        alpha_hat_t = scheduler.alpha_hat[t_gen]
        beta_t = scheduler.beta[t_gen]

        z = (1 / torch.sqrt(alpha_t)) * (z - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * pred)
        if t_gen > 0:
            z += torch.sqrt(beta_t) * torch.randn_like(z)

    return z
