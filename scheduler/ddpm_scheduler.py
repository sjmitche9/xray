# cheduler/ddpm_scheduler.py
import torch

class ddpm_scheduler:

    def __init__(self, num_timesteps=500, beta_start=1e-4, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.beta = torch.linspace(beta_start, beta_end, num_timesteps).to(torch.float32)
        self.alpha = (1.0 - self.beta).to(torch.float32)
        self.alpha_hat = torch.cumprod(self.alpha, dim=0).to(torch.float32)

    def add_noise(self, x_start, noise, t):
        if t.device != self.alpha_hat.device:
            t = t.to(self.alpha_hat.device)

        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t]).to(x_start.device)[:, None, None, None]
        sqrt_one_minus = torch.sqrt(1.0 - self.alpha_hat[t]).to(x_start.device)[:, None, None, None]
        return sqrt_alpha_hat * x_start + sqrt_one_minus * noise


    def get_alpha_hat(self, t):
        return self.alpha_hat[t]
    
    def denoise(self, noisy, pred_noise, t):
        """
        Estimate the clean image x_0 from a noisy image and predicted noise.
        """
        device = noisy.device
        alpha = self.alpha.to(device)
        alpha_hat = self.alpha_hat.to(device)

        alpha_t = alpha[t].view(-1, 1, 1, 1)
        alpha_hat_t = alpha_hat[t].view(-1, 1, 1, 1)

        # DDPM reverse step: x_0 = (noisy - (1 - alpha_t) / sqrt(1 - alpha_hat_t) * pred_noise) / sqrt(alpha_t)
        x0_est = (noisy - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * pred_noise) / torch.sqrt(alpha_t)
        return x0_est
