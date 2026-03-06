import numpy as np
import torch
import torch.nn as nn


class NoiseScheduler(nn.Module):
    def __init__(self, num_timesteps, beta_init, beta_end, device='cuda', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.device = device
        self.num_timesteps = num_timesteps
        self.beta_init = beta_init
        self.beta_end = beta_end

        # linear beta schedule
        self.betas = torch.linspace(beta_init, beta_end, num_timesteps, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)

    def add_noise(self, original_image, noise, t):
        """
        Forward diffusion process:
            x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
        """
        device = original_image.device
        t = t.to(device)

        sqrt_alpha_bar_t = self.sqrt_alpha_bar.to(device)[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar.to(device)[t].view(-1, 1, 1, 1)

        return sqrt_alpha_bar_t * original_image + sqrt_one_minus_alpha_bar_t * noise

    def sample_prev_timestep(self, xt, noise_prediction, t, t_prev, eta=0.0):
        """
        DDIM reverse step:
            x_t -> x_{t_prev}

        eta = 0 gives deterministic DDIM
        eta > 0 adds stochasticity
        """
        device = xt.device

        if not torch.is_tensor(t):
            t = torch.tensor(t, device=device, dtype=torch.long)
        if not torch.is_tensor(t_prev):
            t_prev = torch.tensor(t_prev, device=device, dtype=torch.long)

        alpha_bar_t = self.alpha_bar.to(device)[t]
        if t_prev >= 0:
            alpha_bar_prev = self.alpha_bar.to(device)[t_prev]
        else:
            alpha_bar_prev = torch.tensor(1.0, device=device)

        if alpha_bar_t.dim() == 0:
            alpha_bar_t = alpha_bar_t.view(1)
        if alpha_bar_prev.dim() == 0:
            alpha_bar_prev = alpha_bar_prev.view(1)

        alpha_bar_t = alpha_bar_t.view(-1, 1, 1, 1)
        alpha_bar_prev = alpha_bar_prev.view(-1, 1, 1, 1)

        # predict x0 from x_t and predicted noise
        x0 = (xt - torch.sqrt(1.0 - alpha_bar_t) * noise_prediction) / torch.sqrt(alpha_bar_t)
        x0 = torch.clamp(x0, -1.0, 1.0)

        sigma_t = eta * torch.sqrt(
            ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)) *
            (1.0 - alpha_bar_t / alpha_bar_prev)
        ) # This one is the same as in DDPM, taking eta = 1.0 gives the DDPM denoising step

        direction = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma_t**2, min=0.0)) * noise_prediction

        if eta > 0.0 and t_prev >= 0:
            z = torch.randn_like(xt)
        else:
            z = torch.zeros_like(xt)

        x_prev = torch.sqrt(alpha_bar_prev) * x0 + direction + sigma_t * z

        return x_prev, x0