import torch
import os
import numpy as np
from tqdm import tqdm

from noise_scheduler import NoiseScheduler

from utils import (psnr, 
                   save_grid)

device = "cuda" if torch.cuda.is_available() else "cpu"

def predict_x0_from_eps(x_t, eps_theta, alpha_t) -> torch.Tensor:
    """
        Computing x0 from eps, as seen in the DDIM paper
    """
    sqrt_alpha_t = torch.sqrt(torch.clamp(alpha_t, min=1e-12))
    sqrt_one_minus_alpha_t = torch.sqrt(torch.clamp(1.0 - alpha_t, min=0.0))
    x0_hat = (x_t - sqrt_one_minus_alpha_t * eps_theta) / sqrt_alpha_t
    return x0_hat

def compute_pseudoinverse_guidance(x_t, hatx_t, y, operator , sigma_y, r_t, mode) -> torch.Tensor:
    """
        Computing the guidance term in the case of linear and non linear operations
        WARNING:
            We can not compute the non linear operator in the case of noiys sigma_y !
            The next part would be to do so
    """
    if (sigma_y == 0.0) or (operator.type == "nonlinear"):

        v = operator.H_pinv(y) - operator.H_pinv(operator.H(hatx_t))
        inner = (v.detach() * hatx_t).sum()
        guidance = torch.autograd.grad(inner, x_t)[0]
        return guidance
    
    else:

        return operator.guidance(x_t, hatx_t, y, operator , sigma_y, r_t)
    
@torch.no_grad()
def ddim_step_from_x0_eps(x0_hat, eps_theta, alpha_t, alpha_s, eta) -> torch.Tensor:
    """
        DDIM denoising step, as seen in the paper
    """
    if eta > 0.0:
        c1 = eta * torch.sqrt(
            torch.clamp(
                (1.0 - alpha_t / alpha_s) * ((1.0 - alpha_s) / (1.0 - alpha_t)),
                min=0.0
            )
        )
        eps = torch.randn_like(x0_hat)
    else:
        c1 = torch.zeros_like(alpha_t)
        eps = torch.zeros_like(x0_hat)

    c2 = torch.sqrt(torch.clamp(1.0 - alpha_s - c1 ** 2, min=0.0))
    x_s = torch.sqrt(alpha_s) * x0_hat + c1 * eps + c2 * eps_theta
    return x_s


from explainability import (
    save_heatmap,
    tensor_norm,
    cosine_similarity_map,
    compute_input_saliency,
    plot_scalar_logs,
)

def pseudoinverse_guided_sample_ddim(
    model,
    scheduler: NoiseScheduler,
    diffusion_config,
    operator,
    x0,
    y,
):
    model.eval()

    batch_size = y.shape[0]

    save = diffusion_config['save']
    mode = diffusion_config['mode']
    num_train_steps = diffusion_config["num_timesteps"]
    num_inference_steps = diffusion_config.get("num_inference_steps", 100)
    eta = diffusion_config.get("eta", 0.0)
    sigma_y = diffusion_config.get("sigma_y", 0.01)
    guidance_scale = diffusion_config.get("guidance_scale", 1.0)

    timesteps = np.linspace(0, num_train_steps - 1, num_inference_steps, dtype=int)[::-1]
    x = torch.randn_like(x0, device=y.device)

    psnr_list = []
    alpha = scheduler.alpha_bar.to(y.device)

    for i, t in enumerate(tqdm(timesteps, desc="PiGDM-DDIM sampling")):
        s = timesteps[i + 1] if i + 1 < len(timesteps) else 0
        t_batch = torch.full((batch_size,), int(t), device=device, dtype=torch.long)

        x = x.detach().requires_grad_(True)

        eps_theta = model(x, t_batch)[:, :3, :, :]

        alpha_t = alpha[t].view(1, 1, 1, 1)
        alpha_s = alpha[s].view(1, 1, 1, 1)

        hatx_t = predict_x0_from_eps(x, eps_theta, alpha_t)
        hatx_t = torch.clamp(hatx_t, -1.0, 1.0)

        r_t = torch.sqrt(torch.clamp(1.0 - alpha_t, min=1e-12))

        guidance = compute_pseudoinverse_guidance(
            x_t=x,
            hatx_t=hatx_t,
            y=y,
            operator=operator,
            sigma_y=sigma_y,
            r_t=r_t,
            mode=mode,
        )

        guidance = torch.nan_to_num(guidance, nan=0.0, posinf=0.0, neginf=0.0)

        x_ddim = ddim_step_from_x0_eps(hatx_t, eps_theta, alpha_t, alpha_s, eta)
        x_next = x_ddim + guidance_scale * torch.sqrt(alpha_t) * guidance

        residual = operator.H(hatx_t) - y
        psnr_x = psnr(x0, x_next)

        x = torch.nan_to_num(x_next, nan=0.0, posinf=1.0, neginf=-1.0).detach()
        psnr_list.append(psnr_x)

        if i % 25 == 0 and save:
            save_grid(hatx_t, path=f"./samples/pigdm_ddim_output_{i}.png")

    return x, psnr_list


def pseudoinverse_guided_sample_ddpm(
    model,
    diffusion_config,
    operator,
    x0,
    y
) -> tuple[torch.Tensor, list]:
    """
    The pseudo inverse guidance algorithm, In this case we use DDPM instead of DDIM
    """
    model.eval()

    save = diffusion_config['save']
    mode = diffusion_config['mode']
    num_train_steps = diffusion_config["num_timesteps"]
    beta_start = diffusion_config["beta_init"]
    beta_end = diffusion_config["beta_end"]
    sigma_y = diffusion_config.get("sigma_y", 0.01)
    guidance_scale = diffusion_config.get("guidance_scale", 1.0)

    beta = torch.linspace(beta_start, beta_end, num_train_steps, device=device)
    alpha = 1.0 - beta
    alphabar = torch.cumprod(alpha, dim=0)
    betabar = 1.0 - alphabar

    psnr_list = list()

    batch_size = y.shape[0]
    reversed_time_steps = np.arange(num_train_steps)[::-1]
    x = torch.randn_like(x0, device=device)

    for i, t in tqdm(enumerate(reversed_time_steps), total=len(reversed_time_steps), desc="PiGDM-DDPM sampling"):

        x = x.detach().requires_grad_(True)
        t_batch = torch.full((batch_size,), int(t), device=device, dtype=torch.long)

        eps = model(x, t_batch)[:, :3, :, :]
        xhat = (x - torch.sqrt(torch.clamp(betabar[t], min=1e-12)) * eps) / torch.sqrt(torch.clamp(alphabar[t], min=1e-12))
        xhat = torch.clamp(xhat, -1.0, 1.0)

        mu = (x - (beta[t] / torch.sqrt(torch.clamp(betabar[t], min=1e-12))) * eps) / torch.sqrt(torch.clamp(alpha[t], min=1e-12))

        r_t = torch.sqrt(torch.clamp(1.0 - alphabar[t], min=1e-12))

        guidance = compute_pseudoinverse_guidance(
            x_t=x,
            hatx_t=xhat,
            y=y,
            operator=operator,
            sigma_y=sigma_y,
            r_t=r_t,
            mode = mode
        )
        guidance = torch.nan_to_num(guidance, nan=0.0, posinf=0.0, neginf=0.0)

        if t > 0:
            z = torch.randn_like(x)
        else:
            z = torch.zeros_like(x)

        x = mu + torch.sqrt(torch.clamp(beta[t], min=0.0)) * z + guidance_scale * torch.sqrt(torch.clamp(alphabar[t], min=1e-12)) * guidance
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0).detach()

        psnr_x = psnr(x0, x)
        psnr_list.append(psnr_x)

        if i % 100 == 0 and save:
            save_grid(x, path=f"./samples/pigdm_ddpm_output_{i}.png")

    return x, psnr_list

def dps_sample_diffsion(
    model,
    diffusion_config,
    operator,
    x0,
    y  
) -> tuple[torch.Tensor, list]:
    
    """
        The DPS algorithm as seen in the practical session, which uses DDPM and calculates the gradient of the error
    """
    
    model.eval()

    save = diffusion_config['save']
    num_train_steps = diffusion_config["num_timesteps"]
    beta_start = diffusion_config["beta_init"]
    beta_end = diffusion_config["beta_end"]
    guidance_scale = diffusion_config["guidance_scale"]

    beta = torch.linspace(beta_start, beta_end, num_train_steps, device=device)
    alpha = 1.0 - beta
    alphabar = torch.cumprod(alpha, dim=0)
    betabar = 1.0 - alphabar

    psnr_list = list()

    batch_size = y.shape[0]
    reversed_time_steps = np.arange(num_train_steps)[::-1]
    x = torch.randn_like(x0, device=device)

    for i, t in tqdm(enumerate(reversed_time_steps), total=len(reversed_time_steps), desc="PiGDM-DPS sampling"):

        x = x.detach().requires_grad_(True)
        t_batch = torch.full((batch_size,), int(t), device=device, dtype=torch.long)

        eps = model(x, t_batch)[:, :3, :, :]
        xhat = (x - torch.sqrt(torch.clamp(betabar[t], min=1e-12)) * eps) / torch.sqrt(torch.clamp(alphabar[t], min=1e-12))
        # xhat = torch.clamp(xhat, -1.0, 1.0)

        error = (operator.H(xhat) - y).pow(2).sum()
        grad = torch.autograd.grad(error, x)[0]

        mu = (x - (beta[t] / torch.sqrt(torch.clamp(betabar[t], min=1e-12))) * eps) / torch.sqrt(torch.clamp(alpha[t], min=1e-12))

        if t > 0:
            z = torch.randn_like(x)
        else:
            z = torch.zeros_like(x)

        x = mu + torch.sqrt(torch.clamp(beta[t], min=0.0)) * z - guidance_scale * grad / torch.sqrt(error)
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0).detach()

        psnr_x = psnr(x0, x)
        psnr_list.append(psnr_x)

        if (i % 100 == 0 or t == 0) and save:
            save_grid(x, path=f"./samples/dps_ddpm_output_t={t}.png")

    return x, psnr_list


def simple_ddpm(
    model,
    diffusion_config,
    y,
    x0,
    dir="simple_ddpm"
):
    """
        Sampling from the DDPM without guidance
    """
    model.eval()

    save = diffusion_config['save']
    num_train_steps = diffusion_config["num_timesteps"]
    beta_start = diffusion_config["beta_init"]
    beta_end = diffusion_config["beta_end"]

    beta = torch.linspace(beta_start, beta_end, num_train_steps, device=device)
    alpha = 1.0 - beta
    alphabar = torch.cumprod(alpha, dim=0)
    betabar = 1.0 - alphabar

    reversed_time_steps = np.arange(num_train_steps)[::-1]
    x = y.clone().to(device)

    psnrx = []
    psnrxhat = []
    for i, t in tqdm(enumerate(reversed_time_steps), total=len(reversed_time_steps), desc="DDPM sampling"):
        with torch.no_grad():
            model_output = model(x, torch.tensor(t, device=device).unsqueeze(0))
        eps = model_output[:,:3,:,:]
        xhat = (x - torch.sqrt(betabar[t]) * eps) / torch.sqrt(alphabar[t])

        z = torch.randn_like(x)
        x = torch.sqrt(alpha[t]) * x + torch.sqrt(beta[t]) * z

        psnr_x = psnr(x, x0)
        psnrhat = psnr(xhat, x0)

        psnrx.append(psnr_x)
        psnrxhat.append(psnrhat)

        if (t+1)%100==0 or t==0 and save:
            print('Iteration :', t+1)
            save_grid(torch.cat((x, xhat, x0), dim=3), path=f"./samples/{dir}/ddpm_no_guidance_output_t={t}.png")
        
    plot_scalar_logs({"psnrx": psnrx, "psnrxhat": psnrxhat}, path=f"./samples/{dir}/ddpm_no_guidance_psnr.png")

    return x