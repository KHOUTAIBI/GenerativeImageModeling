import torch
import os
import numpy as np
from tqdm import tqdm

from noise_scheduler import NoiseScheduler

from utils import (psnr, 
                   save_grid)

from explainability import (
    save_heatmap,
    tensor_norm,
    cosine_similarity_map,
    compute_input_saliency,
    plot_scalar_logs,
    compute_hatx_saliency
)

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
    """
    if sigma_y == 0.0 :

        v = operator.H_pinv(y) - operator.H_pinv(operator.H(hatx_t))
        inner = (v.detach() * hatx_t).sum()
        guidance = torch.autograd.grad(inner, x_t)[0]
        return guidance
    
    else:

        assert operator.type == "linear", "Can not pseudo guide with a non linear operator in the case of noisy y"
        
        residual = y - operator.H(hatx_t)
        lam = (sigma_y / r_t).pow(2)
        v = residual / (operator.mask + lam)
        u = operator.H(v)
        inner = (u.detach() * hatx_t).sum()
        guidance = torch.autograd.grad(inner, x_t)[0]
        return guidance
    
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

    explain_cfg = diffusion_config.get("explainability", {})
    explain_enabled = explain_cfg.get("enabled", False)
    save_every = explain_cfg.get("save_every", 20)
    explain_outdir = explain_cfg.get("outdir", "./samples/explain_ddim")

    timesteps = np.linspace(0, num_train_steps - 1, num_inference_steps, dtype=int)[::-1]
    x = torch.randn_like(x0, device=y.device)

    psnr_list = []
    alpha = scheduler.alpha_bar.to(y.device)

    logs = {
        "psnr": [],
        "guidance_norm": [],
        "eps_norm": [],
        "residual_norm": [],
        "xhat_norm": [],
        "cos_guidance_eps": [],
        "pinv_residual_norm": [],
    }

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

        logs["psnr"].append(psnr_x)
        logs["guidance_norm"].append(tensor_norm(guidance))
        logs["eps_norm"].append(tensor_norm(eps_theta))
        logs["residual_norm"].append(tensor_norm(residual))
        logs["xhat_norm"].append(tensor_norm(hatx_t))
        logs["cos_guidance_eps"].append(cosine_similarity_map(guidance, -eps_theta))

        if sigma_y == 0.0:
            pinv_residual = operator.H_pinv(y) - operator.H_pinv(operator.H(hatx_t))
            logs["pinv_residual_norm"].append(tensor_norm(pinv_residual))
        else:
            logs["pinv_residual_norm"].append(float("nan"))

        if explain_enabled and (i % save_every == 0):
            os.makedirs(explain_outdir, exist_ok=True)
            
            save_heatmap(hatx_t, f"{explain_outdir}/hatx_step_{i:04d}.png", title=f"hatx_t step {i}")
            save_heatmap(guidance, f"{explain_outdir}/guidance_step_{i:04d}.png", title=f"guidance step {i}")
            save_heatmap(residual if residual.shape[-2:] == y.shape[-2:] else residual,
                         f"{explain_outdir}/residual_step_{i:04d}.png",
                         title=f"measurement residual step {i}")

        
            saliency = compute_hatx_saliency(model, x.detach(), t_batch, alpha_t)
            save_heatmap(saliency, f"{explain_outdir}/saliency_hatx_step_{i:04d}.png",
                         title=f"|d ||hatx_t||^2 / dx_t| step {i}")

            if sigma_y == 0.0:
                save_heatmap(pinv_residual, f"{explain_outdir}/pinv_residual_step_{i:04d}.png",
                             title=f"H^dag(y)-H^dag(H(hatx_t)) step {i}")

        x = torch.nan_to_num(x_next, nan=0.0, posinf=1.0, neginf=-1.0).detach()
        psnr_list.append(psnr_x)

        if i % 25 == 0 and save:
            save_grid(x, path=f"./samples/pigdm_ddim_output_{i}.png")

    if explain_enabled:
        plot_scalar_logs(logs, explain_outdir)

    return x, psnr_list, logs


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

        if i % 100 == 0 and save:
            save_grid(x, path=f"./samples/dps_ddpm_output_{i}.png")

    return x, psnr_list
