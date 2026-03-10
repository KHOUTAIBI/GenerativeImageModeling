import os
import yaml
import tqdm
import argparse
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from guided_diffusion.unet import create_model
from utils import *
from operators import *
from noise_scheduler import NoiseScheduler
from torchvision.utils import make_grid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_x0_from_eps(x_t, eps_theta, alpha_t):
    sqrt_alpha_t = torch.sqrt(torch.clamp(alpha_t, min=1e-12))
    sqrt_one_minus_alpha_t = torch.sqrt(torch.clamp(1.0 - alpha_t, min=0.0))
    x0_hat = (x_t - sqrt_one_minus_alpha_t * eps_theta) / sqrt_alpha_t
    return x0_hat


def compute_pseudoinverse_guidance(x_t, hatx_t, y, operator, sigma_y, r_t):
    residual = y - operator.H(hatx_t)
    lam = sigma_y ** 2 / (r_t ** 2 + 1e-12)
    v = residual / (operator.mask + lam)
    u = operator.H(v)
    inner = (u.detach() * hatx_t).sum()
    guidance = torch.autograd.grad(inner, x_t)[0]
    return guidance


@torch.no_grad()
def ddim_step_from_x0_eps(x0_hat, eps_theta, alpha_t, alpha_s, eta):
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
    num_train_steps = diffusion_config["num_timesteps"]
    num_inference_steps = diffusion_config.get("num_inference_steps", 100)
    eta = diffusion_config.get("eta", 0.0)
    sigma_y = diffusion_config.get("sigma_y", 0.01)
    guidance_scale = diffusion_config.get("guidance_scale", 1.0)

    timesteps = np.linspace(0, num_train_steps - 1, num_inference_steps, dtype=int)[::-1]
    x = torch.randn_like(y, device=y.device)

    psnr_list = list()

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
            r_t=r_t
        )
        guidance = torch.nan_to_num(guidance, nan=0.0, posinf=0.0, neginf=0.0)

        x_ddim = ddim_step_from_x0_eps(hatx_t, eps_theta, alpha_t, alpha_s, eta)
        x = x_ddim + guidance_scale * torch.sqrt(alpha_t) * guidance

        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0).detach()
        psnr_x = psnr(x0, x)
        psnr_list.append(psnr_x)

        if i % 25 == 0:
            save_grid(x, path=f"./samples/pigdm_ddim_output_{i}.png")

    return x, psnr_list


def pseudoinverse_guided_sample_ddpm(
    model,
    diffusion_config,
    operator,
    x0,
    y
):
    model.eval()

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
    x = torch.randn_like(y, device=device)

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
            r_t=r_t
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

        if i % 100 == 0:
            save_grid(x, path=f"./samples/pigdm_ddpm_output_{i}.png")

    return x, psnr_list

def dps_sample_diffsion(
    model,
    diffusion_config,
    operator,
    x0,
    y  
):
    
    model.eval()

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
    x = torch.randn_like(y, device=device)

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

        if i % 100 == 0:
            save_grid(x, path=f"./samples/dps_ddpm_output_{i}.png")

    return x, psnr_list


def save_grid(x, path, nrow=8):
    ims = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    ims = torch.clamp(ims, -1.0, 1.0).cpu()
    ims = (ims + 1.0) / 2.0
    grid = make_grid(ims, nrow=nrow)
    img = torchvision.transforms.ToPILImage()(grid)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    img.close()


def run(args):
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    diffusion_config = config["diffusion_config"]
    model_config = config["model_config"]
    train_config = config["train_config"]
    image_index = train_config["image_index"]

    model_kwargs = {
        "image_size": 256,
        "num_channels": 128,
        "num_res_blocks": 1,
        "channel_mult": "",
        "learn_sigma": True,
        "class_cond": False,
        "use_checkpoint": False,
        "attention_resolutions": 16,
        "num_heads": 4,
        "num_head_channels": 64,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "dropout": 0.0,
        "resblock_updown": True,
        "use_fp16": False,
        "use_new_attention_order": False,
        "model_path": "./ffhq_10m.pt",
    }

    model = create_model(**model_kwargs)
    model = model.to(device)
    model.eval()

    scheduler = NoiseScheduler(
        num_timesteps=diffusion_config["num_timesteps"],
        beta_init=diffusion_config["beta_init"],
        beta_end=diffusion_config["beta_end"],
        device=device,
    )

    idx = image_index
    x0 = im2tensor(plt.imread("ffhq256-1k-validation/" + str(idx).zfill(5) + ".png")).to(device)
    imgshape = x0.shape

    operator = LinearOperator(
        image_shape=imgshape,
        measurement_dim=0.0,
        device=device
    )

    sigma_noise = diffusion_config.get("sigma_y", 0.01)
    y = operator.H(x0.clone()) + sigma_noise * torch.randn_like(x0)

    save_grid(y, path="./samples/y_init.png")
    save_grid(operator.H(x0), path="./samples/y_clean.png")

    x_init = torch.randn_like(y)
    save_grid(x_init, args.pinv_init_path, nrow=train_config["num_grid_rows"])

    sampler = diffusion_config.get("sampler", "ddim").lower()

    if sampler == "ddpm":
        x_rec, psnr_list = pseudoinverse_guided_sample_ddpm(
            model=model,
            diffusion_config=diffusion_config,
            operator=operator,
            x0 = x0,
            y=y,
        )
    elif sampler == "ddim":
        x_rec, psnr_list = pseudoinverse_guided_sample_ddim(
            model=model,
            scheduler=scheduler,
            diffusion_config=diffusion_config,
            operator=operator,
            x0 = x0,
            y=y,
        )

    else :
        x_rec, psnr_list = dps_sample_diffsion(
            model=model,
            diffusion_config=diffusion_config,
            operator=operator,
            x0 = x0,
            y=y,
        )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    save_grid(x_rec, args.output_path, nrow=train_config["num_grid_rows"])
    plt.plot(np.arange(len(psnr_list)), psnr_list)
    plt.grid()
    plt.xlabel("Step")
    plt.ylabel("PSNR")
    plt.savefig(args.psnr_path + f"_{sampler}.png")
    print(f"Saved reconstruction grid to: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_path", type=str, default="./config.yaml")
    parser.add_argument("--measurement_dim", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_path", type=str, default="./samples/pigdm_randomH.png")
    parser.add_argument("--pinv_init_path", type=str, default="./samples/pinv_init.png")
    parser.add_argument("--psnr_path", type=str, default="./samples/psnr")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run(args)