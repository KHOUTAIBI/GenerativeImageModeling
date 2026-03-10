import os
import yaml
import tqdm
import argparse
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from guided_diffusion.unet import create_model
from PIL import Image
from utils import *
from noise_scheduler import NoiseScheduler
from torchvision.utils import make_grid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LinearOperator:
    def __init__(self, image_shape, measurement_dim, mask_ratio=0.5, seed=0, device="cpu"):
        self.image_shape = image_shape
        self.n = int(np.prod(image_shape))
        self.m = measurement_dim
        self.device = device
        _, C, H, W = image_shape   
        n = C * H * W

        hcrop, wcrop = H // 2, W // 2
        corner_top, corner_left = H // 4, int(0.45 * W)
        mask = torch.ones(image_shape, device=device)
        mask[:, :, corner_top:corner_top + hcrop, corner_left:corner_left + wcrop] = 0
        # g = torch.Generator(device="cpu")
        # g.manual_seed(seed)

        # mask = torch.rand(n, generator=g)
        # mask = (mask > mask_ratio).float()
        # mask = mask.view(1, C, H, W)

        self.mask = mask.to(device)

    def flatten(self, x):
        return x.view(x.shape[0], -1)

    def unflatten(self, x):
        return x.view(x.shape[0], *self.image_shape)

    def H(self, x):
        y = x * self.mask
        return y

    def H_pinv(self, y):
        return y * self.mask

    @torch.no_grad()
    def observe(self, x0):
        return self.H(x0)


def predict_x0_from_eps(x_t, eps_theta, alpha_bar_t):
    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=1e-12))
    x0_hat = (x_t - sqrt_one_minus_alpha_bar_t * eps_theta) / torch.clamp(sqrt_alpha_bar_t, min=1e-12)
    return x0_hat


def compute_pseudoinverse_guidance(x_t, x0_hat, y, operator):
    mat = operator.H_pinv(y) - operator.H_pinv(operator.H(x0_hat))
    inner = (mat.detach() * x0_hat).sum()
    guidance = torch.autograd.grad(inner, x_t)[0]
    return guidance


@torch.no_grad()
def ddim_step_from_x0_eps(x0_hat, eps_theta, alpha_bar_t, alpha_bar_s, eta):
    if eta > 0.0:
        sigma_ts = eta * torch.sqrt(
            torch.clamp(
                ((1.0 - alpha_bar_s) / torch.clamp(1.0 - alpha_bar_t, min=1e-12)) *
                (1.0 - alpha_bar_t / torch.clamp(alpha_bar_s, min=1e-12)),
                min=0.0
            )
        )
        z = torch.randn_like(x0_hat)
    else:
        sigma_ts = torch.zeros_like(alpha_bar_t)
        z = torch.zeros_like(x0_hat)

    c2 = torch.sqrt(torch.clamp(1.0 - alpha_bar_s - sigma_ts**2, min=0.0))
    x_s = torch.sqrt(alpha_bar_s) * x0_hat + c2 * eps_theta + sigma_ts * z
    return x_s


def pseudoinverse_guided_sample(
    model,
    scheduler,
    model_config,
    diffusion_config,
    operator,
    y,
):
    model.eval()

    batch_size = y.shape[0]
    im_channels = model_config["im_channels"]
    im_size = model_config["im_size"]

    num_train_steps = diffusion_config["num_timesteps"]
    num_inference_steps = diffusion_config.get("num_inference_steps", 50)
    
    eta = diffusion_config.get("eta", 0.0)

    timesteps = np.linspace(0, num_train_steps - 1, num_inference_steps, dtype=int)[::-1]

    x = torch.randn_like(y, device=y.device)

    alpha_bar = scheduler.alpha_bar.to(device)

    for i, t in enumerate(tqdm(timesteps, desc="PiGDM sampling")):
        s = timesteps[i + 1] if i + 1 < len(timesteps) else 0

        t_batch = torch.full((batch_size,), int(t), device=device, dtype=torch.long)

        x = x.detach()
        x.requires_grad_(True)

        eps_theta = model(x, t_batch)[:, :3, :, :]

        alpha_bar_t = alpha_bar[t].view(1, 1, 1, 1)
        alpha_bar_s = alpha_bar[s].view(1, 1, 1, 1)

        x0_hat = predict_x0_from_eps(x, eps_theta, alpha_bar_t)
        x0_hat = torch.clamp(x0_hat, -1.0, 1.0)

        guidance = compute_pseudoinverse_guidance(x, x0_hat, y, operator)
        guidance = torch.nan_to_num(guidance, nan=0.0, posinf=0.0, neginf=0.0)

        x_ddim = ddim_step_from_x0_eps(x0_hat, eps_theta, alpha_bar_t, alpha_bar_s, eta)

        x = x_ddim + torch.sqrt(alpha_bar_t) * guidance

        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        x = x.detach()

    return x


def save_grid(x, path, nrow=8):
    ims = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    ims = torch.clamp(ims, -1.0, 1.0).cpu()
    ims = (ims + 1.0) / 2.0
    grid = make_grid(ims, nrow=nrow)
    img = torchvision.transforms.ToPILImage()(grid)
    img.save(path)
    img.close()


def run(args):
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    diffusion_config = config["diffusion_config"]
    model_config = config["model_config"]
    train_config = config["train_config"]
    image_index = train_config['image_index']

    config = {
        'image_size': 256,
        'num_channels': 128,
        'num_res_blocks': 1,
        'channel_mult': '',
        'learn_sigma': True,
        'class_cond': False,
        'use_checkpoint': False,
        'attention_resolutions': 16,
        'num_heads': 4,
        'num_head_channels': 64,
        'num_heads_upsample': -1,
        'use_scale_shift_norm': True,
        'dropout': 0.0,
        'resblock_updown': True,
        'use_fp16': False,
        'use_new_attention_order': False,
        'model_path': './ffhq_10m.pt'
    }

    model = create_model(**config)
    model = model.to(device)
    model.eval()

    scheduler = NoiseScheduler(
        num_timesteps=diffusion_config["num_timesteps"],
        beta_init=diffusion_config["beta_init"],
        beta_end=diffusion_config["beta_end"],
        device=device,
    )

    idx = image_index
    x0 = im2tensor(plt.imread('ffhq256-1k-validation/' + str(idx).zfill(5) + '.png')).to(device)
    imgshape = x0.shape

    h = imgshape[2]
    w = imgshape[3]
    hcrop, wcrop = h // 2, w // 2
    corner_top, corner_left = h // 4, int(0.45 * w)
    mask = torch.ones(imgshape, device=device)
    mask[:, :, corner_top:corner_top + hcrop, corner_left:corner_left + wcrop] = 0

    operator = LinearOperator(
        image_shape=imgshape,
        measurement_dim=0.0,
        mask_ratio=0.5,
        seed = 0,
        device = device
    )

    sigma_noise = 2 * 10 / 255
    noise = torch.normal(0.0, std = sigma_noise, size = x0.size(), device=device)
    y = operator.H(x0.clone()) + noise

    x_init = torch.randn_like(y, device=device, requires_grad=True)

    save_grid(x_init, args.pinv_init_path, nrow=train_config["num_grid_rows"])

    x_rec = pseudoinverse_guided_sample(
        model=model,
        scheduler=scheduler,
        model_config=model_config,
        diffusion_config=diffusion_config,
        operator=operator,
        y=y,
    )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    save_grid(x_rec, args.output_path, nrow=train_config["num_grid_rows"])
    print(f"Saved reconstruction grid to: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_path", type=str, default="./config.yaml")
    parser.add_argument("--measurement_dim", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_path", type=str, default="./samples/pigdm_randomH.png")
    parser.add_argument("--pinv_init_path", type=str, default="./samples/pinv_init.png")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run(args)