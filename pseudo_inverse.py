import os
import yaml
import tqdm
import argparse
import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

from unet import Unet
from noise_scheduler import NoiseScheduler
from torchvision.utils import make_grid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LinearOperator:
    def __init__(self, image_shape, measurement_dim, mask_ratio=0.5, seed=0, device="cpu"):
        self.image_shape = image_shape
        self.n = int(np.prod(image_shape))
        self.m = measurement_dim
        self.device = device

        C, H, W = image_shape
        n = C * H * W

        g = torch.Generator(device="cpu")
        g.manual_seed(seed)

        mask = torch.rand(n, generator=g)
        mask = (mask > mask_ratio).float()   
        mask = mask.view(1, C, H, W)

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
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
    x0_hat = (x_t - sqrt_one_minus_alpha_bar_t * eps_theta) / sqrt_alpha_bar_t
    return x0_hat


def compute_pseudoinverse_guidance(x_t, x0_hat, y, operator):
    mat = operator.H_pinv(y) - operator.H_pinv(operator.H(x0_hat))
    inner = (mat.detach() * x0_hat).sum()
    guidance = torch.autograd.grad(inner, x_t, retain_graph=False, create_graph=False)[0]
    return guidance


@torch.no_grad()
def ddim_step_from_x0_eps(x0_hat, eps_theta, alpha_bar_t, alpha_bar_s, eta):
    if eta > 0.0:
        sigma_ts = eta * torch.sqrt(
            ((1.0 - alpha_bar_s) / (1.0 - alpha_bar_t)) *
            (1.0 - alpha_bar_t / alpha_bar_s)
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
    train_config,
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

    x = torch.randn(
        batch_size,
        im_channels,
        im_size,
        im_size,
        device=device
    )

    alpha_bar = scheduler.alpha_bar.to(device)

    for i, t in enumerate(tqdm.tqdm(timesteps, desc="PiGDM sampling")):
        s = timesteps[i + 1] if i + 1 < len(timesteps) else 0

        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

        x = x.detach()
        x.requires_grad_(True)

        eps_theta = model(x, t_batch)

        alpha_bar_t = alpha_bar[t].view(1, 1, 1, 1)
        alpha_bar_s = alpha_bar[s].view(1, 1, 1, 1)

        x0_hat = predict_x0_from_eps(x, eps_theta, alpha_bar_t)
        guidance = compute_pseudoinverse_guidance(x, x0_hat, y, operator)

        x_ddim = ddim_step_from_x0_eps(x0_hat, eps_theta, alpha_bar_t, alpha_bar_s, eta)

        # Add sqrt(alpha_t) * g
        x = x_ddim + torch.sqrt(alpha_bar_t) * guidance
        x = x.detach()

    return x


def save_grid(x, path, nrow=8):
    ims = torch.clamp(x, -1.0, 1.0).cpu()
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

    model = Unet(model_config).to(device)
    model.load_state_dict(torch.load(model_config["save_path"], map_location=device))
    model.eval()

    scheduler = NoiseScheduler(
        num_timesteps=diffusion_config["num_timesteps"],
        beta_init=diffusion_config["beta_init"],
        beta_end=diffusion_config["beta_end"],
        device=device,
    )

    mnist_trainset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: 2 * x - 1)
                    ])
    )

    mnist_loader = DataLoader(
        mnist_trainset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=4
    )

    image_shape = (
        model_config["im_channels"],
        model_config["im_size"],
        model_config["im_size"],
    )

    n = int(np.prod(image_shape))
    m = args.measurement_dim if args.measurement_dim is not None else n // 4

    operator = LinearOperator(
        image_shape=image_shape,
        measurement_dim=m,
        seed=args.seed,
        mask_ratio=0.5,
        device=device,
    )

    batch_size = args.batch_size if args.batch_size is not None else train_config["batch_size"]
    
    # Here y is a random measurement vector.
    indx = np.random.randint(low = 0, high = mnist_trainset.__len__())
    x_true, _ = mnist_trainset.__getitem__(index = indx)
    x_true = x_true.to(device)
    y = operator.observe(x_true)
    noise = torch.normal(mean = 0.0, std = 0.05, size=y.size(), device=device)
    y = y + noise

    with torch.no_grad():
        x_init = operator.H_pinv(y)
    save_grid(x_init, args.pinv_init_path, nrow=train_config["num_grid_rows"])

    x_rec = pseudoinverse_guided_sample(
        model=model,
        scheduler=scheduler,
        model_config=model_config,
        diffusion_config=diffusion_config,
        train_config=train_config,
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

    run(args)