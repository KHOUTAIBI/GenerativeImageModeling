import json
import yaml
import tqdm
import argparse
import torch
import torchvision
import torch.nn as nn
import numpy as np
from unet import Unet
from noise_scheduler import NoiseScheduler
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim import Adam

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_alpha_cumprod(scheduler):
    if hasattr(scheduler, "alpha_cumprod"):
        return scheduler.alpha_cumprod
    if hasattr(scheduler, "alphas_cumprod"):
        return scheduler.alphas_cumprod
    if hasattr(scheduler, "alpha_cumulative"):
        return scheduler.alpha_cumulative
    if hasattr(scheduler, "alpha_bar"):
        return scheduler.alpha_bar
    raise AttributeError("Could not find alpha cumulative product in scheduler.")


# -------------------------------
# Train
# -------------------------------
def train(args):
    """
    DDIM uses the same noise-prediction training objective as DDPM.
    """
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except Exception as e:
            print(f'Error loading config: {e}')
            return None

    diffusion_config = config['diffusion_config']
    dataset_config = config['dataset_config']
    model_config = config['model_config']
    train_config = config['train_config']
    load = train_config['load']


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


    scheduler = NoiseScheduler(
        num_timesteps=diffusion_config['num_timesteps'],
        beta_init=diffusion_config['beta_init'],
        beta_end=diffusion_config['beta_end']
    )

    model = Unet(model_config).to(device)
    if load :
        model.load_state_dict(torch.load(train_config['ckpt_name']))
    model.train()

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=train_config['lr'])
    num_epochs = train_config['num_epochs']

    for epoch in range(num_epochs):
        losses = []

        for img, _ in tqdm.tqdm(mnist_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            img = img.float().to(device)
            optimizer.zero_grad()

            noise = torch.randn_like(img)
            t = torch.randint(
                0,
                diffusion_config['num_timesteps'],
                size=(img.shape[0],),
                device=device
            )

            image_noisy = scheduler.add_noise(img, noise, t)
            noise_prediction = model(image_noisy, t)

            loss = criterion(noise_prediction, noise)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()

        print(f"Finished epoch {epoch+1}/{num_epochs} | Loss: {np.mean(losses):.4f}")
        torch.save(model.state_dict(), f'./saves/ddim_mnist_chkpt_final.pth')

    print("Finished training!")


# -------------------------------
# DDIM step
# -------------------------------
def ddim_step(xt, noise_pred, t, t_prev, scheduler, eta=0.0):
    alpha_cumprod = get_alpha_cumprod(scheduler)

    alpha_t = alpha_cumprod[t]
    alpha_prev = alpha_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=xt.device)

    if not torch.is_tensor(alpha_t):
        alpha_t = torch.tensor(alpha_t, device=xt.device)
    if not torch.is_tensor(alpha_prev):
        alpha_prev = torch.tensor(alpha_prev, device=xt.device)

    alpha_t = alpha_t.float()
    alpha_prev = alpha_prev.float()

    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)

    x0_pred = (xt - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
    x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

    sigma_t = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))
    dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_prev - sigma_t ** 2, min=0.0)) * noise_pred

    if eta > 0 and t_prev >= 0:
        z = torch.randn_like(xt)
    else:
        z = torch.zeros_like(xt)

    x_prev = torch.sqrt(alpha_prev) * x0_pred + dir_xt + sigma_t * z
    return x_prev, x0_pred


# -------------------------------
# DDIM Sample
# -------------------------------
def sample(model, scheduler, train_config, model_config, diffusion_config):
    """
    DDIM sampling.
    """
    batch_size = train_config['batch_size']
    num_timesteps = diffusion_config['num_timesteps']
    num_inference_steps = diffusion_config.get('num_inference_steps', num_timesteps)
    eta = diffusion_config.get('eta', 0.0)

    xt = torch.randn(
        size=(
            batch_size,
            model_config['im_channels'],
            model_config['im_size'],
            model_config['im_size']
        ),
        device=device
    )

    timesteps = np.linspace(0, num_timesteps - 1, num_inference_steps, dtype=int)[::-1]

    for idx, t in enumerate(tqdm.tqdm(timesteps, desc="DDIM Sampling")):
        t_prev = timesteps[idx + 1] if idx + 1 < len(timesteps) else -1

        t_batch = torch.full((xt.shape[0],), t, device=device, dtype=torch.long)
        noise_prediction = model(xt, t_batch)

        xt, x0_pred = ddim_step(
            xt=xt,
            noise_pred=noise_prediction,
            t=t,
            t_prev=t_prev,
            scheduler=scheduler,
            eta=eta
        )

        ims = torch.clamp(xt, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2

        grid = make_grid(ims, nrow=train_config['num_grid_rows'])
        img = torchvision.transforms.ToPILImage()(grid)

        if idx % 100 == 0 or idx == len(timesteps) - 1:
            img.save(f'./samples/sample_step_{idx}.png')
            img.close()


# -------------------------------
# Inference
# -------------------------------
def infer(args):
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
            return

    diffusion_config = config['diffusion_config']
    model_config = config['model_config']
    train_config = config['train_config']

    model = Unet(model_config).to(device)
    model.load_state_dict(torch.load(model_config['save_path'], map_location=device))
    model.eval()

    scheduler = NoiseScheduler(
        num_timesteps=diffusion_config['num_timesteps'],
        beta_init=diffusion_config['beta_init'],
        beta_end=diffusion_config['beta_end']
    )

    with torch.no_grad():
        sample(model, scheduler, train_config, model_config, diffusion_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for DDIM training / inference')
    parser.add_argument('--config', dest='config_path', default='./config.yaml', type=str)
    parser.add_argument('--mode', default='infer', type=str, choices=['train', 'infer'])
    args = parser.parse_args()

    if args.mode == 'train':
        print(f"Training on: {device}")
        train(args)
    else:
        infer(args)