import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import csv
import os
from time import time


device = "cuda:0" if torch.cuda.is_available() else "cpu"

def tensor2im(x):
    x = 0.5+0.5*x # [-1,1]->[0,1]
    return x.detach().cpu().permute(2,3,1,0).squeeze()

def im2tensor(x):
    x = torch.tensor(x,device=device)
    x = 2*x-1 # [0,1]->[-1,1]
    return x.permute(2,0,1).unsqueeze(0)

def rgb2gray(u):
    return 0.2989 * u[:,:,0] + 0.5870 * u[:,:,1] + 0.1140 * u[:,:,2]

def renorm(x):
    return (x - x.min()) / (x.max() - x.min())
 
def psnr(uref, ut, M=2):
    rmse = np.sqrt(np.mean((np.array(uref.cpu())-np.array(ut.cpu()))**2))
    return 20*np.log10(M/rmse)

def save_grid(x, path, nrow=8):
    ims = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    ims = torch.clamp(ims, -1.0, 1.0).cpu()
    ims = (ims + 1.0) / 2.0
    grid = make_grid(ims, nrow=nrow)
    img = torchvision.transforms.ToPILImage()(grid)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    img.close()


def benchmark_denoiser(
    sampler_fn,
    sampler_name,
    indexes,
    model,
    operator,
    diffusion_config,
    sigma_noise,
    config,
    device,
    **sampler_kwargs,
):
   
    csv_path = config["csv_path"] + f"_{sampler_name}.csv"

    times = []
    psnrs = []

    if not os.path.exists(csv_path):
        with open(csv_path, mode="w", newline="") as file:
            
            writer = csv.writer(file)
            writer.writerow(["idx", "psnr", "elapsed_time"])

            for idx in indexes:
                x0 = im2tensor(
                    plt.imread(f"ffhq256-1k-validation/{str(idx).zfill(5)}.png")
                ).to(device)

                y = operator.observe(x0, sigma_y=sigma_noise)

                start = time()
                x_rec, psnr_list = sampler_fn(
                    model=model,
                    diffusion_config=diffusion_config,
                    operator=operator,
                    x0=x0,
                    y=y,
                    **sampler_kwargs,
                )
                end = time()

                elapsed = end - start
                final_psnr = psnr_list[-1]

                times.append(elapsed)
                psnrs.append(final_psnr)

                writer.writerow([idx, final_psnr, elapsed])

    return psnrs, times