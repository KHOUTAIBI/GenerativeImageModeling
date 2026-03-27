import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
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


def _ensure_batched(img: torch.Tensor) -> torch.Tensor:
    """
    Convert [C,H,W] -> [1,C,H,W], keep [B,C,H,W] unchanged.
    """
    if img.dim() == 3:
        return img.unsqueeze(0)
    return img


def _to_display_range(img: torch.Tensor) -> torch.Tensor:
    """
    Clamp to [-1,1] for diffusion outputs.
    """
    return torch.clamp(img, -1.0, 1.0)


def save_benchmark_trajectory_grid(
    recon_lists,
    save_path,
    max_frames=10,
    value_range=(-1, 1),
    prefix_len=2, 
):
    if len(recon_lists) == 0:
        return

    processed_rows = []

    for recon_list in recon_lists:
        if len(recon_list) == 0:
            continue

        prefix = recon_list[:prefix_len]
        traj = recon_list[prefix_len:]

        # only subsample trajectory, not prefix
        remaining = max_frames - len(prefix)
        if remaining > 0 and len(traj) > remaining:
            idxs = torch.linspace(0, len(traj) - 1, steps=remaining).long().tolist()
            traj = [traj[i] for i in idxs]

        recon_list = prefix + traj

        row = []
        for x in recon_list:
            if x.dim() == 3:
                x = x.unsqueeze(0)
            x = x.detach().cpu().clamp(-1.0, 1.0)
            row.append(x)

        row = torch.cat(row, dim=0)
        processed_rows.append(row)

    if len(processed_rows) == 0:
        return

    num_cols = min(row.shape[0] for row in processed_rows)
    processed_rows = [row[:num_cols] for row in processed_rows]

    grid_tensor = torch.cat(processed_rows, dim=0)

    grid = make_grid(
        grid_tensor,
        nrow=num_cols,
        normalize=True,
        value_range=value_range,
        padding=2,
    )
    save_image(grid, save_path)


# --------------------------------
# Benchmarking PSNR and images in grid
# --------------------------------
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
    save_trajectories=True,
    max_trajectory_frames=10,
    **sampler_kwargs,
):
    csv_path = config["csv_path"] + f"_{sampler_name}_{operator.name}.csv"
    samples_dir = config.get("samples_dir", "./samples")
    os.makedirs(samples_dir, exist_ok=True)

    times = []
    psnrs = []
    all_recon_lists = []

    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["idx", "psnr", "elapsed_time"])

        for idx in indexes:
            x0 = im2tensor(
                plt.imread(f"ffhq256-1k-validation/{str(idx).zfill(5)}.png")
            ).to(device)

            x0 = _ensure_batched(x0)
            y = operator.observe(x0, sigma_y=sigma_noise)

            start = time()
            x_rec, psnr_list, recon_list = sampler_fn(
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

            if save_trajectories:
                row_list = [x0.detach().cpu(), y.detach().cpu()] + [r.detach().cpu() for r in recon_list]
                all_recon_lists.append(row_list)

        if len(psnrs) > 0:
            writer.writerow([])
            writer.writerow(["mean_psnr", np.mean(psnrs), "mean_time", np.mean(times)])
            writer.writerow(["std_psnr", np.std(psnrs), "std_time", np.std(times)])

    if save_trajectories and len(all_recon_lists) > 0:
        
        traj_grid_path = os.path.join(
            samples_dir, f"{sampler_name}_{operator.name}_benchmark_trajectories.png"
        )
        save_benchmark_trajectory_grid(
            recon_lists=all_recon_lists,
            save_path=traj_grid_path,
            max_frames=max_trajectory_frames,
        )

    return psnrs, times