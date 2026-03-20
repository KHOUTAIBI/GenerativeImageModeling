import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from tqdm import tqdm
import yaml
import argparse
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