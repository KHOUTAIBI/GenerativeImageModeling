import torch
import torchvision
import numpy as np

import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm

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

def str2(chars):
    return "{:.2f}".format(chars)

def psnr(uref,ut,M=2):
    rmse = np.sqrt(np.mean((np.array(uref.cpu())-np.array(ut.cpu()))**2))
    return 20*np.log10(M/rmse)