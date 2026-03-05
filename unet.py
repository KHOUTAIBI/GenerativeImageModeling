import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy 

# The UNET class as in the DDPM/DDIM paper
class Unet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)



# Sinusoidal Time Embedding
def time_embedding(time_steps, embedding_dim):
    
    device = time_steps.device
    half_dim = embedding_dim // 2
    emb_scale = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
    emb = time_steps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb

# Downsampling class
class Downsampling(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 embedding_dim,
                 down_sample = True,
                 num_heads = 4,
                 num_layers = 1,
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embedding_dim = embedding_dim,
        self.down_sample = down_sample
        self.num_heads = num_heads
        self.num_layers = num_layers

        # layers, we do noot change the dimensions of the layers, as kernel = 3 and padidng = 1
        self.unet_first_conv = nn.ModuleList([
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=in_channels if i == 0 else out_channels, out_channels=out_channels, kernel_size=3, padding = 1, stride = 1),                         
            ) for i in range(num_layers)])

        self.unet_second_convolution = nn.ModuleList([
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding=1)
            ) for _ in range(num_layers)])

        self.downsample_layer = nn.Conv2d(in_channels=out_channels, out_channels = out_channels, kernel_size=4, stride = 2, padding = 0)\
            if self.down_sample else nn.Identity()        

        # Attention heads and self attention, residuals and time mebeddings
        self.attention_norms = nn.ModuleList([
            nn.GroupNorm(8, out_channels) for _ in range(num_layers)])

        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(out_channels, num_heads, batch_first=True) for _ in range(num_layers)])
        
        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, 1)
            for i in range(num_layers)])

        self.time_embedding_layers = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Linear(embedding_dim, out_channels)
            ) for _ in range(num_layers)
        ])
        
    def forward(self, x, embedding):
        
        # Forward
        output = x
        for i in range(self.num_layers):
            # First conv
            unet_input = output
            output = self.unet_first_conv[i](unet_input)
            output = output + self.time_embedding_layers[i](embedding)[:, :, None, None]
            # Second conv
            output = self.unet_second_convolution[i](output)
            output = output + self.residual_input_conv[i](unet_input)

            # Attention
            attn_input = self.attention_norms[i](output)
            B, C, H, W = attn_input.shape
            attn_input = attn_input.view(B, C, H * W).transpose(1, 2)
            attn_out, _ = self.attentions[i](attn_input, attn_input, attn_input)
            attn_out = attn_out.transpose(1, 2).view(B, C, H, W)
            output = output + attn_out

        downsampled_image = self.downsample_layer(output) # either downsample or not
        return downsampled_image
    

class Middle(nn.Module):
    def __init__(self, 
                 in_channels,
                out_channels,
                time_embed_dim,
                num_heads = 4,
                num_layers = 1,
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)   
