import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Sinusoidal Time Embedding
def time_embedding(time_steps, embedding_dim):
    """
    Standard sinusoidal timestep embedding used in diffusion models.
    """
    device = time_steps.device
    half_dim = embedding_dim // 2

    emb_scale = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)

    emb = time_steps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    return emb


# Downsampling block of the U-Net encoder
class Downsampling(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 embedding_dim,
                 down_sample=True,
                 num_heads=4,
                 num_layers=1,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embedding_dim = embedding_dim
        self.down_sample = down_sample
        self.num_heads = num_heads
        self.num_layers = num_layers

        # conv blocks
        self.unet_first_conv = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1
                )
            ) for i in range(num_layers)
        ])

        self.unet_second_convolution = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            ) for _ in range(num_layers)
        ])

        # spatial downsampling
        self.downsample_layer = (
            nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding = 1)
            if self.down_sample else nn.Identity()
        )

        # attention blocks
        self.attention_norms = nn.ModuleList([
            nn.GroupNorm(8, out_channels) for _ in range(num_layers)
        ])

        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])

        # residual projections
        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, 1)
            for i in range(num_layers)
        ])

        # timestep conditioning
        self.time_embedding_layers = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Linear(embedding_dim, out_channels)
            ) for _ in range(num_layers)
        ])

    def forward(self, x, embedding):
        """Apply conv + attention blocks then optional downsampling."""

        output = x

        for i in range(self.num_layers):

            # conv block
            unet_input = output
            output = self.unet_first_conv[i](unet_input)
            output = output + self.time_embedding_layers[i](embedding)[:, :, None, None]

            output = self.unet_second_convolution[i](output)
            output = output + self.residual_input_conv[i](unet_input)

            # spatial attention
            attn_input = self.attention_norms[i](output)

            B, C, H, W = attn_input.shape
            attn_input = attn_input.view(B, C, H * W).transpose(1, 2)

            attn_out, _ = self.attentions[i](attn_input, attn_input, attn_input)

            attn_out = attn_out.transpose(1, 2).view(B, C, H, W)

            output = output + attn_out

        return self.downsample_layer(output)


# Bottleneck block
class Middle(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 time_embed_dim,
                 num_heads=4,
                 num_layers=1,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_embed_dim = time_embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.unet_first_conv = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1
                )
            ) for i in range(num_layers + 1)
        ])

        self.time_emb_layers = nn.ModuleList([
            nn.Sequential(nn.ReLU(), nn.Linear(time_embed_dim, out_channels))
            for _ in range(num_layers + 1)
        ])

        self.unet_conv_second = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            ) for _ in range(num_layers + 1)
        ])

        self.attention_norms = nn.ModuleList([
            nn.GroupNorm(8, out_channels) for _ in range(num_layers)
        ])

        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])

        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, 1)
            for i in range(num_layers + 1)
        ])

    def forward(self, x, time_embed):
        """Latent U-Net block with attention."""

        out = x

        # first conv block
        unet_input = out
        out = self.unet_first_conv[0](out)
        out = out + self.time_emb_layers[0](time_embed)[:, :, None, None]
        out = self.unet_conv_second[0](out)
        out = out + self.residual_input_conv[0](unet_input)

        # attention + conv blocks
        for i in range(self.num_layers):

            attn_input = self.attention_norms[i](out)

            B, C, H, W = attn_input.shape
            attn_input = attn_input.view(B, C, H * W).transpose(1, 2)

            attn_out, _ = self.attentions[i](attn_input, attn_input, attn_input)
            attn_out = attn_out.transpose(1, 2).view(B, C, H, W)

            out = out + attn_out

            unet_input = out
            out = self.unet_conv_second[i + 1](out)
            out = out + self.time_emb_layers[i + 1](time_embed)[:, :, None, None]
            out = self.unet_conv_second[i + 1](out)
            out = out + self.residual_input_conv[i + 1](unet_input)

        return out


# Decoder block
class Upsampling(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 time_embed_dim,
                 up_sample=True,
                 num_heads=4,
                 num_layers=1,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_embed_dim = time_embed_dim
        self.up_sample = up_sample
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.unet_conv_first = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1
                )
            ) for i in range(num_layers)
        ])

        self.time_embed_layers = nn.ModuleList([
            nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, out_channels))
            for _ in range(num_layers)
        ])

        self.unet_conv_second = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            ) for _ in range(num_layers)
        ])

        self.attention_norms = nn.ModuleList([
            nn.GroupNorm(8, out_channels) for _ in range(num_layers)
        ])

        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])

        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, 1)
            for i in range(num_layers)
        ])

        # spatial upsampling
        self.up_sample_conv = (
            nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=4, stride=2, padding=1)
            if up_sample else nn.Identity()
        )

    def forward(self, x, skip, time_embed):
        """Upsample + fuse skip connection."""

        x = self.up_sample_conv(x)
        x = torch.cat([x, skip], dim=1)

        out = x

        for i in range(self.num_layers):

            unet_input = out
            out = self.unet_conv_first[i](out)

            out = out + self.time_embed_layers[i](time_embed)[:, :, None, None]

            out = self.unet_conv_second[i](out)
            out = out + self.residual_input_conv[i](unet_input)

            attn_input = self.attention_norms[i](out)

            B, C, H, W = attn_input.shape
            attn_input = attn_input.view(B, C, H * W).transpose(1, 2)

            attn_out, _ = self.attentions[i](attn_input, attn_input, attn_input)
            attn_out = attn_out.transpose(1, 2).view(B, C, H, W)

            out = out + attn_out

        return out