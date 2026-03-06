import torch
import torch.nn as nn
from blocks import Downsampling, Middle, Upsampling, time_embedding


class Unet(nn.Module):
    """
    U-Net backbone used in diffusion models (DDPM/DDIM style).

    Expects blocks:
        - Downsampling
        - Middle
        - Upsampling
        - time_embedding
    defined in blocks.py.
    """

    def __init__(self, model_config):
        super().__init__()

        im_channels = model_config["im_channels"]
        self.down_channels = model_config["down_channels"]
        self.mid_channels = model_config["mid_channels"]
        self.time_embed_dim = model_config["time_emb_dim"]
        self.down_sample = model_config["down_sample"]
        self.num_down_layers = model_config["num_down_layers"]
        self.num_mid_layers = model_config["num_mid_layers"]
        self.num_up_layers = model_config["num_up_layers"]

        # basic consistency checks
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-2]
        assert len(self.down_sample) == len(self.down_channels) - 1

        # time embedding projection
        self.time_proj = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        # first convolution
        self.conv_in = nn.Conv2d(
            in_channels=im_channels,
            out_channels=self.down_channels[0],
            kernel_size=3,
            padding=1
        )

        # encoder
        self.downs = nn.ModuleList()
        for i in range(len(self.down_channels) - 1):
            self.downs.append(
                Downsampling(
                    in_channels=self.down_channels[i],
                    out_channels=self.down_channels[i + 1],
                    embedding_dim=self.time_embed_dim,
                    down_sample=self.down_sample[i],
                    num_layers=self.num_down_layers,
                )
            )

        # bottleneck
        self.mids = nn.ModuleList()
        for i in range(len(self.mid_channels) - 1):
            self.mids.append(
                Middle(
                    in_channels=self.mid_channels[i],
                    out_channels=self.mid_channels[i + 1],
                    time_embed_dim=self.time_embed_dim,
                    num_layers=self.num_mid_layers,
                )
            )

        # decoder
        self.ups = nn.ModuleList()
        for i in reversed(range(len(self.down_channels) - 1)):
            in_ch = self.down_channels[i] * 2  # skip concat
            out_ch = self.down_channels[i - 1] if i > 0 else self.down_channels[0]

            self.ups.append(
                Upsampling(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    time_embed_dim=self.time_embed_dim,
                    up_sample=self.down_sample[i],
                    num_layers=self.num_up_layers,
                )
            )

        # output projection
        self.norm_out = nn.GroupNorm(8, self.down_channels[0])
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(
            in_channels=self.down_channels[0],
            out_channels=im_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x, t):
        """
        x : (B, C, H, W) noisy image
        t : diffusion timestep (B,)
        """

        # initial projection
        out = self.conv_in(x)

        # prepare timestep embedding
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=x.device)

        t = t.to(x.device).long()

        if t.dim() == 0:
            t = t.unsqueeze(0).repeat(x.shape[0])

        t_emb = time_embedding(t, self.time_embed_dim)
        t_emb = self.time_proj(t_emb)

        # encoder
        skip_connections = []
        for down in self.downs:
            skip_connections.append(out)
            out = down(out, t_emb)

        # bottleneck
        for mid in self.mids:
            out = mid(out, t_emb)

        # decoder
        for up in self.ups:
            skip = skip_connections.pop()
            out = up(out, skip, t_emb)

        # output layer
        out = self.norm_out(out)
        out = self.act_out(out)
        out = self.conv_out(out)

        return out