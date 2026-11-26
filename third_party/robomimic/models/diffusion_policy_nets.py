""" This file contains nets used for Diffusion Policy. """
import math
from typing import Union

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
        Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            cond_dim,
            kernel_size=3,
            n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        """
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:,0,...]
        bias = embed[:,1,...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self, 
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=5,
        n_groups=8
        ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM 
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level. 
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            global_cond=None):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1,-2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)
        
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1,-2)
        # (B,T,C)
        return x


class SimpleResidualBlock1D(nn.Module):
    """Residual block with FiLM conditioning."""

    def __init__(self, channels, cond_dim, kernel_size=5, n_groups=8):
        super().__init__()
        self.cond_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, channels * 2),
        )
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.norm1 = nn.GroupNorm(n_groups, channels)
        self.norm2 = nn.GroupNorm(n_groups, channels)

    def forward(self, x, cond):
        residual = x
        out = self.norm1(torch.nn.functional.mish(self.conv1(x)))
        cond_embed = self.cond_mlp(cond).unsqueeze(-1)
        scale, bias = cond_embed.chunk(2, dim=1)
        out = scale * out + bias
        out = self.norm2(torch.nn.functional.mish(self.conv2(out)))
        return out + residual


class SimpleConditionalResNet1D(nn.Module):
    """
    Lightweight diffusion denoiser adapted from the user-provided simple model.
    Expects the same interface as ConditionalUnet1D.
    """

    def __init__(
        self,
        input_dim,
        global_cond_dim,
        num_train_timesteps=100,
        hidden_dim=256,
        cond_feature_dim=256,
        time_embed_dim=64,
        num_blocks=3,
        kernel_size=5,
        n_groups=8,
    ):
        super().__init__()
        self.num_train_timesteps = max(int(num_train_timesteps), 1)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Mish(),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.Mish(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )

        cond_dim = global_cond_dim + time_embed_dim
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, cond_feature_dim),
            nn.Mish(),
            nn.Linear(cond_feature_dim, cond_feature_dim),
        )

        self.input_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                SimpleResidualBlock1D(
                    channels=hidden_dim,
                    cond_dim=cond_feature_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, max(hidden_dim // 2, input_dim), kernel_size=1),
            nn.Mish(),
            nn.Conv1d(max(hidden_dim // 2, input_dim), input_dim, kernel_size=1),
        )

        print("number of parameters: {:e}".format(sum(p.numel() for p in self.parameters())))

    def _normalize_timestep(self, timestep, batch_size, device):
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.float32, device=device)
        else:
            timestep = timestep.to(device=device, dtype=torch.float32)

        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0)
        if timestep.ndim > 1:
            timestep = timestep.reshape(-1)
        timestep = timestep.expand(batch_size)

        max_t = float(max(self.num_train_timesteps - 1, 1))
        if torch.max(timestep).item() > 1.0:
            timestep = timestep / max_t
        return timestep

    def forward(self, sample: torch.Tensor, timestep: Union[torch.Tensor, float, int], global_cond=None):
        """
        sample: (B, T, C)
        timestep: (B,) or int or normalized float
        global_cond: (B, D)
        """
        x = sample.moveaxis(-1, -2)  # (B, C, T)
        B = x.shape[0]
        t = self._normalize_timestep(timestep=timestep, batch_size=B, device=x.device)
        t_embed = self.time_mlp(t)

        if global_cond is None:
            global_cond = torch.zeros(B, 0, device=x.device, dtype=x.dtype)
        cond = torch.cat([t_embed, global_cond], dim=-1)
        cond_feat = self.cond_encoder(cond)

        h = self.input_conv(x)
        for block in self.blocks:
            h = block(h, cond_feat)

        out = self.final_conv(h)
        return out.moveaxis(-1, -2)  # (B, T, C)
