

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvPool(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(kernel_size=pool_size, stride=pool_size),
        )


class MSConv(nn.Module):


    def __init__(self, channels=15, pool=4):
        super().__init__()
        self.spatial_branches = nn.ModuleList([
            _ConvPool(1, channels, (1, kernel), (1, pool))
            for kernel in (4, 8, 16)
        ])
        self.spatial_norm = nn.BatchNorm2d(channels)
        self.spectral_branches = nn.ModuleList([
            _ConvPool(channels, 1, (kernel, 1), (1, 1))
            for kernel in (2, 4)
        ])
        self.spectral_norm = nn.BatchNorm2d(1)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4 or tuple(x.shape[1:]) != (1, 4, 72):
            raise ValueError(
                "MSConv expects (B,1,4,72) or (B,4,72); got "
                f"{tuple(x.shape)}"
            )


        spatial = torch.cat([branch(x) for branch in self.spatial_branches], dim=-1)
        spatial = self.spatial_norm(spatial)
        spectral = torch.cat(
            [branch(spatial) for branch in self.spectral_branches], dim=2
        )
        return self.spectral_norm(spectral)


class MCAR(nn.Module):


    def __init__(self, d_model, core_dim=None, num_cores=4, dropout=0.1):
        super().__init__()
        self.d_model = int(d_model)
        self.core_dim = int(core_dim or max(1, d_model // 4))
        self.num_cores = int(num_cores)
        if self.num_cores < 1:
            raise ValueError("num_cores must be at least one")


        self.core_projection = nn.Linear(
            self.d_model, self.num_cores * self.core_dim
        )
        self.token_gate = nn.Linear(self.d_model, self.num_cores)
        self.sample_router = nn.Linear(self.d_model, self.num_cores)

        self.core_mlp = nn.Sequential(
            nn.Linear(self.core_dim, self.core_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.core_dim, self.core_dim),
        )
        self.redistribute = nn.Sequential(
            nn.Linear(self.d_model + self.core_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.d_model),
        )

    def forward(self, tokens):
        if tokens.ndim != 3 or tokens.shape[-1] != self.d_model:
            raise ValueError(
                f"MCAR expects (B,T,{self.d_model}); got {tuple(tokens.shape)}"
            )
        batch, token_count, _ = tokens.shape


        raw = F.gelu(self.core_projection(tokens)).view(
            batch, token_count, self.num_cores, self.core_dim
        )
        gates = self.token_gate(tokens).transpose(1, 2)
        aggregation = F.softmax(gates, dim=-1).transpose(1, 2)
        cores = (aggregation.unsqueeze(-1) * raw).sum(dim=1)


        route = F.softmax(self.sample_router(tokens.mean(dim=1)), dim=-1)
        transformed = self.core_mlp(cores)
        fused = (route.unsqueeze(-1) * transformed).sum(dim=1)

        
        fused = fused.unsqueeze(1).expand(-1, token_count, -1)
        return self.redistribute(torch.cat((tokens, fused), dim=-1))


class MCARBlock(nn.Module):
    

    def __init__(self, d_model, core_dim, num_cores, dropout):
        super().__init__()
        self.norm_mcar = nn.LayerNorm(d_model)
        self.mcar = MCAR(d_model, core_dim, num_cores, dropout)
        self.norm_mlp = nn.LayerNorm(d_model)
        hidden = 2 * d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, tokens):
        tokens = tokens + self.mcar(self.norm_mcar(tokens))
        return tokens + self.mlp(self.norm_mlp(tokens))


class Model(nn.Module):
    

    def __init__(self, configs, token_dim=4):
        super().__init__()
        self.token_dim = int(getattr(configs, "token_dim", token_dim))
        self.d_model = int(getattr(configs, "d_model", 128))
        self.num_cores = int(getattr(configs, "num_cores", 4))
        self.core_dim = int(getattr(configs, "core_dim", self.d_model // 4))
        self.dropout = float(getattr(configs, "dropout", 0.1))
        num_layers = int(getattr(configs, "mcar_layers", 2))
        if num_layers < 1:
            raise ValueError("mcar_layers must be at least one")

        self.input_projection = nn.Linear(self.token_dim, self.d_model)
        self.position = nn.Parameter(torch.zeros(1, 512, self.d_model))
        nn.init.normal_(self.position, mean=0.0, std=0.02)
        self.input_dropout = nn.Dropout(self.dropout)
        self.encoder = nn.ModuleList([
            MCARBlock(self.d_model, self.core_dim, self.num_cores, self.dropout)
            for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(self.d_model)
        self.projector = nn.Linear(self.d_model, int(configs.num_class))

    def forward_features(self, tokens):
        if tokens.ndim != 3 or tokens.shape[-1] != self.token_dim:
            raise ValueError(
                f"Model expects (B,T,{self.token_dim}); got {tuple(tokens.shape)}"
            )
        if tokens.shape[1] > self.position.shape[1]:
            raise ValueError("token sequence is longer than the positional embedding")
        tokens = self.input_projection(tokens)
        tokens = self.input_dropout(tokens + self.position[:, :tokens.shape[1]])
        for layer in self.encoder:
            tokens = layer(tokens)
        return self.output_norm(tokens).mean(dim=1)

    def forward(self, tokens):
        return self.projector(self.forward_features(tokens))


class MS_MCARNet(nn.Module):
    

    def __init__(self, configs):
        super().__init__()
        self.msconv = MSConv(
            channels=int(getattr(configs, "msconv_channels", 15)),
            pool=int(getattr(configs, "msconv_pool", 4)),
        )
        configs.token_dim = 4
        configs.num_cores = int(getattr(configs, "num_cores", 4))
        configs.core_dim = int(getattr(configs, "core_dim", configs.d_model // 4))
        configs.mcar_layers = int(getattr(configs, "mcar_layers", 2))
        self.encoder = Model(configs, token_dim=4)

    def forward_features(self, x):
        
        local_tokens = self.msconv(x).squeeze(1).transpose(1, 2).contiguous()
        return self.encoder.forward_features(local_tokens)

    def forward(self, x):
        return self.encoder.projector(self.forward_features(x))


__all__ = ["MSConv", "MCAR", "MCARBlock", "Model", "MS_MCARNet"]
