import math
import copy

import torch
from torch import nn
import torch.nn.functional as F

from .layers import DropPath, LayerNorm2d


class MaskDownSampler(nn.Module):
    """官方 SAM2 风格 mask 下采样器。

    默认 stride=2、total_stride=16，因此会做 4 次 2x 下采样，通道数按
    1 -> 4 -> 16 -> 64 -> 256 增长，最后再接 1x1 投影。这和官方
    `memory_encoder.mask_downsampler.encoder.*` 的权重形状一致。
    """

    def __init__(
        self,
        embed_dim: int = 256,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        total_stride: int = 16,
    ) -> None:
        super().__init__()
        num_layers = int(math.log2(total_stride) // math.log2(stride))
        if stride**num_layers != total_stride:
            raise ValueError("total_stride 必须是 stride 的整数次幂")

        self.encoder = nn.Sequential()
        in_chans = 1
        out_chans = 1
        for _ in range(num_layers):
            out_chans = in_chans * (stride**2)
            self.encoder.append(
                nn.Conv2d(
                    in_chans,
                    out_chans,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            self.encoder.append(LayerNorm2d(out_chans))
            self.encoder.append(nn.GELU())
            in_chans = out_chans
        self.encoder.append(nn.Conv2d(out_chans, embed_dim, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class CXBlock(nn.Module):
    """ConvNeXt 风格 fuser block，用于融合图像特征和 mask 特征。"""

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        padding: int = 3,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2).contiguous()
        return residual + self.drop_path(x)


class Fuser(nn.Module):
    def __init__(self, layer: nn.Module, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(copy.deepcopy(layer) for _ in range(num_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class MemoryEncoder(nn.Module):
    """把当前帧 mask 和图像特征编码成 64 维 memory feature。

    这个模块现在按官方 SAM2.1 tiny checkpoint 的参数形状实现：
    `pix_feat_proj`、`fuser.layers.*`、`out_proj` 都可以参与权重映射。
    """

    def __init__(self, image_dim: int = 256, memory_dim: int = 64) -> None:
        super().__init__()
        self.mask_downsampler = MaskDownSampler(embed_dim=image_dim)
        self.pix_feat_proj = nn.Conv2d(image_dim, image_dim, kernel_size=1)
        self.fuser = Fuser(CXBlock(image_dim), num_layers=2)
        self.out_proj = nn.Conv2d(image_dim, memory_dim, kernel_size=1)

    def forward(self, image_features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        if masks.ndim == 3:
            masks = masks[:, None, :, :]
        masks = F.sigmoid(masks.float())
        mask_features = self.mask_downsampler(masks)
        if mask_features.shape[-2:] != image_features.shape[-2:]:
            mask_features = F.interpolate(
                mask_features,
                size=image_features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        x = self.pix_feat_proj(image_features) + mask_features
        x = self.fuser(x)
        return self.out_proj(x)
