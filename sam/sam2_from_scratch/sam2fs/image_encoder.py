import torch
from torch import nn
import torch.nn.functional as F

from .config import SAM2Config, get_sam2_config
from .hiera import HieraBackbone


class FPNNeck(nn.Module):
    def __init__(
        self,
        in_channels: tuple[int, ...],
        out_channels: int,
        fpn_top_down_levels: tuple[int, ...] = (2, 3),
    ) -> None:
        super().__init__()
        # 官方 SAM2 FPN neck 只有 1x1 conv，不额外加 norm/output conv。
        # convs 的顺序是低分辨率到高分辨率：[768, 384, 192, 96]。
        self.convs = nn.ModuleList()
        for channels in reversed(in_channels):
            current = nn.Sequential()
            current.add_module("conv", nn.Conv2d(channels, out_channels, kernel_size=1))
            self.convs.append(current)
        self.fpn_top_down_levels = set(fpn_top_down_levels)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        out: list[torch.Tensor | None] = [None] * len(features)
        prev_features = None
        n = len(features) - 1
        for idx in range(n, -1, -1):
            lateral = self.convs[n - idx](features[idx])
            if idx in self.fpn_top_down_levels and prev_features is not None:
                top_down = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
                prev_features = lateral + top_down
            else:
                prev_features = lateral
            out[idx] = prev_features
        return [x for x in out if x is not None]


class SAM2ImageEncoder(nn.Module):
    def __init__(self, config: SAM2Config | str = "sam2.1_hiera_tiny") -> None:
        super().__init__()
        if isinstance(config, str):
            config = get_sam2_config(config)
        self.config = config
        self.backbone = HieraBackbone(config)
        self.neck = FPNNeck(self.backbone.out_channels, config.neck_dim)

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        features = self.backbone(image)
        return self.neck(features)
