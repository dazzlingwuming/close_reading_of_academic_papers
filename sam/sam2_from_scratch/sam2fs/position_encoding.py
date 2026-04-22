import math

import torch
from torch import nn


class PositionEmbeddingRandom(nn.Module):
    """随机傅里叶位置编码。

    SAM 系列对点和框坐标使用随机傅里叶特征，把归一化坐标 (x, y) 映射到
    sin/cos 高频空间。这样提示 token 能携带明确的几何位置信息。
    """

    def __init__(self, num_pos_feats: int = 128, scale: float = 1.0) -> None:
        super().__init__()
        gaussian = scale * torch.randn(2, num_pos_feats)
        self.register_buffer("gaussian_matrix", gaussian)

    def encode_coords(
        self,
        coords: torch.Tensor,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        height, width = image_size
        coords = coords.clone().float()
        coords[..., 0] = coords[..., 0] / max(width, 1)
        coords[..., 1] = coords[..., 1] / max(height, 1)
        coords = coords * 2.0 - 1.0
        projected = (2.0 * math.pi) * coords @ self.gaussian_matrix
        return torch.cat([projected.sin(), projected.cos()], dim=-1)

    def forward_with_coords(
        self,
        coords_input: torch.Tensor,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        return self.encode_coords(coords_input, image_size)

    def forward(self, size: tuple[int, int]) -> torch.Tensor:
        height, width = size
        device = self.gaussian_matrix.device
        grid = torch.ones((height, width), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / height
        x_embed = x_embed / width
        coords = torch.stack([x_embed, y_embed], dim=-1)
        coords = 2 * coords - 1
        projected = (2.0 * math.pi) * coords @ self.gaussian_matrix
        encoded = torch.cat([projected.sin(), projected.cos()], dim=-1)
        return encoded.permute(2, 0, 1)
