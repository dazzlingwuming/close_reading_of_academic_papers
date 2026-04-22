import torch
from torch import nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * mask / keep_prob


class MLP(nn.Module):
    """多层感知机。

    在 Hiera backbone 里它默认使用 GELU；在 SAM mask decoder 头里会显式传入
    ReLU。`sigmoid_output=True` 用来对齐官方 SAM2 的 IoU 质量预测头：最后一层
    linear 后直接输出 0~1 的质量分数。
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        out_dim: int | None = None,
        num_layers: int = 2,
        activation: type[nn.Module] = nn.GELU,
        dropout: float = 0.0,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        out_dim = dim if out_dim is None else out_dim
        if num_layers < 2:
            raise ValueError("num_layers 至少为 2")

        layers: list[nn.Module] = []
        in_dim = dim
        for layer_idx in range(num_layers):
            current_out = out_dim if layer_idx == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, current_out))
            if layer_idx != num_layers - 1:
                layers.append(activation())
                layers.append(nn.Dropout(dropout))
            else:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.net = nn.Sequential(*layers)
        self.sigmoid_output = sigmoid_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        if self.sigmoid_output:
            x = x.sigmoid()
        return x


class MLPBlock(nn.Module):
    """Transformer 内常见的 FFN：Linear -> GELU -> Linear。"""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = activation()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def nchw_to_nhwc(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 3, 1).contiguous()


def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 3, 1, 2).contiguous()


def window_partition(x: torch.Tensor, window_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """把 NHWC 特征切成窗口，必要时在右侧和底部补零。

    SAM2 的 Hiera backbone 主要依靠局部窗口注意力降低计算量；这里实现
    学习版窗口切分，避免 1024 输入时出现全局注意力的平方复杂度爆炸。
    """

    bsz, height, width, channels = x.shape
    pad_h = (window_size - height % window_size) % window_size
    pad_w = (window_size - width % window_size) % window_size
    if pad_h or pad_w:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    padded_h, padded_w = height + pad_h, width + pad_w
    x = x.view(
        bsz,
        padded_h // window_size,
        window_size,
        padded_w // window_size,
        window_size,
        channels,
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, channels)
    return windows, (padded_h, padded_w)


def window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    padded_hw: tuple[int, int],
    original_hw: tuple[int, int],
) -> torch.Tensor:
    """把窗口特征还原为 NHWC，并裁掉补零区域。"""

    padded_h, padded_w = padded_hw
    height, width = original_hw
    num_windows_per_image = (padded_h // window_size) * (padded_w // window_size)
    bsz = windows.shape[0] // num_windows_per_image
    x = windows.view(
        bsz,
        padded_h // window_size,
        padded_w // window_size,
        window_size,
        window_size,
        -1,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(bsz, padded_h, padded_w, -1)
    return x[:, :height, :width, :].contiguous()
