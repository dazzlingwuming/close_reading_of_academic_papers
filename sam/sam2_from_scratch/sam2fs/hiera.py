import torch
from torch import nn
import torch.nn.functional as F

from .config import SAM2Config
from .layers import DropPath, MLP, window_partition, window_unpartition


def do_pool(
    x: torch.Tensor,
    pool: nn.Module | None,
    norm: nn.Module | None = None,
) -> torch.Tensor:
    """对 NHWC token 网格做空间池化。

    Hiera 在 stage 切换时不是额外插入一个卷积下采样层，而是在 block 内对
    query/shortcut 做 q-pooling。这个函数就是论文里“分层降采样”的代码化。
    """

    if pool is None:
        return x
    x = x.permute(0, 3, 1, 2).contiguous()
    x = pool(x)
    x = x.permute(0, 2, 3, 1).contiguous()
    if norm is not None:
        x = norm(x)
    return x


class PatchEmbed(nn.Module):
    """图像分块嵌入，参数形状对齐 SAM2.1 checkpoint。

    官方 SAM2.1 Hiera 使用 7x7 卷积、stride=4、padding=3，所以 1024 输入会
    得到 256x256 的初始 token 网格。
    """

    def __init__(self, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=7,
            stride=4,
            padding=3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.permute(0, 2, 3, 1).contiguous()


class MultiScaleAttention(nn.Module):
    """Hiera 的多尺度注意力。

    和普通 MHA 的区别是 qkv 的输出维度可以从 dim 变为 dim_out，并且在 stage
    切换 block 中只对 query 做 q-pooling，从而产生更低分辨率的输出 token。
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_pool: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if dim_out % num_heads != 0:
            raise ValueError("dim_out 必须能整除 num_heads")
        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, height, width, _ = x.shape
        qkv = self.qkv(x).reshape(bsz, height * width, 3, self.num_heads, -1)
        q, k, v = torch.unbind(qkv, dim=2)

        if self.q_pool is not None:
            q = do_pool(q.reshape(bsz, height, width, -1), self.q_pool)
            height, width = q.shape[1:3]
            q = q.reshape(bsz, height * width, self.num_heads, -1)

        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )
        out = out.transpose(1, 2).reshape(bsz, height, width, -1)
        return self.proj(out)


class MultiScaleBlock(nn.Module):
    """Hiera block：Norm -> 多尺度注意力 -> 残差 -> MLP。

    stage 变化发生在某些 block 内：
    - `dim != dim_out` 时用线性投影改变通道；
    - `q_stride` 非空时对 query 和 shortcut 做 2x2 pooling 改变空间尺寸。
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float,
        drop_path: float,
        q_stride: tuple[int, int] | None,
        window_size: int,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.window_size = window_size
        self.q_stride = q_stride
        self.pool = (
            nn.MaxPool2d(kernel_size=q_stride, stride=q_stride, ceil_mode=False)
            if q_stride is not None
            else None
        )

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MultiScaleAttention(dim, dim_out, num_heads, q_pool=self.pool)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim_out, eps=1e-6)
        self.mlp = MLP(dim_out, int(dim_out * mlp_ratio), dim_out)

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)
        else:
            self.proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)

        if self.proj is not None:
            shortcut = self.proj(x)
        shortcut = do_pool(shortcut, self.pool)

        window_size = self.window_size
        if window_size > 0:
            original_hw = x.shape[1:3]
            x, padded_hw = window_partition(x, window_size)

        x = self.attn(x)

        if self.q_stride is not None and window_size > 0:
            window_size = window_size // self.q_stride[0]
            original_hw = shortcut.shape[1:3]
            pad_h = (window_size - original_hw[0] % window_size) % window_size
            pad_w = (window_size - original_hw[1] % window_size) % window_size
            padded_hw = (original_hw[0] + pad_h, original_hw[1] + pad_w)

        if self.window_size > 0:
            x = window_unpartition(x, window_size, padded_hw, original_hw)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class HieraBackbone(nn.Module):
    """SAM2.1 风格的 Hiera 主干。

    这个版本不再用 stage ModuleList 包住 block，而是使用官方 checkpoint 对应
    的扁平 `blocks.0 ... blocks.N` 结构，便于后续建立权重映射。
    """

    def __init__(self, config: SAM2Config, in_channels: int = 3) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed(in_channels, config.embed_dim)
        self.stage_ends = [
            sum(config.stage_depths[:idx]) - 1
            for idx in range(1, len(config.stage_depths) + 1)
        ]
        self.q_pool_blocks = [idx + 1 for idx in self.stage_ends[:-1]][: config.q_pool]
        self.global_att_blocks = set(config.global_att_blocks)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.embed_dim, 7, 7)
        )
        self.pos_embed_window = nn.Parameter(
            torch.zeros(1, config.embed_dim, config.window_spec[0], config.window_spec[0])
        )

        depth = sum(config.stage_depths)
        drop_rates = torch.linspace(0, config.drop_path_rate, depth).tolist()

        embed_dim = config.embed_dim
        num_heads = config.stage_heads[0]
        current_stage = 1
        blocks: list[MultiScaleBlock] = []
        for idx in range(depth):
            dim_out = embed_dim
            window_size = config.window_spec[current_stage - 1]
            if idx in self.global_att_blocks:
                window_size = 0

            if idx - 1 in self.stage_ends:
                dim_out = embed_dim * 2
                num_heads *= 2
                current_stage += 1

            blocks.append(
                MultiScaleBlock(
                    dim=embed_dim,
                    dim_out=dim_out,
                    num_heads=num_heads,
                    mlp_ratio=config.mlp_ratio,
                    drop_path=drop_rates[idx],
                    q_stride=config.q_stride if idx in self.q_pool_blocks else None,
                    window_size=window_size,
                )
            )
            embed_dim = dim_out

        self.blocks = nn.ModuleList(blocks)
        self.out_channels = tuple(block.dim_out for block in self.blocks if False)
        self.out_channels = tuple(self.blocks[idx].dim_out for idx in self.stage_ends)

    def _get_pos_embed(self, hw: tuple[int, int]) -> torch.Tensor:
        height, width = hw
        pos_embed = F.interpolate(self.pos_embed, size=(height, width), mode="bicubic")
        repeats = [
            pos_embed.shape[-2] // self.pos_embed_window.shape[-2],
            pos_embed.shape[-1] // self.pos_embed_window.shape[-1],
        ]
        pos_embed = pos_embed + self.pos_embed_window.tile([1, 1, *repeats])
        return pos_embed.permute(0, 2, 3, 1).contiguous()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.patch_embed(x)
        x = x + self._get_pos_embed(x.shape[1:3])

        outputs = []
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.stage_ends:
                outputs.append(x.permute(0, 3, 1, 2).contiguous())
        return outputs
