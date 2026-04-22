import torch
import torch.nn.functional as F
from torch import nn

from .layers import MLPBlock


class Attention(nn.Module):
    """SAM decoder 风格注意力。

    论文里的 mask decoder 不是普通分类头，而是一个轻量 Transformer decoder：
    prompt token 需要读图像特征，图像 token 也要反向接收 prompt 信息。这个类就是
    decoder 内部的多头注意力。

    官方 SAM2 里 cross-attention 会把 q/k/v 的内部通道降采样：
    `downsample_rate=2` 时，256 维 token 会先投影到 128 维再做注意力，
    既省计算，也和官方权重形状严格一致。
    """

    def __init__(self, dim: int, num_heads: int, downsample_rate: int = 1) -> None:
        super().__init__()
        internal_dim = dim // downsample_rate
        if internal_dim % num_heads != 0:
            raise ValueError("internal_dim 必须能整除 num_heads")
        self.num_heads = num_heads
        self.internal_dim = internal_dim
        self.head_dim = internal_dim // num_heads
        self.q_proj = nn.Linear(dim, internal_dim)
        self.k_proj = nn.Linear(dim, internal_dim)
        self.v_proj = nn.Linear(dim, internal_dim)
        self.out_proj = nn.Linear(internal_dim, dim)
        self.dropout_p = 0.0

    def _separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        """把 B,N,C 拆成 B,heads,N,C_per_head，和官方实现保持相同内存排列。"""

        bsz, num_tokens, channels = x.shape
        x = x.reshape(bsz, num_tokens, self.num_heads, channels // self.num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """把多头结果从 B,heads,N,C_per_head 合回 B,N,C。"""

        bsz, num_heads, num_tokens, channels_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(bsz, num_tokens, num_heads * channels_per_head)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q = self._separate_heads(self.q_proj(q))
        k = self._separate_heads(self.k_proj(k))
        v = self._separate_heads(self.v_proj(v))

        # PyTorch 的 SDPA 会执行和官方 SAM2 相同的缩放点积注意力路径：
        # softmax(q @ k^T / sqrt(d)) @ v。eval 时 dropout 固定为 0。
        dropout_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        out = self._recombine_heads(out)
        return self.out_proj(out)


class TwoWayAttentionBlock(nn.Module):
    """SAM mask decoder 的双向注意力块。

    这对应论文中“提示 token 和图像 token 双向交互”的思想：
    1. prompt token 自注意力；
    2. prompt token 查询图像 token；
    3. prompt token 经过 MLP；
    4. 图像 token 再查询 prompt token，把提示信息写回图像特征。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_dim: int,
        skip_first_layer_pe: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = Attention(dim, num_heads, downsample_rate=1)
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn_token_to_image = Attention(dim, num_heads, downsample_rate=2)
        self.norm2 = nn.LayerNorm(dim)
        # 官方 SAM two-way transformer 的 FFN 激活是 ReLU，不是 GELU。
        self.mlp = MLPBlock(dim, mlp_dim, activation=nn.ReLU)
        self.norm3 = nn.LayerNorm(dim)
        self.cross_attn_image_to_token = Attention(dim, num_heads, downsample_rate=2)
        self.norm4 = nn.LayerNorm(dim)
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self,
        tokens: torch.Tensor,
        image_tokens: torch.Tensor,
        token_pe: torch.Tensor,
        image_pe: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.skip_first_layer_pe:
            tokens = self.self_attn(tokens, tokens, tokens)
        else:
            tokens = tokens + self.self_attn(tokens + token_pe, tokens + token_pe, tokens)
        tokens = self.norm1(tokens)

        tokens = tokens + self.cross_attn_token_to_image(
            tokens + token_pe,
            image_tokens + image_pe,
            image_tokens,
        )
        tokens = self.norm2(tokens)

        tokens = tokens + self.mlp(tokens)
        tokens = self.norm3(tokens)

        image_tokens = image_tokens + self.cross_attn_image_to_token(
            image_tokens + image_pe,
            tokens + token_pe,
            tokens,
        )
        image_tokens = self.norm4(image_tokens)
        return tokens, image_tokens


class TwoWayTransformer(nn.Module):
    def __init__(self, depth: int, dim: int, num_heads: int, mlp_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            TwoWayAttentionBlock(
                dim,
                num_heads,
                mlp_dim,
                skip_first_layer_pe=(idx == 0),
            )
            for idx in range(depth)
        )
        self.final_attn_token_to_image = Attention(dim, num_heads, downsample_rate=2)
        self.norm_final_attn = nn.LayerNorm(dim)

    def forward(
        self,
        image_embedding: torch.Tensor,
        image_pe: torch.Tensor,
        tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, channels, height, width = image_embedding.shape
        image_tokens = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe_tokens = image_pe.flatten(2).permute(0, 2, 1)
        queries = tokens
        keys = image_tokens

        for layer in self.layers:
            queries, keys = layer(
                queries,
                keys,
                tokens,
                image_pe_tokens,
            )

        queries = queries + self.final_attn_token_to_image(
            queries + tokens,
            keys + image_pe_tokens,
            keys,
        )
        queries = self.norm_final_attn(queries)
        return queries, keys
