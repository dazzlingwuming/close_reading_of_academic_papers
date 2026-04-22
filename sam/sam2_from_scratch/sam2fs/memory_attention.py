import torch
from torch import nn


class MemoryAttentionOp(nn.Module):
    """支持 k/v 输入维度不同的多头注意力。

    官方 SAM2 的 memory cross-attention 中，query 来自当前帧 256 维图像 token，
    key/value 来自 64 维 memory token，因此这里显式支持 `kv_in_dim`。
    """

    def __init__(self, embedding_dim: int = 256, num_heads: int = 1, kv_in_dim: int | None = None) -> None:
        super().__init__()
        kv_in_dim = embedding_dim if kv_in_dim is None else kv_in_dim
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim 必须能整除 num_heads")
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(kv_in_dim, embedding_dim)
        self.v_proj = nn.Linear(kv_in_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        bsz, q_len, dim = q.shape
        k_len = k.shape[1]
        q = self.q_proj(q).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(bsz, q_len, dim)
        return self.out_proj(out)


class MemoryAttentionLayer(nn.Module):
    """当前帧 token 读取历史 memory token 的 Transformer 层。"""

    def __init__(self, d_model: int = 256, num_heads: int = 1, dim_feedforward: int = 2048) -> None:
        super().__init__()
        self.self_attn = MemoryAttentionOp(d_model, num_heads)
        self.cross_attn_image = MemoryAttentionOp(d_model, num_heads, kv_in_dim=64)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()

    def forward(self, current: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        current = current + self.self_attn(current, current, current)
        current = self.norm1(current)
        current = current + self.cross_attn_image(current, memory, memory)
        current = self.norm2(current)
        ffn = self.linear2(self.dropout(self.activation(self.linear1(current))))
        current = current + ffn
        return self.norm3(current)


class MemoryAttention(nn.Module):
    """SAM2 风格 memory attention。

    官方实现还使用 RoPE 位置编码；这里先对齐可学习参数结构和数据流，RoPE 作为
    后续精细兼容项继续补。
    """

    def __init__(self, dim: int = 256, depth: int = 4, num_heads: int = 1) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            MemoryAttentionLayer(dim, num_heads, dim_feedforward=dim * 8)
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        current_features: torch.Tensor,
        memories: list[torch.Tensor] | None,
    ) -> torch.Tensor:
        if not memories:
            return current_features

        bsz, channels, height, width = current_features.shape
        current = current_features.flatten(2).transpose(1, 2)
        memory = torch.cat([m.flatten(2).transpose(1, 2) for m in memories], dim=1)
        for layer in self.layers:
            current = layer(current, memory)
        current = self.norm(current)
        return current.transpose(1, 2).reshape(bsz, channels, height, width)
