import math
from collections.abc import Callable
from typing import Optional
import numpy as np
import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub, use_kernel_func_from_hub, use_kernelized_func
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring as hf_auto_docstring, can_return_tuple
from transformers.utils.generic import maybe_autocast, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs
from .configuration import Qwen3Config
from transformers.cache_utils import DynamicCache


def auto_docstring(*args, **kwargs):
    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        obj = args[0]
        try:
            return hf_auto_docstring(obj)
        except Exception:
            return obj

    decorator = hf_auto_docstring(*args, **kwargs)

    def wrapped(obj):
        try:
            return decorator(obj)
        except Exception:
            return obj

    return wrapped


def normal_ppf(quantiles):
    quantiles = torch.as_tensor(quantiles, dtype=torch.float64)
    return (torch.sqrt(torch.tensor(2.0, dtype=torch.float64)) * torch.erfinv(2 * quantiles - 1)).cpu().numpy()


#定义一个新的 TurboQuantCache 类，继承自 DynamicCache，并重写 update 方法以适应 TurboQuant 的需求
class TurboQuantCache(DynamicCache):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bits = config.turboquant_bits
        self.mse_bits = max(self.bits - 1, 0)
        self.head_dim = config.head_dim
        self.index_mask = (1 << self.mse_bits) - 1 if self.mse_bits > 0 else 0
        self.packed_dim = (self.head_dim * self.mse_bits + 7) // 8 if self.mse_bits > 0 else 0
        self.qjl_packed_dim = (self.head_dim + 7) // 8
        self.qjl_scale = math.sqrt(math.pi / 2.0) / self.head_dim

        self.key_mse_cache = [None] * config.num_hidden_layers
        self.key_qjl_cache = [None] * config.num_hidden_layers
        self.key_gamma_cache = [None] * config.num_hidden_layers
        self.key_norm_cache = [None] * config.num_hidden_layers
        self.value_mse_cache = [None] * config.num_hidden_layers
        self.value_qjl_cache = [None] * config.num_hidden_layers
        self.value_gamma_cache = [None] * config.num_hidden_layers
        self.value_norm_cache = [None] * config.num_hidden_layers

        self.layer_rot_mats = [None] * config.num_hidden_layers
        self.layer_codebooks = [None] * config.num_hidden_layers
        self.layer_proj_mats = [None] * config.num_hidden_layers

    def set_layer_params(self, layer_idx, rot_mat, codebook, proj_mat):
        self.layer_rot_mats[layer_idx] = rot_mat
        self.layer_codebooks[layer_idx] = codebook
        self.layer_proj_mats[layer_idx] = proj_mat

    def _quantize(self, x, codebook):
        """x: [..., head_dim] 浮点张量，返回整数索引"""
        if self.mse_bits == 0:
            return None
        x_flat = x.reshape(-1, 1)
        dist = (x_flat - codebook.to(x.device)) ** 2
        idx = torch.argmin(dist, dim=1).reshape(x.shape)
        return idx.to(torch.uint8)

    def _dequantize(self, idx, codebook):
        if idx is None:
            raise ValueError("MSE indices are not available when turboquant_bits=1.")
        return codebook.to(idx.device)[idx.to(torch.long)]

    def _pack_indices(self, idx):
        if idx is None:
            return None
        original_shape = idx.shape[:-1]
        flat = idx.reshape(-1, idx.shape[-1]).to(torch.uint8)
        packed = torch.zeros((flat.shape[0], self.packed_dim), dtype=torch.uint8, device=idx.device)
        bit_pos = 0
        for col in range(flat.shape[1]):
            byte_idx = bit_pos // 8
            offset = bit_pos % 8
            value = flat[:, col] & self.index_mask
            packed[:, byte_idx] |= value << offset
            if offset + self.mse_bits > 8:
                packed[:, byte_idx + 1] |= value >> (8 - offset)
            bit_pos += self.mse_bits
        return packed.reshape(*original_shape, self.packed_dim)

    def _unpack_indices(self, packed):
        if packed is None:
            return None
        original_shape = packed.shape[:-1]
        flat = packed.reshape(-1, packed.shape[-1]).to(torch.int32)
        cols = torch.arange(self.head_dim, device=packed.device, dtype=torch.int32)
        bit_pos = cols * self.mse_bits
        byte_idx = bit_pos // 8
        offset = bit_pos % 8

        values = (flat[:, byte_idx] >> offset) & self.index_mask
        cross_byte = offset + self.mse_bits > 8
        if torch.any(cross_byte):
            remaining = 8 - offset[cross_byte]
            next_bits = self.mse_bits - remaining
            next_mask = ((1 << next_bits) - 1).to(torch.int32)
            values[:, cross_byte] |= (flat[:, byte_idx[cross_byte] + 1] & next_mask) << remaining

        return values.to(torch.long).reshape(*original_shape, self.head_dim)

    def _pack_signs(self, signs):
        original_shape = signs.shape[:-1]
        flat = (signs > 0).reshape(-1, signs.shape[-1]).to(torch.uint8)
        packed = torch.zeros((flat.shape[0], self.qjl_packed_dim), dtype=torch.uint8, device=signs.device)
        for col in range(flat.shape[1]):
            byte_idx = col // 8
            offset = col % 8
            packed[:, byte_idx] |= flat[:, col] << offset
        return packed.reshape(*original_shape, self.qjl_packed_dim)

    def _unpack_signs(self, packed, dtype):
        original_shape = packed.shape[:-1]
        flat = packed.reshape(-1, packed.shape[-1]).to(torch.int32)
        cols = torch.arange(self.head_dim, device=packed.device, dtype=torch.int32)
        byte_idx = cols // 8
        offset = cols % 8
        bits = (flat[:, byte_idx] >> offset) & 1
        unpacked = bits.to(dtype) * 2 - 1
        return unpacked.reshape(*original_shape, self.head_dim)

    def _append_cache(self, caches, layer_idx, tensor):
        if tensor is None:
            return
        if caches[layer_idx] is None:
            caches[layer_idx] = tensor
        else:
            caches[layer_idx] = torch.cat([caches[layer_idx], tensor], dim=2)

    def _rotate(self, x, rot_mat, inverse=False):
        """x: [batch, heads, head_dim] 或 [batch, heads, seq_len, head_dim]
           rot_mat: [num_heads, head_dim, head_dim]
           当 inverse=False 时应用旋转，否则应用逆旋转
        """
        # 处理输入形状
        if x.dim() == 4:  # [batch, heads, seq_len, head_dim]
            # 将 seq_len 维度移到前面，方便批量处理
            batch, heads, seq_len, head_dim = x.shape
            x_reshaped = x.permute(0, 2, 1, 3).reshape(batch * seq_len, heads, head_dim)
            if inverse:
                rot = rot_mat.transpose(-1, -2)  # 逆旋转使用转置
            else:
                rot = rot_mat
            # 应用旋转: (b*s, heads, head_dim) @ (heads, head_dim, head_dim) -> (b*s, heads, head_dim)
            rotated = torch.einsum('bhd,hde->bhe', x_reshaped, rot)
            rotated = rotated.reshape(batch, seq_len, heads, head_dim).permute(0, 2, 1, 3)
            return rotated
        else:  # [batch, heads, head_dim]
            if inverse:
                rot = rot_mat.transpose(-1, -2)
            else:
                rot = rot_mat
            return torch.einsum('bhd,hde->bhe', x, rot)

    def update(self, key_states, value_states, layer_idx):
        """
        key_states, value_states: [batch, num_kv_heads, seq_len, head_dim]
        """
        rot_mat = self.layer_rot_mats[layer_idx]
        codebook = self.layer_codebooks[layer_idx]
        proj_mat = self.layer_proj_mats[layer_idx]

        self._quantize_and_store(key_states, layer_idx, rot_mat, codebook, proj_mat, is_key=True)
        self._quantize_and_store(value_states, layer_idx, rot_mat, codebook, proj_mat, is_key=False)

        return None, None

    def _quantize_and_store(self, states, layer_idx, rot_mat, codebook, proj_mat, is_key):
        norms = states.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        unit_states = states / norms
        rotated_states = self._rotate(unit_states, rot_mat, inverse=False)

        idx = self._quantize(rotated_states, codebook)
        mse_hat_rot = self._dequantize(idx, codebook) if self.mse_bits > 0 else torch.zeros_like(rotated_states)
        mse_hat = self._rotate(mse_hat_rot, rot_mat, inverse=True)

        residual = unit_states - mse_hat
        gamma = residual.norm(dim=-1, keepdim=True)
        residual_unit = residual / gamma.clamp_min(1e-6)
        qjl_projection = torch.einsum("bhsd,hde->bhse", residual_unit, proj_mat.to(states.dtype))
        qjl_signs = torch.where(qjl_projection >= 0, torch.ones_like(qjl_projection), -torch.ones_like(qjl_projection))

        mse_cache = self.key_mse_cache if is_key else self.value_mse_cache
        qjl_cache = self.key_qjl_cache if is_key else self.value_qjl_cache
        gamma_cache = self.key_gamma_cache if is_key else self.value_gamma_cache
        norm_cache = self.key_norm_cache if is_key else self.value_norm_cache

        self._append_cache(mse_cache, layer_idx, self._pack_indices(idx))
        self._append_cache(qjl_cache, layer_idx, self._pack_signs(qjl_signs))
        self._append_cache(gamma_cache, layer_idx, gamma.to(torch.float16))
        self._append_cache(norm_cache, layer_idx, norms.to(torch.float16))

    def get_prod_block(self, layer_idx, start, end, dtype=None):
        if dtype is None:
            dtype = torch.float16

        def load_block(mse_cache, qjl_cache, gamma_cache, norm_cache):
            packed_idx = mse_cache[layer_idx][:, :, start:end, :] if mse_cache[layer_idx] is not None else None
            idx = self._unpack_indices(packed_idx)
            signs = self._unpack_signs(qjl_cache[layer_idx][:, :, start:end, :], dtype)
            gamma = gamma_cache[layer_idx][:, :, start:end, :].to(dtype)
            norms = norm_cache[layer_idx][:, :, start:end, :].to(dtype)
            return idx, signs, gamma, norms

        key_idx, key_signs, key_gamma, key_norms = load_block(
            self.key_mse_cache, self.key_qjl_cache, self.key_gamma_cache, self.key_norm_cache
        )
        value_idx, value_signs, value_gamma, value_norms = load_block(
            self.value_mse_cache, self.value_qjl_cache, self.value_gamma_cache, self.value_norm_cache
        )
        return {
            "key_idx": key_idx,
            "key_signs": key_signs,
            "key_gamma": key_gamma,
            "key_norms": key_norms,
            "value_idx": value_idx,
            "value_signs": value_signs,
            "value_gamma": value_gamma,
            "value_norms": value_norms,
        }

    def get_seq_length(self, layer_idx: int = 0) -> int:
        cache = self.key_qjl_cache[layer_idx]
        if cache is None:
            return 0
        return cache.shape[2]


@use_kernel_forward_from_hub("RMSNorm")
class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Qwen3RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Qwen3Config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: Qwen3Config | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@use_kernel_func_from_hub("rotary_pos_emb")
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def turboquant_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    cache: "TurboQuantCache",
    layer_idx: int,
    attention_mask: torch.Tensor | None,
    scaling: float,
):
    block_size = getattr(module, "turboquant_block_size", 256)
    total_seq_len = cache.get_seq_length(layer_idx)

    rot_mats = module.rot_mat.repeat_interleave(module.num_key_value_groups, dim=0).to(query.device)
    proj_mats = module.proj_mat.repeat_interleave(module.num_key_value_groups, dim=0).to(query.device)
    query_rot = torch.einsum("bhqd,hde->bhqe", query, rot_mats)
    query_proj = torch.einsum("bhqd,hde->bhqe", query, proj_mats)
    inv_rot_mats = rot_mats.transpose(-1, -2)
    inv_proj_mats = proj_mats.transpose(-1, -2)
    qjl_scale = cache.qjl_scale

    output = torch.zeros_like(query)
    running_max = torch.full(
        (*query_rot.shape[:-1], 1),
        torch.finfo(torch.float32).min,
        dtype=torch.float32,
        device=query.device,
    )
    running_sum = torch.zeros_like(running_max)

    for start in range(0, total_seq_len, block_size):
        end = min(start + block_size, total_seq_len)
        block = cache.get_prod_block(layer_idx, start, end, dtype=query.dtype)
        key_norms = repeat_kv(block["key_norms"], module.num_key_value_groups)
        key_residual = repeat_kv(block["key_norms"] * block["key_gamma"] * block["key_signs"], module.num_key_value_groups)
        value_norms = repeat_kv(block["value_norms"], module.num_key_value_groups)
        value_residual = repeat_kv(
            block["value_norms"] * block["value_gamma"] * block["value_signs"], module.num_key_value_groups
        )

        qjl_logits = qjl_scale * torch.matmul(query_proj, key_residual.transpose(2, 3))
        mse_logits = torch.zeros_like(qjl_logits)
        key_idx = repeat_kv(block["key_idx"], module.num_key_value_groups) if block["key_idx"] is not None else None
        value_idx = repeat_kv(block["value_idx"], module.num_key_value_groups) if block["value_idx"] is not None else None
        if key_idx is not None:
            key_centroids = module.codebook.to(query.dtype)[key_idx]
            mse_logits = torch.matmul(query_rot, (key_norms * key_centroids).transpose(2, 3))

        logits = (mse_logits + qjl_logits) * scaling
        if attention_mask is not None:
            logits = logits + attention_mask[..., start:end]

        logits_fp32 = logits.to(torch.float32)
        block_max = logits_fp32.max(dim=-1, keepdim=True).values
        new_max = torch.maximum(running_max, block_max)

        old_scale = torch.exp(running_max - new_max)
        block_scale = torch.exp(logits_fp32 - new_max)

        block_mse = torch.zeros_like(torch.matmul(block_scale.to(query.dtype), value_residual))
        if value_idx is not None:
            value_centroids = module.codebook.to(query.dtype)[value_idx]
            block_mse = torch.matmul(block_scale.to(query.dtype), value_norms * value_centroids)
        block_qjl = torch.matmul(block_scale.to(query.dtype), value_residual)
        block_output = torch.einsum("bhqd,hde->bhqe", block_mse, inv_rot_mats)
        block_output = block_output + qjl_scale * torch.einsum("bhqd,hde->bhqe", block_qjl, inv_proj_mats)

        output = output * old_scale.to(output.dtype)
        output = output + block_output
        running_sum = running_sum * old_scale + block_scale.sum(dim=-1, keepdim=True)
        running_max = new_max

    attn_output = (output / running_sum.clamp_min(1e-6).to(output.dtype)).transpose(1, 2).contiguous()
    return attn_output, None


@use_kernelized_func(apply_rotary_pos_emb)
class Qwen3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None

        #如果配置启用 TurboQuant，则设置相关参数并预生成旋转矩阵和码本
        self.use_turboquant = getattr(config, 'use_turboquant', False)
        if self.use_turboquant:
            self.bits = config.turboquant_bits
            self.turboquant_block_size = getattr(config, "turboquant_block_size", 256)
            self.head_dim = config.head_dim
            num_heads = config.num_key_value_heads
            # 为每个头生成随机正交矩阵（旋转矩阵），使用 float32 避免 QR 分解错误
            mat = torch.randn(num_heads, self.head_dim, self.head_dim, device='cpu', dtype=torch.float32)
            q, _ = torch.linalg.qr(mat)
            self.register_buffer('rot_mat', q.to(torch.float16))  # [num_heads, head_dim, head_dim]
            proj = torch.randn(num_heads, self.head_dim, self.head_dim, device='cpu', dtype=torch.float32)
            self.register_buffer('proj_mat', proj.to(torch.float16))
            # 预计算标量量化码本（简化版：线性分割）
            self.register_buffer('codebook', self._compute_codebook())
        else:
            self.rot_mat = None
            self.proj_mat = None
            self.codebook = None

    def _compute_codebook(self):
        """生成最优标量码本（基于标准正态分布，缩放至方差 1/head_dim）"""
        b = max(self.bits - 1, 0)
        if b == 0:
            return torch.empty(0, dtype=torch.float16)
        # 方差：旋转后坐标方差 = 1/head_dim
        variance = 1.0 / self.head_dim
        std = math.sqrt(variance)

        # 预定义已知的最优码本（对于标准正态分布，方差1）
        if b == 1:
            centers_std = np.array([-math.sqrt(2.0 / math.pi), math.sqrt(2.0 / math.pi)])
        elif b == 2:
            centers_std = np.array([-1.51, -0.453, 0.453, 1.51])
        elif b == 3:
            # 3-bit 高斯最优码本（8个点）
            centers_std = np.array([-3.596, -2.057, -1.045, -0.276, 0.276, 1.045, 2.057, 3.596])
        elif b == 4:
            # 4-bit 高斯最优码本（16个点），这里使用正态分位数近似
            n = 2 ** b
            quantiles = np.linspace(0, 1, n + 1)
            centers_std = normal_ppf(quantiles[1:-1])
            centers_std = (centers_std - np.mean(centers_std)) / np.std(centers_std)
        else:
            # b > 4: 使用分位数法生成
            n = 2 ** b
            quantiles = np.linspace(0, 1, n + 1)
            centers_std = normal_ppf(quantiles[1:-1])
            centers_std = (centers_std - np.mean(centers_std)) / np.std(centers_std)

        # 缩放至目标方差
        centers = centers_std * std
        return torch.tensor(centers, dtype=torch.float16)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            if isinstance(past_key_values, TurboQuantCache) and self.use_turboquant:
                # 确保该层的旋转矩阵和码本已设置
                if past_key_values.layer_rot_mats[self.layer_idx] is None:
                    past_key_values.set_layer_params(self.layer_idx, self.rot_mat, self.codebook, self.proj_mat)
                past_key_values.update(key_states, value_states, self.layer_idx)
                attn_output, attn_weights = turboquant_attention_forward(
                    self,
                    query_states,
                    past_key_values,
                    self.layer_idx,
                    attention_mask,
                    self.scaling,
                )
                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                attn_output = self.o_proj(attn_output)
                return attn_output, attn_weights
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class Qwen3PreTrainedModel(PreTrainedModel):
    config: Qwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Qwen3DecoderLayer,
        "attentions": Qwen3Attention,
    }


@auto_docstring
class Qwen3Model(Qwen3PreTrainedModel):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # if use_cache and past_key_values is None:
        #     past_key_values = DynamicCache(config=self.config)
        #如果启用 TurboQuant 且 past_key_values 为空，则使用 TurboQuantCache 替代 DynamicCache
        if use_cache and past_key_values is None:
            if self.config.use_turboquant:
                from .modeling import TurboQuantCache  # 确保 TurboQuantCache 已经定义
                past_key_values = TurboQuantCache(self.config)
            else:
                past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


@auto_docstring
class Qwen3ForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Qwen3ForSequenceClassification(GenericForSequenceClassification, Qwen3PreTrainedModel):
    pass


class Qwen3ForTokenClassification(GenericForTokenClassification, Qwen3PreTrainedModel):
    pass


class Qwen3ForQuestionAnswering(GenericForQuestionAnswering, Qwen3PreTrainedModel):
    base_model_prefix = "transformer"  # For BC, where `transformer` was used instead of `model`


__all__ = [
    "Qwen3ForCausalLM",
    "Qwen3ForQuestionAnswering",
    "Qwen3PreTrainedModel",
    "Qwen3Model",
    "Qwen3ForSequenceClassification",
    "Qwen3ForTokenClassification",
]
