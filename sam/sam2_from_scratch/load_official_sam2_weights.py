"""把官方 SAM2.1 checkpoint 尽量加载到手写学习版。

注意：
    这个脚本不是“魔法兼容”。它会显式打印哪些参数成功映射，哪些参数因为
    我们的学习版结构尚未完全对齐而被跳过。这样可以逐步推进兼容，而不是
    用 strict=False 掩盖问题。
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import torch

from sam2fs import SAM2FromScratch


def map_trunk_key(official_key: str) -> str | None:
    prefix = "image_encoder.trunk."
    if not official_key.startswith(prefix):
        return None

    rest = official_key[len(prefix) :]
    if rest in {"pos_embed", "pos_embed_window", "patch_embed.proj.weight", "patch_embed.proj.bias"}:
        return "image_encoder.backbone." + rest

    if not rest.startswith("blocks."):
        return None

    # 官方：image_encoder.trunk.blocks.0.mlp.layers.0.weight
    # 我们：image_encoder.backbone.blocks.0.mlp.net.0.weight
    rest = rest.replace(".mlp.layers.0.", ".mlp.net.0.")
    rest = rest.replace(".mlp.layers.1.", ".mlp.net.3.")
    return "image_encoder.backbone." + rest


def map_top_level_key(official_key: str) -> str | None:
    if official_key in {
        "maskmem_tpos_enc",
        "no_mem_embed",
        "no_mem_pos_enc",
        "no_obj_ptr",
        "no_obj_embed_spatial",
    }:
        return official_key
    if official_key.startswith("mask_downsample."):
        return official_key
    if official_key.startswith("obj_ptr_proj.") or official_key.startswith("obj_ptr_tpos_proj."):
        return official_key
    return None


def map_neck_key(official_key: str) -> str | None:
    if official_key.startswith("image_encoder.neck.convs."):
        return official_key
    return None


def map_prompt_key(official_key: str) -> str | None:
    prefix = "sam_prompt_encoder."
    if not official_key.startswith(prefix):
        return None
    rest = official_key[len(prefix) :]
    if rest == "pe_layer.positional_encoding_gaussian_matrix":
        return "prompt_encoder.pe_layer.gaussian_matrix"
    if rest.startswith("point_embeddings.") or rest.startswith("mask_downscaling."):
        return "prompt_encoder." + rest
    if rest in {"not_a_point_embed.weight", "no_mask_embed.weight"}:
        return "prompt_encoder." + rest
    return None


def map_mask_decoder_key(official_key: str) -> str | None:
    prefix = "sam_mask_decoder."
    if not official_key.startswith(prefix):
        return None
    rest = official_key[len(prefix) :]
    if rest.startswith("conv_s0.") or rest.startswith("conv_s1."):
        return "mask_decoder." + rest

    if rest.startswith("transformer."):
        rest = rest.replace(".mlp.layers.0.", ".mlp.fc1.")
        rest = rest.replace(".mlp.layers.1.", ".mlp.fc2.")
        return "mask_decoder." + rest

    if rest.startswith("output_hypernetworks_mlps.") or rest.startswith(
        "iou_prediction_head."
    ) or rest.startswith("pred_obj_score_head."):
        rest = rest.replace(".layers.0.", ".net.0.")
        rest = rest.replace(".layers.1.", ".net.3.")
        rest = rest.replace(".layers.2.", ".net.6.")
        return "mask_decoder." + rest

    return "mask_decoder." + rest


def map_memory_encoder_key(official_key: str) -> str | None:
    prefix = "memory_encoder."
    if not official_key.startswith(prefix):
        return None
    rest = official_key[len(prefix) :]
    if rest.startswith("position_encoding."):
        return None
    return "memory_encoder." + rest


def map_memory_attention_key(official_key: str) -> str | None:
    if official_key.startswith("memory_attention."):
        return official_key
    return None


def build_mapped_state_dict(
    official_state: dict[str, torch.Tensor],
    ours_state: dict[str, torch.Tensor],
) -> tuple[OrderedDict[str, torch.Tensor], list[str], list[tuple[str, str, tuple[int, ...], tuple[int, ...]]]]:
    mapped: OrderedDict[str, torch.Tensor] = OrderedDict()
    skipped: list[str] = []
    mismatched: list[tuple[str, str, tuple[int, ...], tuple[int, ...]]] = []

    for official_key, tensor in official_state.items():
        ours_key = (
            map_top_level_key(official_key)
            or map_trunk_key(official_key)
            or map_neck_key(official_key)
            or map_prompt_key(official_key)
            or map_mask_decoder_key(official_key)
            or map_memory_encoder_key(official_key)
            or map_memory_attention_key(official_key)
        )
        if ours_key is None:
            skipped.append(official_key)
            continue
        if ours_key not in ours_state:
            skipped.append(official_key)
            continue
        if tuple(ours_state[ours_key].shape) != tuple(tensor.shape):
            mismatched.append(
                (official_key, ours_key, tuple(tensor.shape), tuple(ours_state[ours_key].shape))
            )
            continue
        mapped[ours_key] = tensor

    # 我们保留了 point_embeddings.4 作为“无点占位”的学习显式槽位；官方只有
    # not_a_point_embed。这里复制同一份权重，避免这个槽位保持随机初始化。
    official_not_a = "sam_prompt_encoder.not_a_point_embed.weight"
    ours_placeholder = "prompt_encoder.point_embeddings.4.weight"
    if official_not_a in official_state and ours_placeholder in ours_state:
        mapped[ours_placeholder] = official_state[official_not_a]

    return mapped, skipped, mismatched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(r"..\sam2_impl\checkpoints\sam2.1_hiera_tiny.pt"),
    )
    parser.add_argument("--config", default="sam2.1_hiera_tiny")
    parser.add_argument("--save", type=Path, default=Path("sam2fs_tiny_partial_official.pth"))
    args = parser.parse_args()

    model = SAM2FromScratch(args.config)
    ours_state = model.state_dict()
    official_state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model"]

    mapped, skipped, mismatched = build_mapped_state_dict(official_state, ours_state)
    load_report = model.load_state_dict(mapped, strict=False)

    torch.save(
        {
            "model": model.state_dict(),
            "mapped_keys": list(mapped.keys()),
            "missing_keys": load_report.missing_keys,
            "unexpected_keys": load_report.unexpected_keys,
            "skipped_official_keys": skipped,
            "mismatched": mismatched,
        },
        args.save,
    )

    print(f"official keys: {len(official_state)}")
    print(f"ours keys: {len(ours_state)}")
    print(f"mapped and loaded: {len(mapped)}")
    print(f"skipped official keys: {len(skipped)}")
    print(f"shape mismatches: {len(mismatched)}")
    print(f"model missing after partial load: {len(load_report.missing_keys)}")
    print(f"saved: {args.save}")
    if mismatched[:10]:
        print("\nfirst mismatches:")
        for official_key, ours_key, official_shape, ours_shape in mismatched[:10]:
            print(f"{official_key} -> {ours_key}: {official_shape} != {ours_shape}")


if __name__ == "__main__":
    main()
