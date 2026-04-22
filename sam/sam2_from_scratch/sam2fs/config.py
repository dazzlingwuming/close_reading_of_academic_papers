from dataclasses import dataclass


@dataclass(frozen=True)
class SAM2Config:
    """SAM2.1 不同模型尺寸的结构超参数。

    这些字段对应论文和公开模型配置中的关键结构选择：
    - image_size/patch_size 决定图像 token 网格大小；
    - embed_dim/stage_depths/stage_heads 决定 Hiera backbone 规模；
    - neck_dim 是进入 prompt/mask/memory 模块的统一通道数；
    - memory_* 控制视频流式记忆模块。
    """

    name: str
    image_size: int
    patch_size: int
    embed_dim: int
    stage_depths: tuple[int, int, int, int]
    stage_heads: tuple[int, int, int, int]
    window_size: int
    window_spec: tuple[int, int, int, int]
    global_att_blocks: tuple[int, ...]
    q_pool: int
    q_stride: tuple[int, int]
    mlp_ratio: float
    drop_path_rate: float
    neck_dim: int
    mask_in_chans: int = 16
    num_mask_tokens: int = 4
    transformer_depth: int = 2
    transformer_heads: int = 8
    memory_dim: int = 256
    memory_attention_layers: int = 4


_CONFIGS = {
    # tiny 用于学习和调试最方便，参数量最小，前向速度最快。
    "sam2.1_hiera_tiny": SAM2Config(
        name="sam2.1_hiera_tiny",
        image_size=1024,
        patch_size=4,
        embed_dim=96,
        stage_depths=(1, 2, 7, 2),
        stage_heads=(1, 2, 4, 8),
        window_size=8,
        window_spec=(8, 4, 14, 7),
        global_att_blocks=(5, 7, 9),
        q_pool=3,
        q_stride=(2, 2),
        mlp_ratio=4.0,
        drop_path_rate=0.1,
        neck_dim=256,
    ),
    # small 加深第三个 stage，表达力比 tiny 更强。
    "sam2.1_hiera_small": SAM2Config(
        name="sam2.1_hiera_small",
        image_size=1024,
        patch_size=4,
        embed_dim=96,
        stage_depths=(1, 2, 11, 2),
        stage_heads=(1, 2, 4, 8),
        window_size=8,
        window_spec=(8, 4, 14, 7),
        global_att_blocks=(7, 10, 13),
        q_pool=3,
        q_stride=(2, 2),
        mlp_ratio=4.0,
        drop_path_rate=0.1,
        neck_dim=256,
    ),
    # base_plus 增大宽度和深度，是质量与成本之间的中间档。
    "sam2.1_hiera_base_plus": SAM2Config(
        name="sam2.1_hiera_base_plus",
        image_size=1024,
        patch_size=4,
        embed_dim=112,
        stage_depths=(2, 3, 16, 3),
        stage_heads=(2, 4, 8, 16),
        window_size=8,
        window_spec=(8, 4, 14, 7),
        global_att_blocks=(12, 16, 20),
        q_pool=3,
        q_stride=(2, 2),
        mlp_ratio=4.0,
        drop_path_rate=0.2,
        neck_dim=256,
    ),
    # large 是公开系列中最大的版本，适合追求最高质量。
    "sam2.1_hiera_large": SAM2Config(
        name="sam2.1_hiera_large",
        image_size=1024,
        patch_size=4,
        embed_dim=144,
        stage_depths=(2, 6, 36, 4),
        stage_heads=(2, 4, 8, 16),
        window_size=8,
        window_spec=(8, 4, 16, 8),
        global_att_blocks=(23, 33, 43),
        q_pool=3,
        q_stride=(2, 2),
        mlp_ratio=4.0,
        drop_path_rate=0.4,
        neck_dim=256,
    ),
}


def get_sam2_config(name: str = "sam2.1_hiera_tiny") -> SAM2Config:
    try:
        return _CONFIGS[name]
    except KeyError as exc:
        known = ", ".join(sorted(_CONFIGS))
        raise ValueError(f"unknown config {name!r}; expected one of: {known}") from exc
