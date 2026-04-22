import torch
from torch import nn

from .config import SAM2Config, get_sam2_config
from .image_encoder import SAM2ImageEncoder
from .mask_decoder import MaskDecoder
from .memory_attention import MemoryAttention
from .memory_encoder import MemoryEncoder
from .prompt_encoder import PromptEncoder


class LinearStack(nn.Module):
    """只为兼容官方 checkpoint 命名的线性层栈。"""

    def __init__(self, dims: list[int]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Linear(dims[idx], dims[idx + 1]) for idx in range(len(dims) - 1)
        )


class SAM2FromScratch(nn.Module):
    """学习版 SAM2 主模型。

    代码结构按论文主线组织：
    image encoder -> memory attention -> prompt encoder -> mask decoder -> memory encoder。
    图像任务可以忽略 memory；视频任务可把历史 memory 传进来。
    """

    def __init__(self, config: SAM2Config | str = "sam2.1_hiera_tiny") -> None:
        super().__init__()
        if isinstance(config, str):
            config = get_sam2_config(config)
        self.config = config
        self.image_encoder = SAM2ImageEncoder(config)
        # SAM mask decoder 的主图像 embedding 是 stride-16，即 1024 -> 64。
        # stride-4/stride-8 特征作为 high-res features 辅助恢复边界。
        embedding_size = (config.image_size // 16, config.image_size // 16)
        self.prompt_encoder = PromptEncoder(
            embed_dim=config.neck_dim,
            image_embedding_size=embedding_size,
            input_image_size=(config.image_size, config.image_size),
            mask_in_chans=config.mask_in_chans,
        )
        self.memory_attention = MemoryAttention(
            dim=config.neck_dim,
            depth=config.memory_attention_layers,
            num_heads=config.transformer_heads,
        )
        self.mask_decoder = MaskDecoder(
            embed_dim=config.neck_dim,
            transformer_depth=config.transformer_depth,
            transformer_heads=config.transformer_heads,
            num_mask_tokens=config.num_mask_tokens,
        )
        self.memory_encoder = MemoryEncoder(config.neck_dim, memory_dim=64)
        # SAM2 视频推理中的无记忆/无目标/时间位置编码参数。当前学习版先保留
        # 参数结构，便于加载官方 checkpoint；完整视频状态机后续再接入。
        self.maskmem_tpos_enc = nn.Parameter(torch.zeros(7, 1, 1, 64))
        self.no_mem_embed = nn.Parameter(torch.zeros(1, 1, config.neck_dim))
        self.no_mem_pos_enc = nn.Parameter(torch.zeros(1, 1, config.neck_dim))
        self.no_obj_ptr = nn.Parameter(torch.zeros(1, config.neck_dim))
        self.no_obj_embed_spatial = nn.Parameter(torch.zeros(1, 64))
        self.mask_downsample = nn.Conv2d(1, 1, kernel_size=4, stride=4)
        self.obj_ptr_proj = LinearStack([config.neck_dim, config.neck_dim, config.neck_dim, config.neck_dim])
        self.obj_ptr_tpos_proj = nn.Linear(config.neck_dim, 64)

    def encode_image(self, image: torch.Tensor) -> list[torch.Tensor]:
        return self.image_encoder(image)

    def forward(
        self,
        image: torch.Tensor,
        points: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        boxes: torch.Tensor | None = None,
        mask_inputs: torch.Tensor | None = None,
        memories: list[torch.Tensor] | None = None,
        multimask_output: bool = True,
    ) -> dict[str, torch.Tensor]:
        features = self.encode_image(image)
        high_res_features = [
            self.mask_decoder.conv_s0(features[0]),
            self.mask_decoder.conv_s1(features[1]),
        ]
        image_embeddings = features[2]
        image_embeddings = self.memory_attention(image_embeddings, memories)

        point_tuple = None
        if points is not None:
            if labels is None:
                raise ValueError("传入 points 时必须同时传入 labels")
            point_tuple = (points, labels)

        sparse_prompt, dense_prompt = self.prompt_encoder(
            points=point_tuple,
            boxes=boxes,
            masks=mask_inputs,
        )
        image_pe = self.prompt_encoder.get_dense_pe(image.device)
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt,
            dense_prompt_embeddings=dense_prompt,
            multimask_output=multimask_output,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        memory = self.memory_encoder(image_embeddings, low_res_masks[:, :1])
        return {
            "low_res_masks": low_res_masks,
            "iou_predictions": iou_predictions,
            "memory": memory,
            "image_features": image_embeddings,
        }
