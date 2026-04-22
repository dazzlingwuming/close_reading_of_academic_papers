import torch
from torch import nn

from .layers import LayerNorm2d, MLP
from .two_way_transformer import TwoWayTransformer


class MaskDecoder(nn.Module):
    """由图像特征和提示特征预测 mask。

    结构对应 SAM/SAM2 的 decoder：IoU token 预测 mask 质量，多个 mask token
    通过小型 hypernetwork 生成动态卷积权重，和上采样后的图像特征相乘得到
    多个候选 mask。
    """

    def __init__(
        self,
        embed_dim: int = 256,
        transformer_depth: int = 2,
        transformer_heads: int = 8,
        num_mask_tokens: int = 4,
    ) -> None:
        super().__init__()
        self.num_mask_tokens = num_mask_tokens
        self.iou_token = nn.Embedding(1, embed_dim)
        self.mask_tokens = nn.Embedding(num_mask_tokens, embed_dim)
        # SAM2 额外预测 object score，用于视频场景判断目标是否出现。
        # 当前学习版前向暂未使用它，但保留模块便于加载官方权重。
        self.obj_score_token = nn.Embedding(1, embed_dim)
        self.transformer = TwoWayTransformer(
            depth=transformer_depth,
            dim=embed_dim,
            num_heads=transformer_heads,
            mlp_dim=embed_dim * 8,
        )
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(embed_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=2, stride=2),
            nn.GELU(),
        )
        # 官方 SAM2 decoder 会融合高分辨率 FPN 特征；当前学习版还没把高分辨率
        # 特征接入 forward，但先定义这两个投影层以保持参数结构。
        self.conv_s0 = nn.Conv2d(embed_dim, embed_dim // 8, kernel_size=1)
        self.conv_s1 = nn.Conv2d(embed_dim, embed_dim // 4, kernel_size=1)
        # 注意：这里必须是 ReLU。Hiera backbone 的 MLP 常用 GELU，但官方 SAM2
        # mask decoder 的 hypernetwork MLP 和 IoU/object score MLP 都沿用 SAM
        # decoder 的 ReLU 头。权重形状相同但激活函数不同，会导致输出明显偏移。
        self.output_hypernetworks_mlps = nn.ModuleList(
            MLP(embed_dim, embed_dim, embed_dim // 8, num_layers=3, activation=nn.ReLU, dropout=0.0)
            for _ in range(num_mask_tokens)
        )
        self.iou_prediction_head = MLP(
            embed_dim,
            256,
            num_mask_tokens,
            num_layers=3,
            activation=nn.ReLU,
            dropout=0.0,
            sigmoid_output=True,
        )
        self.iou_prediction_use_sigmoid = False
        self.pred_obj_score_head = MLP(
            embed_dim,
            256,
            1,
            num_layers=3,
            activation=nn.ReLU,
            dropout=0.0,
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool = True,
        repeat_image: bool = False,
        high_res_features: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        masks, iou_pred, _, _ = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
        )

        if multimask_output:
            return masks[:, 1:, :, :], iou_pred[:, 1:]
        return masks[:, :1, :, :], iou_pred[:, :1]

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        repeat_image: bool,
        high_res_features: list[torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # SAM2 图像/视频路径默认会带 object score token，因此 token 顺序是：
        # obj_score, iou, mask_0, mask_1, mask_2, mask_3, prompt...
        output_tokens = torch.cat(
            [self.obj_score_token.weight, self.iou_token.weight, self.mask_tokens.weight],
            dim=0,
        )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0),
            -1,
            -1,
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        bsz, channels, height, width = src.shape

        hs, src_tokens = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 1, :]
        mask_tokens_out = hs[:, 2 : 2 + self.num_mask_tokens, :]

        src = src_tokens.transpose(1, 2).view(bsz, channels, height, width)
        if high_res_features is None:
            upscaled = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled = act1(ln1(dc1(src) + feat_s1))
            upscaled = act2(dc2(upscaled) + feat_s0)

        hyper_in = torch.stack(
            [
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
                for i in range(self.num_mask_tokens)
            ],
            dim=1,
        )
        bsz, channels, height, width = upscaled.shape
        masks = (hyper_in @ upscaled.view(bsz, channels, height * width)).view(
            bsz,
            self.num_mask_tokens,
            height,
            width,
        )
        iou_pred = self.iou_prediction_head(iou_token_out)
        if self.iou_prediction_use_sigmoid:
            iou_pred = iou_pred.sigmoid()
        object_score_logits = self.pred_obj_score_head(hs[:, 0, :])

        return masks, iou_pred, mask_tokens_out, object_score_logits
