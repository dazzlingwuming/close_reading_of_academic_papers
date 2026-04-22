import torch
from torch import nn

from .layers import LayerNorm2d
from .position_encoding import PositionEmbeddingRandom


class PromptEncoder(nn.Module):
    """把用户提示编码成 sparse token 和 dense feature。

    论文中 SAM2 延续了 SAM 的 promptable segmentation 设计：点、框属于稀疏
    提示，mask 属于稠密提示。mask decoder 会同时读取这两类提示。
    """

    def __init__(
        self,
        embed_dim: int = 256,
        image_embedding_size: tuple[int, int] = (64, 64),
        input_image_size: tuple[int, int] = (1024, 1024),
        mask_in_chans: int = 16,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        # 官方语义：0=负点，1=正点，2/3=框的两个角点。
        # 额外的 point_embeddings.4 只作为学习版兼容槽位，不参与 forward。
        self.point_embeddings = nn.ModuleList(nn.Embedding(1, embed_dim) for _ in range(5))
        self.not_a_point_embed = nn.Embedding(1, embed_dim)
        self.mask_input_size = (
            4 * image_embedding_size[0],
            4 * image_embedding_size[1],
        )

        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            nn.GELU(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            nn.GELU(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self, device: torch.device | None = None) -> torch.Tensor:
        pe = self.pe_layer(self.image_embedding_size)
        pe = pe.unsqueeze(0)
        return pe if device is None else pe.to(device)

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        # 官方实现会把坐标从像素左上角移动到像素中心。
        points = points + 0.5
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)

        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding = torch.where(
            (labels == -1).unsqueeze(-1),
            torch.zeros_like(point_embedding) + self.not_a_point_embed.weight,
            point_embedding,
        )
        for label_value in range(4):
            point_embedding = torch.where(
                (labels == label_value).unsqueeze(-1),
                point_embedding + self.point_embeddings[label_value].weight,
                point_embedding,
            )
        return point_embedding

    def _embed_boxes(
        self,
        boxes: torch.Tensor,
    ) -> torch.Tensor:
        boxes = boxes + 0.5
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        return self.mask_downscaling(masks)

    def _get_batch_size(
        self,
        points: tuple[torch.Tensor, torch.Tensor] | None,
        boxes: torch.Tensor | None,
        masks: torch.Tensor | None,
    ) -> int:
        if points is not None:
            return points[0].shape[0]
        if boxes is not None:
            return boxes.shape[0]
        if masks is not None:
            return masks.shape[0]
        return 1

    def forward(
        self,
        points: tuple[torch.Tensor, torch.Tensor] | None = None,
        boxes: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = self._get_batch_size(points, boxes, masks)
        device = self.point_embeddings[0].weight.device
        sparse_embeddings = torch.empty((batch_size, 0, self.embed_dim), device=device)

        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                batch_size,
                -1,
                self.image_embedding_size[0],
                self.image_embedding_size[1],
            )

        return sparse_embeddings, dense_embeddings
