import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Normalize, Resize, ToTensor

from .model import SAM2FromScratch


class ResizeLongestSide:
    """
    官方 SAM2 图像 predictor 会直接把输入 resize 到 resolution x resolution，
    而不是最长边缩放再 padding。坐标则先除以原图宽高归一化，再乘 resolution。
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length
        self.to_tensor = ToTensor()
        self.transforms = torch.nn.Sequential(
            Resize((target_length, target_length)),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        )

    def get_preprocess_shape(self, old_h: int, old_w: int) -> tuple[int, int]:
        return self.target_length, self.target_length

    def apply_image(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        old_h, old_w = image.shape[:2]
        new_h, new_w = self.get_preprocess_shape(old_h, old_w)
        return image, (new_h, new_w)

    def image_to_tensor(self, image: np.ndarray | Image.Image, device: torch.device) -> torch.Tensor:
        """执行官方 SAM2Transforms 的 ToTensor -> Resize -> Normalize。"""

        tensor = self.to_tensor(image)
        tensor = self.transforms(tensor)
        return tensor.unsqueeze(0).to(device)

    def apply_coords(self, coords: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(old_h, old_w)
        coords = coords.copy().astype(np.float32)
        coords[..., 0] = coords[..., 0] / old_w * new_w
        coords[..., 1] = coords[..., 1] / old_h * new_h
        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
        boxes = boxes.reshape(-1, 2, 2)
        boxes = self.apply_coords(boxes, original_size)
        return boxes.reshape(-1, 4)


class SAM2ImagePredictorFS:
    """学习版图像 predictor。

    负责图像预处理、点/框坐标缩放和 mask 放回原图尺寸。模型未训练时输出没有
    语义质量，但接口和数据流已经完整。
    """

    def __init__(self, model: SAM2FromScratch) -> None:
        self.model = model.eval()
        self.transform = ResizeLongestSide(model.config.image_size)
        self.original_size: tuple[int, int] | None = None
        self.input_size: tuple[int, int] | None = None
        self.image_tensor: torch.Tensor | None = None
        self.features: list[torch.Tensor] | None = None

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def set_image(self, image: np.ndarray | Image.Image) -> None:
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image 必须是 RGB 图像")

        self.original_size = image.shape[:2]
        _, self.input_size = self.transform.apply_image(image)
        self.image_tensor = self.transform.image_to_tensor(image, self.device)
        with torch.inference_mode():
            self.features = self.model.encode_image(self.image_tensor)
            # 官方 SAM2.1 图像 predictor 会在没有视频记忆时给最低分辨率图像特征
            # 加 no_mem_embed；这对应论文里“无记忆状态”的占位 token。
            no_mem = self.model.no_mem_embed.transpose(0, 1).view(1, -1, 1, 1)
            self.features[2] = self.features[2] + no_mem

    def predict(
        self,
        point_coords: np.ndarray | None = None,
        point_labels: np.ndarray | None = None,
        box: np.ndarray | None = None,
        multimask_output: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.image_tensor is None or self.original_size is None or self.features is None:
            raise RuntimeError("请先调用 set_image")

        points_t = labels_t = boxes_t = None
        if point_coords is not None:
            coords = self.transform.apply_coords(point_coords[None, :, :], self.original_size)
            points_t = torch.as_tensor(coords, device=self.device, dtype=torch.float32)
            labels_t = torch.as_tensor(point_labels[None, :], device=self.device, dtype=torch.long)
        if box is not None:
            boxes = self.transform.apply_boxes(np.asarray(box, dtype=np.float32)[None, :], self.original_size)
            boxes_t = torch.as_tensor(boxes, device=self.device, dtype=torch.float32)

        # 官方 predictor 会把 box 的左上/右下两个角转换成 label=2/3 的 prompt
        # token，并放在普通点前面；这样 decoder 看到的 token 顺序完全一致。
        if boxes_t is not None:
            box_coords = boxes_t.reshape(-1, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.long, device=self.device)
            box_labels = box_labels.repeat(boxes_t.size(0), 1)
            if points_t is not None and labels_t is not None:
                points_t = torch.cat([box_coords, points_t], dim=1)
                labels_t = torch.cat([box_labels, labels_t], dim=1)
            else:
                points_t = box_coords
                labels_t = box_labels

        with torch.inference_mode():
            high_res_features = [
                self.model.mask_decoder.conv_s0(self.features[0]),
                self.model.mask_decoder.conv_s1(self.features[1]),
            ]
            image_embeddings = self.model.memory_attention(self.features[2], memories=None)
            sparse_prompt, dense_prompt = self.model.prompt_encoder(
                points=(points_t, labels_t) if points_t is not None else None,
                boxes=None,
                masks=None,
            )
            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.model.prompt_encoder.get_dense_pe(self.device),
                sparse_prompt_embeddings=sparse_prompt,
                dense_prompt_embeddings=dense_prompt,
                multimask_output=multimask_output,
                repeat_image=False,
                high_res_features=high_res_features,
            )

        masks = F.interpolate(
            low_res_masks.float(),
            size=self.original_size,
            mode="bilinear",
            align_corners=False,
        )
        masks = masks.float().cpu().numpy()[0]
        scores = iou_predictions.float().cpu().numpy()[0]
        return masks, scores
