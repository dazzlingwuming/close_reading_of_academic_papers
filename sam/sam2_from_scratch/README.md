# SAM2.1 复现代码使用说明

本目录提供 SAM2.1 图像分割路径的 PyTorch 实现，支持加载 SAM2.1 tiny 官方 checkpoint，并提供单图 prompt 分割测试脚本。

## 目录结构

```text
sam2_from_scratch/
├── sam2fs/
│   ├── config.py
│   ├── layers.py
│   ├── hiera.py
│   ├── image_encoder.py
│   ├── position_encoding.py
│   ├── prompt_encoder.py
│   ├── two_way_transformer.py
│   ├── mask_decoder.py
│   ├── memory_encoder.py
│   ├── memory_attention.py
│   ├── model.py
│   └── predictor.py
├── load_official_sam2_weights.py
├── smoke_test_encoder.py
├── smoke_test_full_model.py
└── test_image_2_from_scratch.py
```

## 环境要求

需要安装：

```bash
pip install torch torchvision pillow numpy
```

如果使用 CUDA，请安装与本机 CUDA 版本匹配的 PyTorch。

## 权重准备

默认使用 SAM2.1 Hiera-Tiny checkpoint：

```text
sam/sam2_impl/checkpoints/sam2.1_hiera_tiny.pt
```

如果 checkpoint 在其他位置，可以通过参数指定：

```bash
python load_official_sam2_weights.py --checkpoint "D:/path/to/sam2.1_hiera_tiny.pt"
```

生成的复现结构权重文件为：

```text
sam2fs_tiny_partial_official.pth
```

该文件约 156MB，超过 GitHub 普通文件大小限制，建议不要提交到仓库。需要运行时在本地用映射脚本重新生成。

预期加载输出：

```text
official keys: 471
ours keys: 472
mapped and loaded: 472
skipped official keys: 0
shape mismatches: 0
model missing after partial load: 0
saved: sam2fs_tiny_partial_official.pth
```

`ours keys` 多出的 1 个参数是 `prompt_encoder.point_embeddings.4.weight`，用于和 prompt encoder 中“无点占位”表示对齐，权重来自官方的 `not_a_point_embed.weight`。

## 快速测试

进入目录：

```bash
cd sam/sam2_from_scratch
```

检查 image encoder：

```bash
python smoke_test_encoder.py
```

检查完整前向：

```bash
python smoke_test_full_model.py
```

加载官方权重：

```bash
python load_official_sam2_weights.py
```

运行单图分割测试：

```bash
python test_image_2_from_scratch.py
```

测试脚本会输出：

```text
result_image_2_fs_overlay.png
result_image_2_fs_mask.png
result_image_2_fs_cutout.png
```

## 在自己的图片上使用

可以直接使用 `SAM2FromScratch` 和 `SAM2ImagePredictorFS`：

```python
import numpy as np
import torch
from PIL import Image

from sam2fs import SAM2FromScratch, SAM2ImagePredictorFS

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SAM2FromScratch("sam2.1_hiera_tiny")
checkpoint = torch.load("sam2fs_tiny_partial_official.pth", map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint["model"], strict=True)
model.to(device).eval()

image = np.array(Image.open("your_image.jpg").convert("RGB"))
predictor = SAM2ImagePredictorFS(model)

with torch.inference_mode():
    if device == "cuda":
        autocast = torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        autocast = torch.autocast("cpu", enabled=False)
    with autocast:
        predictor.set_image(image)
        masks, scores = predictor.predict(
            point_coords=np.array([[500, 500]], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
            box=None,
            multimask_output=True,
        )

best_idx = int(np.argmax(scores))
best_mask = masks[best_idx] > 0.0
```

坐标格式：

- `point_coords`：`N x 2`，格式为 `(x, y)`，原图像素坐标。
- `point_labels`：`N`，`1` 表示前景点，`0` 表示背景点。
- `box`：`[x1, y1, x2, y2]`，原图像素坐标。
- `masks`：返回 mask logits，默认按 `> 0.0` 转为二值 mask。

## 论文模块与代码对应

| 论文模块 | 代码位置 | 主要对应关系 |
|---|---|---|
| Image Encoder | `sam2fs/hiera.py` | Hiera 分层 backbone：patch embedding、多尺度 attention、window attention、stage 输出 |
| Image Encoder Neck | `sam2fs/image_encoder.py` | FPN neck：把 Hiera 多尺度输出投影到统一的 256 通道 |
| Prompt Encoder | `sam2fs/prompt_encoder.py` | 点、框、mask prompt 的编码；点/框是 sparse token，mask 是 dense feature |
| Prompt Position Encoding | `sam2fs/position_encoding.py` | 随机傅里叶位置编码，对应 SAM/SAM2 prompt 坐标编码 |
| Two-Way Transformer | `sam2fs/two_way_transformer.py` | prompt token 与 image token 的双向注意力：prompt 读图像，图像再读 prompt |
| Mask Decoder | `sam2fs/mask_decoder.py` | IoU token、mask token、object score token、hypernetwork mask 生成、高分辨率特征融合 |
| Memory Attention | `sam2fs/memory_attention.py` | 当前帧特征通过 cross-attention 读取历史 memory |
| Memory Encoder | `sam2fs/memory_encoder.py` | mask 下采样后与图像特征融合，生成 spatial memory |
| Model Wrapper | `sam2fs/model.py` | 串联 image encoder、memory attention、prompt encoder、mask decoder、memory encoder |
| Image Predictor | `sam2fs/predictor.py` | 图像预处理、prompt 坐标变换、box prompt 合并、mask 后处理 |

## 与官方结果的对齐结果

当前同条件单图测试中，复现结构和 SAM2.1 tiny 官方实现的输出结果：

```text
mask IoU between two outputs: 0.9994
```

## 不建议提交的文件

以下文件建议本地生成，不提交到 GitHub：

```text
sam2fs_tiny_partial_official.pth
result_image_2_fs_overlay.png
result_image_2_fs_mask.png
result_image_2_fs_cutout.png
__pycache__/
```
