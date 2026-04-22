import numpy as np
import torch
from PIL import Image

from sam2fs import SAM2FromScratch, SAM2ImagePredictorFS


IMAGE_PATH = r"C:\Users\lihaodong\Pictures\Saved Pictures\静态壁纸\动漫壁纸\2.jpg"
WEIGHT_PATH = "sam2fs_tiny_partial_official.pth"

OUT_OVERLAY = "result_image_2_fs_overlay.png"
OUT_MASK = "result_image_2_fs_mask.png"
OUT_CUTOUT = "result_image_2_fs_cutout.png"


def blend_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.copy().astype(np.float32)
    color = np.array([30, 220, 120], dtype=np.float32)
    alpha = 0.45
    overlay[mask] = overlay[mask] * (1.0 - alpha) + color * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    image = np.array(Image.open(IMAGE_PATH).convert("RGB"))
    height, width = image.shape[:2]
    print(f"image: {width}x{height}")

    model = SAM2FromScratch("sam2.1_hiera_tiny")
    checkpoint = torch.load(WEIGHT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    predictor = SAM2ImagePredictorFS(model)

    # 这个图的主体人物在中间偏左。坐标用比例写，换分辨率也能工作。
    point_coords = np.array(
        [
            [int(width * 0.50), int(height * 0.54)],  # 上衣/身体
            [int(width * 0.50), int(height * 0.30)],  # 头发
        ],
        dtype=np.float32,
    )
    point_labels = np.array([1, 1], dtype=np.int32)
    box = np.array(
        [
            int(width * 0.38),
            int(height * 0.16),
            int(width * 0.61),
            int(height * 0.96),
        ],
        dtype=np.float32,
    )

    with torch.inference_mode():
        if device == "cuda":
            autocast = torch.autocast("cuda", dtype=torch.bfloat16)
        else:
            autocast = torch.autocast("cpu", enabled=False)
        with autocast:
            predictor.set_image(image)
            masks, scores = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=True,
            )
    best_idx = int(np.argmax(scores))
    # SAM2 predictor 返回的是 mask logits，官方默认阈值是 0。
    best_mask = masks[best_idx] > 0.0

    print(f"scores: {scores}")
    print(f"best mask index: {best_idx}, score: {scores[best_idx]:.4f}")

    Image.fromarray(blend_mask(image, best_mask)).save(OUT_OVERLAY)
    Image.fromarray((best_mask.astype(np.uint8) * 255), mode="L").save(OUT_MASK)
    cutout_rgba = np.dstack([image, best_mask.astype(np.uint8) * 255])
    Image.fromarray(cutout_rgba, mode="RGBA").save(OUT_CUTOUT)

    print(f"saved overlay: {OUT_OVERLAY}")
    print(f"saved mask: {OUT_MASK}")
    print(f"saved cutout: {OUT_CUTOUT}")


if __name__ == "__main__":
    main()
