from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
FS_DIR = ROOT / "sam2_from_scratch"
OFFICIAL_DIR = ROOT / "sam2_impl"

FS_OVERLAY = FS_DIR / "result_image_2_fs_overlay.png"
OFFICIAL_OVERLAY = OFFICIAL_DIR / "result_image_2_official_overlay.png"
FS_MASK = FS_DIR / "result_image_2_fs_mask.png"
OFFICIAL_MASK = OFFICIAL_DIR / "result_image_2_official_mask.png"

OUT_COMPARE = ROOT / "image_2_compare_fs_vs_official.png"
OUT_DIFF = ROOT / "image_2_mask_diff.png"


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def add_label(image: Image.Image, label: str) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 42)
    except OSError:
        font = ImageFont.load_default()
    pad = 24
    draw.rectangle([0, 0, image.width, 88], fill=(0, 0, 0))
    draw.text((pad, pad), label, fill=(255, 255, 255), font=font)
    return image


def main() -> None:
    fs = load_rgb(FS_OVERLAY)
    official = load_rgb(OFFICIAL_OVERLAY)
    if fs.size != official.size:
        official = official.resize(fs.size, Image.BILINEAR)

    fs_labeled = add_label(fs, "sam2_from_scratch: loaded official weights, not numerically aligned")
    official_labeled = add_label(official, "sam2_impl official: original SAM2.1 tiny weights")

    compare = Image.new("RGB", (fs.width * 2, fs.height), (255, 255, 255))
    compare.paste(fs_labeled, (0, 0))
    compare.paste(official_labeled, (fs.width, 0))
    compare.save(OUT_COMPARE)

    fs_mask = np.array(Image.open(FS_MASK).convert("L")) > 127
    official_mask = np.array(Image.open(OFFICIAL_MASK).convert("L")) > 127
    if fs_mask.shape != official_mask.shape:
        official_mask_img = Image.fromarray((official_mask.astype(np.uint8) * 255), mode="L")
        official_mask = np.array(official_mask_img.resize(fs_mask.shape[::-1], Image.NEAREST)) > 127

    diff = np.zeros((*fs_mask.shape, 3), dtype=np.uint8)
    diff[fs_mask & official_mask] = [80, 220, 120]      # 两者都选中：绿色
    diff[fs_mask & ~official_mask] = [255, 80, 80]      # 只有手写版选中：红色
    diff[~fs_mask & official_mask] = [80, 140, 255]     # 只有官方版选中：蓝色
    Image.fromarray(diff).save(OUT_DIFF)

    intersection = np.logical_and(fs_mask, official_mask).sum()
    union = np.logical_or(fs_mask, official_mask).sum()
    iou = intersection / union if union else 0.0
    print(f"saved compare: {OUT_COMPARE}")
    print(f"saved diff: {OUT_DIFF}")
    print(f"mask IoU between two outputs: {iou:.4f}")
    print("diff legend: green=both, red=from_scratch_only, blue=official_only")


if __name__ == "__main__":
    main()
