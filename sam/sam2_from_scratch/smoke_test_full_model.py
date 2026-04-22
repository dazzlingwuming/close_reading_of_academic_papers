import torch

from sam2fs import SAM2FromScratch


def main() -> None:
    # 完整链路测试：图像 -> 编码器 -> 提示编码器 -> mask decoder -> memory。
    # 这里使用随机图像和人工点/框，只检查结构、维度和数据流是否正确。
    model = SAM2FromScratch("sam2.1_hiera_tiny").eval()
    image = torch.randn(1, 3, 1024, 1024)
    points = torch.tensor([[[512.0, 512.0]]])
    labels = torch.tensor([[1]])
    boxes = torch.tensor([[128.0, 128.0, 900.0, 900.0]])

    with torch.inference_mode():
        out = model(image, points=points, labels=labels, boxes=boxes)

    print("low_res_masks:", tuple(out["low_res_masks"].shape))
    print("iou_predictions:", tuple(out["iou_predictions"].shape))
    print("memory:", tuple(out["memory"].shape))


if __name__ == "__main__":
    main()
