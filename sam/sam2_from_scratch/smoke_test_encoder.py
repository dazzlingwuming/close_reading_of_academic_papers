import torch

from sam2fs import SAM2ImageEncoder


def main() -> None:
    # 这里只用 256x256 随机图像做结构测试，目的是快速验证 Hiera + FPN 的
    # 多尺度输出形状。真实模型默认输入仍然是 1024x1024。
    model = SAM2ImageEncoder("sam2.1_hiera_tiny").eval()
    image = torch.randn(1, 3, 256, 256)

    with torch.inference_mode():
        features = model(image)

    for idx, feature in enumerate(features):
        print(f"feature[{idx}]: {tuple(feature.shape)}")


if __name__ == "__main__":
    main()
