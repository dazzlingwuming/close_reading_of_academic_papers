import sys
from pathlib import Path

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "sam2_impl"))
sys.path.insert(0, str(ROOT / "sam2_from_scratch"))

from sam2.build_sam import build_sam2
from sam2.utils.transforms import SAM2Transforms
from sam2fs import SAM2FromScratch


IMAGE_PATH = r"C:\Users\lihaodong\Pictures\Saved Pictures\静态壁纸\动漫壁纸\2.jpg"


def stats(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    a = a.float().cpu()
    b = b.float().cpu()
    if a.shape != b.shape:
        print(f"{name}: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
        return
    diff = (a - b).abs()
    denom = b.abs().mean().item() + 1e-6
    cosine = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    print(
        f"{name}: shape={tuple(a.shape)} "
        f"max={diff.max().item():.6f} "
        f"mean={diff.mean().item():.6f} "
        f"rel_mean={diff.mean().item() / denom:.6f} "
        f"cos={cosine:.6f}"
    )


def decoder_trace(decoder, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, high_res_features):
    """按官方 MaskDecoder.predict_masks 的顺序展开，便于逐层对比。

    这里不改变模型逻辑，只把 token 拼接、Transformer、上采样和 hypernetwork
    拆开打印。这样可以判断“权重已经正确加载，但输出不同”到底从哪一步开始。
    """

    output_tokens = torch.cat(
        [decoder.obj_score_token.weight, decoder.iou_token.weight, decoder.mask_tokens.weight],
        dim=0,
    )
    output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
    tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

    src = image_embeddings + dense_prompt_embeddings
    pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
    bsz, channels, height, width = src.shape

    hs, src_tokens = decoder.transformer(src, pos_src, tokens)
    iou_token_out = hs[:, 1, :]
    mask_tokens_out = hs[:, 2 : 2 + decoder.num_mask_tokens, :]

    src_grid = src_tokens.transpose(1, 2).view(bsz, channels, height, width)
    dc1, ln1, act1, dc2, act2 = decoder.output_upscaling
    feat_s0, feat_s1 = high_res_features
    upscaled = act1(ln1(dc1(src_grid) + feat_s1))
    upscaled = act2(dc2(upscaled) + feat_s0)

    hyper_in = torch.stack(
        [decoder.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]) for i in range(decoder.num_mask_tokens)],
        dim=1,
    )
    bsz, channels, height, width = upscaled.shape
    all_masks = (hyper_in @ upscaled.view(bsz, channels, height * width)).view(
        bsz, decoder.num_mask_tokens, height, width
    )
    iou_pred = decoder.iou_prediction_head(iou_token_out)

    return {
        "tokens": tokens,
        "hs": hs,
        "src_tokens": src_tokens,
        "iou_token_out": iou_token_out,
        "mask_tokens_out": mask_tokens_out,
        "src_grid": src_grid,
        "upscaled": upscaled,
        "hyper_in": hyper_in,
        "all_masks": all_masks,
        "iou_pred": iou_pred,
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = Image.open(IMAGE_PATH).convert("RGB")
    transform = SAM2Transforms(1024, mask_threshold=0.0)
    img = transform(image).unsqueeze(0).to(device)

    official = build_sam2(
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        str(ROOT / "sam2_impl" / "checkpoints" / "sam2.1_hiera_tiny.pt"),
        device=device,
    ).eval()

    ours = SAM2FromScratch("sam2.1_hiera_tiny")
    ckpt = torch.load(
        ROOT / "sam2_from_scratch" / "sam2fs_tiny_partial_official.pth",
        map_location="cpu",
        weights_only=False,
    )
    ours.load_state_dict(ckpt["model"], strict=True)
    ours.to(device).eval()

    with torch.inference_mode():
        off_trunk = official.image_encoder.trunk(img)
        our_trunk = ours.image_encoder.backbone(img)
        print("trunk outputs")
        for idx, (a, b) in enumerate(zip(our_trunk, off_trunk)):
            stats(f"trunk[{idx}]", a, b)

        off_fpn = official.image_encoder(img)["backbone_fpn"]
        our_fpn = ours.image_encoder(img)
        print("\nfpn outputs")
        for idx, (a, b) in enumerate(zip(our_fpn, off_fpn)):
            stats(f"fpn[{idx}]", a, b)

        width, height = image.size
        point_coords = torch.tensor(
            [[[int(width * 0.50), int(height * 0.54)], [int(width * 0.50), int(height * 0.30)]]],
            dtype=torch.float32,
            device=device,
        )
        point_labels = torch.tensor([[1, 1]], dtype=torch.int64, device=device)
        box = torch.tensor(
            [[int(width * 0.38), int(height * 0.16), int(width * 0.61), int(height * 0.96)]],
            dtype=torch.float32,
            device=device,
        )
        # 官方 predictor 的坐标变换：原图绝对坐标 -> 归一化 -> 乘 1024。
        point_coords = transform.transform_coords(point_coords, normalize=True, orig_hw=(height, width))
        box = transform.transform_boxes(box, normalize=True, orig_hw=(height, width))

        off_sparse, off_dense = official.sam_prompt_encoder(
            points=(point_coords, point_labels),
            boxes=box,
            masks=None,
        )
        our_sparse, our_dense = ours.prompt_encoder(
            points=(point_coords, point_labels),
            boxes=box,
            masks=None,
        )
        print("\nprompt outputs")
        stats("sparse_prompt", our_sparse, off_sparse)
        stats("dense_prompt", our_dense, off_dense)
        stats("dense_pe", ours.prompt_encoder.get_dense_pe(device), official.sam_prompt_encoder.get_dense_pe())

        off_high_res = [
            official.sam_mask_decoder.conv_s0(off_fpn[0]),
            official.sam_mask_decoder.conv_s1(off_fpn[1]),
        ]
        our_high_res = [
            ours.mask_decoder.conv_s0(our_fpn[0]),
            ours.mask_decoder.conv_s1(our_fpn[1]),
        ]
        print("\nhigh-res feature projections")
        stats("high_res[0]", our_high_res[0], off_high_res[0])
        stats("high_res[1]", our_high_res[1], off_high_res[1])

        off_masks, off_iou, _, _ = official.sam_mask_decoder(
            image_embeddings=off_fpn[2],
            image_pe=official.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=off_sparse,
            dense_prompt_embeddings=off_dense,
            multimask_output=True,
            repeat_image=False,
            high_res_features=off_high_res,
        )
        our_masks, our_iou = ours.mask_decoder(
            image_embeddings=our_fpn[2],
            image_pe=ours.prompt_encoder.get_dense_pe(device),
            sparse_prompt_embeddings=our_sparse,
            dense_prompt_embeddings=our_dense,
            multimask_output=True,
            repeat_image=False,
            high_res_features=our_high_res,
        )
        print("\nmask decoder outputs")
        stats("masks", our_masks, off_masks)
        stats("iou", our_iou, off_iou)

        print("\nmask decoder trace")
        off_trace = decoder_trace(
            official.sam_mask_decoder,
            off_fpn[2],
            official.sam_prompt_encoder.get_dense_pe(),
            off_sparse,
            off_dense,
            off_high_res,
        )
        our_trace = decoder_trace(
            ours.mask_decoder,
            our_fpn[2],
            ours.prompt_encoder.get_dense_pe(device),
            our_sparse,
            our_dense,
            our_high_res,
        )
        for key in off_trace:
            stats(key, our_trace[key], off_trace[key])


if __name__ == "__main__":
    main()
