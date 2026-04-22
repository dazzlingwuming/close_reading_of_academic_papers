import argparse

import torch

from sam2fs import SAM2FromScratch


def print_items(items, limit: int) -> None:
    for idx, (key, value) in enumerate(items):
        print(key, tuple(value.shape))
        if idx + 1 >= limit:
            break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=120)
    args = parser.parse_args()

    official = torch.load(
        r"..\sam2_impl\checkpoints\sam2.1_hiera_tiny.pt",
        map_location="cpu",
        weights_only=True,
    )["model"]
    ours = SAM2FromScratch("sam2.1_hiera_tiny").state_dict()

    print("official:", len(official))
    print_items(official.items(), args.limit)
    print("\nours:", len(ours))
    print_items(ours.items(), args.limit)


if __name__ == "__main__":
    main()
