import argparse

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    state = torch.load(
        r"..\sam2_impl\checkpoints\sam2.1_hiera_tiny.pt",
        map_location="cpu",
        weights_only=True,
    )["model"]
    count = 0
    for key, value in state.items():
        if args.pattern in key:
            print(key, tuple(value.shape))
            count += 1
            if count >= args.limit:
                break
    print("matched:", count)


if __name__ == "__main__":
    main()
