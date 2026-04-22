import torch


KEYS = [
    "mask_downsample.weight",
    "mask_downsample.bias",
    "obj_ptr_proj.layers.0.weight",
    "obj_ptr_proj.layers.0.bias",
    "obj_ptr_proj.layers.1.weight",
    "obj_ptr_proj.layers.1.bias",
    "obj_ptr_proj.layers.2.weight",
    "obj_ptr_proj.layers.2.bias",
    "obj_ptr_tpos_proj.weight",
    "obj_ptr_tpos_proj.bias",
]


state = torch.load(
    r"..\sam2_impl\checkpoints\sam2.1_hiera_tiny.pt",
    map_location="cpu",
    weights_only=True,
)["model"]

for key in KEYS:
    print(key, tuple(state[key].shape))
