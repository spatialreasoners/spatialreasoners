from typing import Literal

from torch import nn

Norm = Literal["batch", "group", "instance"]


NORM = {
    "batch": nn.BatchNorm2d,
    "group": lambda d: nn.GroupNorm(32, d),
    "instance": nn.InstanceNorm2d,
}


def get_norm(norm_layer_string: str, dim: int) -> nn.Module:
    if norm_layer_string in NORM:
        return NORM[norm_layer_string](dim)
    else:
        raise KeyError(f"function {norm_layer_string} not found in NORM mapping {list(NORM.keys())}")
