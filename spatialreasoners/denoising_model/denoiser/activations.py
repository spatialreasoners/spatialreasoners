from functools import partial
from typing import Literal

from torch import nn

Activation = Literal[    
    "gelu",
    "gelu_approx",
    "leaky_relu",
    "linear",
    "mish",
    "relu",
    "relu6",
    "sigmoid",
    "silu",
    "swish",
    "tanh"
]


ACTIVATION = {
    "gelu": nn.GELU,
    "gelu_approx": partial(nn.GELU, approximate="tanh"),
    "leaky_relu": nn.LeakyReLU,
    "linear": nn.Identity,
    "mish": nn.Mish,
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "tanh": nn.Tanh,
}


def get_activation(activation_string):
    if activation_string in ACTIVATION:
        return ACTIVATION[activation_string]()
    else:
        raise KeyError(f"function {activation_string} not found in ACTIVATION mapping {list(ACTIVATION.keys())}")
