from spatialreasoners.registry import Registry
from .norm_2d import Norm2d, Norm2dCfg


_norm_2d_registry = Registry(Norm2d, Norm2dCfg)


def get_norm_2d(
    cfg: Norm2dCfg, 
    dim: int
) -> Norm2d:
    return _norm_2d_registry.build(cfg, dim)


register_norm_2d = _norm_2d_registry.register


__all__ = [
    "Norm2d", "Norm2dCfg",
    "get_norm_2d",
    "register_norm_2d"
]


# Functionality for usage of PyTorch normalization layers 
# as registered interface implementations
from types import new_class

from jaxtyping import Float
from torch import Tensor
from torch.nn import Module


def _register_pytorch_norm_2d(
    torch_cls: type[Module],
    name: str | None = None,
    cfg_cls: type[Norm2dCfg] | None = None,
    dim_key: str | None = None
) -> type[Norm2d]:
    """
    Post-hoc interface implementations and registration 
    using normalization layers from PyTorch
    """
    def __init__(self, cfg: Norm2dCfg, dim: int) -> None:
        # NOTE this order is essential to not re-initialize nn.Module (common parent class) 
        # and as a result lose parameters (nn.Parameter attributes) initialized by torch_cls 
        Norm2d.__init__(self, cfg, dim)
        if dim_key is None:
            # If no dim_key given, assume dimension to be first positional argument
            torch_cls.__init__(self, dim, **cfg.__dict__)
        else:
            torch_cls.__init__(self, **({dim_key: dim} | cfg.__dict__))

    def forward(
        self, 
        input: Float[Tensor, "batch channel height width"]
    ) -> Float[Tensor, "batch channel height width"]:
        return torch_cls.forward(self, input)
    
    # dynamic torch_cls "re-definition" with Norm1D[cfg_cls] acting as "Mixin"
    cls = new_class(
        torch_cls.__name__, 
        bases=(Norm2d[cfg_cls], torch_cls),
        exec_body=lambda ns: ns.update({
            "_ignore_mro": True,
            "__init__": __init__,
            "forward": forward
        })
    )
    globals()[cls.__name__] = cls
    global __all__
    __all__.append(cls.__name__)
    __all__.append(cfg_cls.__name__)
    return register_norm_2d(name, cfg_cls, cls)


# Register important normalization layers
from dataclasses import dataclass

from torch.nn import (
    BatchNorm2d as _BatchNorm2d, 
    GroupNorm as _GroupNorm, 
    InstanceNorm2d as _InstanceNorm2d
)


@dataclass
class BatchNorm2dCfg(Norm2dCfg):
    eps: float = 1e-5
    momentum: float | None = 0.1
    affine: bool = True
    track_running_stats: bool = True

BatchNorm2d: type[Norm2d[BatchNorm2dCfg]] = _register_pytorch_norm_2d(
    _BatchNorm2d, "batch", BatchNorm2dCfg
)


@dataclass
class GroupNormCfg(Norm2dCfg):
    num_groups: int = 32
    eps: float = 1e-5
    affine: bool = True

GroupNorm: type[Norm2d[GroupNormCfg]] = _register_pytorch_norm_2d(
    _GroupNorm, "group", GroupNormCfg
)


@dataclass
class InstanceNorm2dCfg(Norm2dCfg):
    eps: float = 1e-5
    momentum: float = 0.1
    affine: bool = False
    track_running_stats: bool = False

InstanceNorm2d: type[Norm2d[InstanceNorm2dCfg]] = _register_pytorch_norm_2d(
    _InstanceNorm2d, "instance", InstanceNorm2dCfg
)
