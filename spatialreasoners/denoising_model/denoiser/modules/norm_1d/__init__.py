from spatialreasoners.registry import Registry
from .norm_1d import Norm1d, Norm1dCfg


_norm_1d_registry = Registry(Norm1d, Norm1dCfg)


def get_norm_1d(
    cfg: Norm1dCfg, 
    dim: int
) -> Norm1d:
    return _norm_1d_registry.build(cfg, dim)


register_norm_1d = _norm_1d_registry.register


__all__ = [
    "Norm1d", "Norm1dCfg",
    "get_norm_2d",
    "register_norm_2d"
]


# Functionality for usage of PyTorch normalization layers 
# as registered interface implementations
from types import new_class

from jaxtyping import Float
from torch import Tensor
from torch.nn import Module


def _register_pytorch_norm_1d(
    torch_cls: type[Module],
    name: str | None = None,
    cfg_cls: type[Norm1dCfg] | None = None,
    dim_key: str | None = None
) -> type[Norm1d]:
    """
    Post-hoc interface implementations and registration 
    using normalization layers from PyTorch
    """
    def __init__(self, cfg: Norm1dCfg, dim: int) -> None:
        # NOTE this order is essential to not re-initialize nn.Module (common parent class) 
        # and as a result lose parameters (nn.Parameter attributes) initialized by torch_cls 
        Norm1d.__init__(self, cfg, dim)
        if dim_key is None:
            # If no dim_key given, assume dimension to be first positional argument
            torch_cls.__init__(self, dim, **cfg.__dict__)
        else:
            torch_cls.__init__(self, **({dim_key: dim} | cfg.__dict__))

    def forward(
        self, 
        input: Float[Tensor, "*batch dim"]
    ) -> Float[Tensor, "*batch dim"]:
        return torch_cls.forward(self, input)
    
    # dynamic torch_cls "re-definition" with Norm1D[cfg_cls] acting as "Mixin"
    cls = new_class(
        torch_cls.__name__, 
        bases=(Norm1d[cfg_cls], torch_cls),
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
    return register_norm_1d(name, cfg_cls, cls)


# Register important normalization layers
from dataclasses import dataclass

from torch.nn import (
    LayerNorm as _LayerNorm, 
    RMSNorm as _RMSNorm
)


@dataclass
class LayerNormCfg(Norm1dCfg):
    eps: float = 1e-5
    elementwise_affine: bool = True
    bias: bool = True

LayerNorm: type[Norm1d[LayerNormCfg]] = _register_pytorch_norm_1d(
    _LayerNorm, "layer", LayerNormCfg
)


@dataclass
class RMSNormCfg(Norm1dCfg):
    eps: float | None = None
    elementwise_affine: bool = True

RMSNorm: type[Norm1d[RMSNormCfg]] = _register_pytorch_norm_1d(
    _RMSNorm, "rms", RMSNormCfg
)
