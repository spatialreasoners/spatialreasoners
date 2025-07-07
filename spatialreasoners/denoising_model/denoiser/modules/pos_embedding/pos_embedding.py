from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from math import prod
from typing import Generic, Literal, TypeVar

from jaxtyping import Float
from torch import Tensor
from torch.nn import Module


@dataclass
class PosEmbeddingCfg:
    """NOTE assumes xy coordinates and therefore (width, height) as grid_size in 2D"""
    indexing: Literal["ij", "xy"] = "xy"


T = TypeVar("T", bound=PosEmbeddingCfg)


class PosEmbedding(Generic[T], Module, ABC):
    def __init__(
        self,
        cfg: T,
        dim: int,
        grid_size: Sequence[int]
    ) -> None:
        super(PosEmbedding, self).__init__()
        self.cfg = cfg
        self.dim = dim
        self.num_spatial_dims = len(grid_size)
        self.num_grid_elem = prod(grid_size)
    
    @abstractmethod
    def set_state(
        self,
        pos_xy: Float[Tensor, "*batch token spatial"],
        pos_ij: Float[Tensor, "*batch token spatial"]
    ) -> None:
        pass
    
    @abstractmethod
    def modulate(
        self,
        x: Float[Tensor, "*batch token dim"],
    ) -> Float[Tensor, "*batch token dim"]:
        pass
    
    @abstractmethod
    def del_state(self) -> None:
        pass

    def forward(
        self,
        x: Float[Tensor, "*batch token dim"],
        pos_xy: Float[Tensor, "*batch token spatial"],
        pos_ij: Float[Tensor, "*batch token spatial"]
    ) -> Float[Tensor, "*batch token dim"]:
        self.set_state(pos_xy, pos_ij)
        out = self.modulate(x)
        self.del_state()
        return out
