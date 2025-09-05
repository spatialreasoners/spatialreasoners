from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from jaxtyping import Float
from torch import Tensor
from torch.nn import Module


@dataclass
class Norm1dCfg:
    pass


T = TypeVar("T", bound=Norm1dCfg)


class Norm1d(Module, ABC, Generic[T]):
    # Used only for post-hoc interface implementation 
    # with already defined PyTorch modules
    _ignore_mro: bool = False

    def __init__(
        self,
        cfg: T,
        dim: int
    ):
        if self._ignore_mro:
            Module.__init__(self)
        else:
            super().__init__()
        self.cfg = cfg
        self.dim = dim
    
    @abstractmethod
    def forward(
        self, 
        input: Float[Tensor, "*batch dim"]
    ) -> Float[Tensor, "*batch dim"]:
        pass
