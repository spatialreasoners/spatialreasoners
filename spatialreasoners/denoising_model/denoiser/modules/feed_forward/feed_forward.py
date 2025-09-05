from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from jaxtyping import Float
from torch import Tensor
from torch.nn import Module


@dataclass
class FeedForwardCfg:
    pass


T = TypeVar("T", bound=FeedForwardCfg)


class FeedForward(Generic[T], Module, ABC):
    def __init__(
        self,
        cfg: T,
        d_in: int,
        d_hid: int | None = None,
        d_out: int | None = None
    ) -> None:
        super(FeedForward, self).__init__()
        self.cfg = cfg
        self.d_in = d_in
        self.d_hid = d_hid or d_in
        self.d_out = d_out or d_in

    @abstractmethod
    def forward(
        self, 
        input: Float[Tensor, "*batch d_in"]
    ) -> Float[Tensor, "*batch d_out"]:
        pass
