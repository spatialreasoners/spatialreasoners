from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from jaxtyping import Float
from torch import Tensor


@dataclass
class TimeWeightingCfg:
    pass


T = TypeVar("T", bound=TimeWeightingCfg)


class TimeWeighting(Generic[T], ABC):
    def __init__(
        self,
        cfg: T
    ) -> None:
        self.cfg = cfg    

    @abstractmethod
    def __call__(
        self,
        t: Float[Tensor, "*batch"]
    ) -> Float[Tensor, "*batch"]:
        pass
