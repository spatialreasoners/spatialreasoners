from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from jaxtyping import Float
from torch import Tensor, device


@dataclass
class ScalarTimeSamplerCfg:
    pass


T = TypeVar("T", bound=ScalarTimeSamplerCfg)


class ScalarTimeSampler(Generic[T], ABC):
    def __init__(
        self,
        cfg: T
    ) -> None:
        self.cfg = cfg    
    
    @abstractmethod
    def __call__(
        self,
        shape: Sequence[int],
        device: device | str = "cpu"
    ) -> Float[Tensor, "*shape"]:
        pass
