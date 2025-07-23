from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from jaxtyping import Float
from torch import Tensor


@dataclass
class CFGSchedulerCfg:
    scale: float | int = 1.0


T = TypeVar("T", bound=CFGSchedulerCfg)


class CFGScheduler(Generic[T], ABC):
    def __init__(
        self,
        cfg: T
    ) -> None:
        self.cfg = cfg

    @abstractmethod
    def __call__(
        self,
        t: Float[Tensor, "*batch dim"],
    ) -> Float[Tensor, "*batch"]:
        pass
