from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from jaxtyping import Float
from numpy import ndarray


@dataclass
class BetaScheduleCfg:
    pass


T = TypeVar("T", bound=BetaScheduleCfg)


class BetaSchedule(Generic[T], ABC):
    def __init__(
        self,
        cfg: T
    ) -> None:
        self.cfg = cfg    

    @abstractmethod
    def __call__(
        self,
        num_timesteps: int
    ) -> Float[ndarray, "num_timesteps"]:
        pass
