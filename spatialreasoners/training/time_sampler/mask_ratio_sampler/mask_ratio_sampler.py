from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar


@dataclass
class MaskRatioSamplerCfg:
    pass


T = TypeVar("T", bound=MaskRatioSamplerCfg)


class MaskRatioSampler(Generic[T], ABC):
    def __init__(
        self,
        cfg: T
    ) -> None:
        self.cfg = cfg    

    @abstractmethod
    def __call__(self) -> float:
        pass
