from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, TypeVar
from typing import Generic

from jaxtyping import Bool, Float
from torch import Tensor, device


@dataclass(frozen=True, kw_only=True)
class NormalizationWeighingCfg:
    pass

T = TypeVar("T", bound=NormalizationWeighingCfg)

class NormalizationWeighing(ABC, Generic[T]):
    def __init__(
        self, 
        cfg: T, 
        get_time_func: Callable[
            [int, int, str], 
            tuple[Float[Tensor, "batch sample num_variables"], Bool[Tensor, "batch #sample num_variables"] | None]
        ],
        device: device | str = "cpu"
    ) -> None:
        self.cfg = cfg
        self.get_time_func = get_time_func  
        self.device = device

    @abstractmethod
    def __call__(
        self, 
        t: Float[Tensor, "*batch"],
        mask: Bool[Tensor, "*#batch"] | None = None
    ) -> Float[Tensor, "*#batch"]:
        pass