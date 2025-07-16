from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from . import TimeWeighting, TimeWeightingCfg, register_time_weighting


@dataclass
class SigmaSqrtCfg(TimeWeightingCfg):
    pass


@register_time_weighting("sigma_sqrt", SigmaSqrtCfg)
class SigmaSqrt(TimeWeighting[SigmaSqrtCfg]): 
    
    def __call__(
        self,
        t: Float[Tensor, "*batch"]
    ) -> Float[Tensor, "*batch"]:
        return t ** -2.0
