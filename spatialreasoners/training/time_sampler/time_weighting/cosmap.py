from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor, pi

from . import TimeWeighting, TimeWeightingCfg, register_time_weighting


@dataclass
class CosmapCfg(TimeWeightingCfg):
    pass


@register_time_weighting("cosmap", CosmapCfg)
class Cosmap(TimeWeighting[CosmapCfg]): 
    
    def __call__(
        self,
        t: Float[Tensor, "*batch"]
    ) -> Float[Tensor, "*batch"]:
        bot = 1 - 2 * t + 2 * t**2
        return 2 / (pi * bot)
