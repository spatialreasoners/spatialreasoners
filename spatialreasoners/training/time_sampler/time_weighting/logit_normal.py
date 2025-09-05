from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor, pi

from . import TimeWeighting, TimeWeightingCfg, register_time_weighting


@dataclass
class LogitNormalCfg(TimeWeightingCfg):
    mean: float = 0.0
    std: float = 1.0
    eps: float = 1.e-5


@register_time_weighting("logit_normal", LogitNormalCfg)
class LogitNormal(TimeWeighting[LogitNormalCfg]): 
    
    def __call__(
        self,
        t: Float[Tensor, "*batch"]
    ) -> Float[Tensor, "*batch"]:
        logit = torch.log(t/(1-t).clamp_min_(self.cfg.eps))
        denominator: Tensor = self.cfg.std * (2 * pi) ** 0.5 * t * (1-t)
        denominator.clamp_min_(self.cfg.eps)
        return torch.exp(-((logit - self.cfg.mean)**2)/(2 * self.cfg.std**2))\
            .div_(denominator)
