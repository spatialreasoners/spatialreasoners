from collections.abc import Sequence
from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor, device

from . import ScalarTimeSampler, ScalarTimeSamplerCfg, register_scalar_time_sampler


@dataclass
class LogitNormalCfg(ScalarTimeSamplerCfg):
    mean: float = 0.0
    std: float = 1.0


@register_scalar_time_sampler("logit_normal", LogitNormalCfg)
class LogitNormal(ScalarTimeSampler[LogitNormalCfg]): 
    
    def __call__(
        self,
        shape: Sequence[int],
        device: device | str = "cpu"
    ) -> Float[Tensor, "*shape"]:
        return torch.normal(
            self.cfg.mean, 
            self.cfg.std, 
            size=shape, 
            device=device
        ).sigmoid_()
