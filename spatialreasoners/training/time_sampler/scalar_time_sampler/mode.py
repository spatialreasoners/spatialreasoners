from collections.abc import Sequence
from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor, device, pi

from . import ScalarTimeSampler, ScalarTimeSamplerCfg, register_scalar_time_sampler


@dataclass
class ModeCfg(ScalarTimeSamplerCfg):
    scale: float = 1.29


@register_scalar_time_sampler("mode", ModeCfg)
class Mode(ScalarTimeSampler[ModeCfg]): 
    def __call__(
        self,
        shape: Sequence[int],
        device: device | str = "cpu"
    ) -> Float[Tensor, "*shape"]:
        u = torch.rand(shape, device=device)
        return 1 - u - self.cfg.scale * (torch.cos(pi * u / 2) ** 2 - 1 + u)
