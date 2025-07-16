from collections.abc import Sequence
from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor, device

from . import ScalarTimeSampler, ScalarTimeSamplerCfg, register_scalar_time_sampler


@dataclass
class UniformCfg(ScalarTimeSamplerCfg):
    pass


@register_scalar_time_sampler("uniform", UniformCfg)
class Uniform(ScalarTimeSampler[UniformCfg]):
    def __call__(
        self,
        shape: Sequence[int],
        device: device | str = "cpu"
    ) -> Float[Tensor, "*shape"]:
        return torch.rand(shape, device=device)
