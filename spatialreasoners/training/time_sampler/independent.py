from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float
from torch import Tensor, device

from . import register_time_sampler
from .two_stage_time_sampler import TwoStageTimeSampler, TwoStageTimeSamplerCfg


@dataclass
class IndependentCfg(TwoStageTimeSamplerCfg):
    pass


@register_time_sampler("independent", IndependentCfg)
class Independent(TwoStageTimeSampler[IndependentCfg]):

    def get_time(
        self, 
        batch_size: int, 
        num_samples: int = 1,
        device: device | str = "cpu",
    ) -> tuple[
        Float[Tensor, "batch sample num_variables"], # t
        None
    ]:
        return self.scalar_time_sampler(
            (batch_size, num_samples, self.num_variables), 
            device
        ), None
    
    def get_normalization_weights(
        self, 
        t: Float[Tensor, "*batch"],
        mask: Bool[Tensor, "*#batch"] | None = None
    ) -> Float[Tensor, "*batch"]:
        return torch.ones((1,), device=t.device).expand_as(t)
