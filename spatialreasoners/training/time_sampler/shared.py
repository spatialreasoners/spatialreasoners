from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float
from torch import Tensor, device

from . import register_time_sampler
from .two_stage_time_sampler import TwoStageTimeSampler, TwoStageTimeSamplerCfg


@dataclass
class SharedCfg(TwoStageTimeSamplerCfg):
    pass


@register_time_sampler("shared", SharedCfg)
class Shared(TwoStageTimeSampler[SharedCfg]):

    def get_time(
        self, 
        batch_size: int, 
        num_samples: int = 1,
        device: device | str = "cpu",
    ) -> tuple[
        Float[Tensor, "batch sample num_variables"], # t
        None # mask
    ]:
        t = self.scalar_time_sampler(
            (batch_size, num_samples, 1, 1), device
        ).expand(-1, -1, self.num_variables)
        return t, None
    
    def get_normalization_weights(
        self, 
        t: Float[Tensor, "*batch"],
        mask: Bool[Tensor, "*#batch"] | None = None
    ) -> Float[Tensor, "*#batch"]:
        return torch.ones((1,), device=t.device).expand_as(t)
