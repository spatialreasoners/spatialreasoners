from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float
from torch import Tensor, device

from spatialreasoners.misc.tensor import unsqueeze_as

from . import register_time_sampler
from .two_stage_time_sampler import TwoStageTimeSampler, TwoStageTimeSamplerCfg


@dataclass
class MeanSimpleCfg(TwoStageTimeSamplerCfg):
    pass


@register_time_sampler("mean_simple", MeanSimpleCfg)
class MeanSimple(TwoStageTimeSampler[MeanSimpleCfg]):
    """NOTE this sampling is limited in the sense that 
    it does not sample from all possible time maps with a given mean"""

    def get_time_with_mean(
        self,
        mean: Float[Tensor, "*batch"]
    ) -> Float[Tensor, "*batch height width"]:
        beta = torch.minimum(mean, 1 - mean)
        samples = torch.rand(beta.shape + [self.num_variables], device=beta.device)
        samples.mul_(unsqueeze_as(2 * beta, samples))
        samples.add_(unsqueeze_as(mean - beta, samples))
        return samples

    def get_time(
        self, 
        batch_size: int, 
        num_samples: int = 1,
        device: device | str = "cpu",
    ) -> tuple[
        Float[Tensor, "batch sample num_variables"], # t
        Bool[Tensor, "batch #sample num_variables"] | None # mask
    ]:
        mean = self.scalar_time_sampler((batch_size, num_samples), device)
        return self.get_time_with_mean(mean), None
