from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from .. import register_sampling_schedule
from .certainty_sequential import CertaintySequential, CertaintySequentialCfg


@dataclass(kw_only=True)
class AdaptiveSequentialCfg(CertaintySequentialCfg):
    pass


@register_sampling_schedule("adaptive_sequential", AdaptiveSequentialCfg)
class AdaptiveSequential(CertaintySequential[AdaptiveSequentialCfg]):
    def _get_uncertainty(
        self,
        t: Float[Tensor, "batch total_patches"],
        sigma_theta: Float[Tensor, "batch total_patches"],
    ) -> Float[Tensor, "batch total_patches"]:

        return sigma_theta
