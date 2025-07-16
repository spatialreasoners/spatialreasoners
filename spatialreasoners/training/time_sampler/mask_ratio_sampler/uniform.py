from dataclasses import dataclass
from random import uniform

from . import MaskRatioSampler, MaskRatioSamplerCfg, register_mask_ratio_sampler


@dataclass
class UniformCfg(MaskRatioSamplerCfg):
    min_ratio: float = 0.
    max_ratio: float = 1.


@register_mask_ratio_sampler("uniform", UniformCfg)
class Uniform(MaskRatioSampler[UniformCfg]):
    def __init__(
        self,
        cfg: UniformCfg
    ) -> None:
        super(Uniform, self).__init__(cfg)

    def __call__(self) -> float:
        return uniform(self.cfg.min_ratio, self.cfg.max_ratio)
