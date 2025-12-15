from dataclasses import dataclass

from scipy.stats import truncnorm

from . import MaskRatioSampler, MaskRatioSamplerCfg, register_mask_ratio_sampler


@dataclass
class TruncNormCfg(MaskRatioSamplerCfg):
    min_ratio: float = 0.7
    mean: float = 1.0
    std: float = 0.25


@register_mask_ratio_sampler("trunc_norm", TruncNormCfg)
class TruncNorm(MaskRatioSampler[TruncNormCfg]):
    def __init__(
        self,
        cfg: TruncNormCfg
    ) -> None:
        super(TruncNorm, self).__init__(cfg)
        self.generator = truncnorm(
            (self.cfg.min_ratio - self.cfg.mean) / self.cfg.std, 0, 
            loc=self.cfg.mean, scale=self.cfg.std
        )

    def __call__(self) -> float:
        return self.generator.rvs(1)[0]
