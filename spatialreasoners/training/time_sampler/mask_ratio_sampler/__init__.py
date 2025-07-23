from spatialreasoners.registry import Registry
from .mask_ratio_sampler import MaskRatioSampler, MaskRatioSamplerCfg


_mask_ratio_sampler_registry = Registry(MaskRatioSampler, MaskRatioSamplerCfg)


def get_mask_ratio_sampler(cfg: MaskRatioSamplerCfg) -> MaskRatioSampler:
    return _mask_ratio_sampler_registry.build(cfg)


register_mask_ratio_sampler = _mask_ratio_sampler_registry.register


from .trunc_norm import TruncNorm, TruncNormCfg
from .uniform import Uniform, UniformCfg


__all__ = [
    "MaskRatioSampler", "MaskRatioSamplerCfg",
    "TruncNorm", "TruncNormCfg",
    "Uniform", "UniformCfg",
    "get_mask_ratio_sampler",
    "register_mask_ratio_sampler"
]
