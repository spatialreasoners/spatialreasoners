from spatialreasoners.registry import Registry

from .scalar_time_sampler import ScalarTimeSampler, ScalarTimeSamplerCfg

_scalar_time_sampler_registry = Registry(ScalarTimeSampler, ScalarTimeSamplerCfg)


def get_scalar_time_sampler(
    cfg: ScalarTimeSamplerCfg
) -> ScalarTimeSampler:
    return _scalar_time_sampler_registry.build(cfg)


register_scalar_time_sampler = _scalar_time_sampler_registry.register


from .logit_normal import LogitNormal, LogitNormalCfg
from .mode import Mode, ModeCfg
from .uniform import Uniform, UniformCfg

__all__ = [
    "ScalarTimeSampler", "ScalarTimeSamplerCfg",
    "LogitNormal", "LogitNormalCfg",
    "Mode", "ModeCfg",
    "Uniform", "UniformCfg",
    "get_scalar_time_sampler",
    "register_scalar_time_sampler"
]
