from spatialreasoners.registry import Registry
from spatialreasoners.variable_mapper import VariableMapper

from .time_sampler import TimeSampler, TimeSamplerCfg

_time_sampler_registry = Registry(TimeSampler, TimeSamplerCfg)


def get_time_sampler(
    cfg: TimeSamplerCfg, 
    variable_mapper: VariableMapper,
) -> TimeSampler:
    return _time_sampler_registry.build(cfg, variable_mapper=variable_mapper)


register_time_sampler = _time_sampler_registry.register


from .independent import Independent, IndependentCfg
from .mean_beta import MeanBeta, MeanBetaCfg
from .mean_simple import MeanSimple, MeanSimpleCfg
from .sequential import Sequential, SequentialCfg
from .shared import Shared, SharedCfg

__all__ = [
    "TimeSampler", "TimeSamplerCfg",
    "Independent", "IndependentCfg",
    "MeanBeta", "MeanBetaCfg",
    "MeanSimple", "MeanSimpleCfg",
    "Sequential", "SequentialCfg",
    "Shared", "SharedCfg",
    "get_time_sampler",
    "register_time_sampler"
]
