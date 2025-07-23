from spatialreasoners.registry import Registry

from .time_weighting import TimeWeighting, TimeWeightingCfg

_time_weighting_registry = Registry(TimeWeighting, TimeWeightingCfg)


def get_time_weighting(
    cfg: TimeWeightingCfg
) -> TimeWeighting:
    return _time_weighting_registry.build(cfg)


register_time_weighting = _time_weighting_registry.register


from .cosmap import Cosmap, CosmapCfg
from .logit_normal import LogitNormal, LogitNormalCfg
from .sigma_sqrt import SigmaSqrt, SigmaSqrtCfg

__all__ = [
    "TimeWeighting", "TimeWeightingCfg",
    "Cosmap", "CosmapCfg",
    "LogitNormal", "LogitNormalCfg",
    "SigmaSqrt", "SigmaSqrtCfg",
    "get_time_weighting",
    "register_time_weighting"
]
