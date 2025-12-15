from spatialreasoners.registry import Registry

from .beta_schedule import BetaSchedule, BetaScheduleCfg

_beta_schedule_registry = Registry(BetaSchedule, BetaScheduleCfg)


def get_beta_schedule(
    cfg: BetaScheduleCfg
) -> BetaSchedule:
    return _beta_schedule_registry.build(cfg)


register_beta_schedule = _beta_schedule_registry.register


from .cosine import Cosine, CosineCfg
from .cosine_simple_diffusion import CosineSimpleDiffusion, CosineSimpleDiffusionCfg
from .linear import Linear, LinearCfg

__all__ = [
    "BetaSchedule", "BetaScheduleCfg",
    "Cosine", "CosineCfg",
    "Linear", "LinearCfg",
    "CosineSimpleDiffusion", "CosineSimpleDiffusionCfg",
    "get_beta_schedule",
    "register_beta_schedule"
]
