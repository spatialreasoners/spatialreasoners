from spatialreasoners.registry import Registry
from .cfg_scheduler import CFGScheduler, CFGSchedulerCfg


_cfg_scheduler_registry = Registry(CFGScheduler, CFGSchedulerCfg)


def get_cfg_scheduler(cfg: CFGSchedulerCfg) -> CFGScheduler:
    return _cfg_scheduler_registry.build(cfg)


register_cfg_scheduler = _cfg_scheduler_registry.register


from .constant import Constant, ConstantCfg
from .linear import Linear, LinearCfg


__all__ = [
    "CFGScheduler", "CFGSchedulerCfg",
    "Constant", "ConstantCfg",
    "Linear", "LinearCfg",
    "get_cfg_scheduler",
    "register_cfg_scheduler"
]
