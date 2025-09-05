from spatialreasoners.registry import Registry
from spatialreasoners.type_extensions import Parameterization

from .flow import Flow, FlowCfg

_flow_registry = Registry(Flow, FlowCfg)


def get_flow(
    cfg: FlowCfg, 
    parameterization: Parameterization
) -> Flow:
    return _flow_registry.build(cfg, parameterization)


register_flow = _flow_registry.register


from .continuous_diffusion_flow import (
    ContinuousCosineLogSNRFlow,
    ContinuousCosineLogSNRFlowCfg,
    ContinuousLinearBetaFlow,
    ContinuousLinearBetaFlowCfg,
)
from .cosine_flow import CosineFlow, CosineFlowCfg
from .diffusion import Diffusion, DiffusionCfg
from .original_diffusion import OriginalDiffusion, OriginalDiffusionCfg
from .rectified_flow import RectifiedFlow, RectifiedFlowCfg

__all__ = [
    "Flow", "FlowCfg",
    "Diffusion", "DiffusionCfg",
    "CosineFlow", "CosineFlowCfg",
    "OriginalDiffusion", "OriginalDiffusionCfg",
    "RectifiedFlow", "RectifiedFlowCfg",
    "ContinuousCosineLogSNRFlow", "ContinuousCosineLogSNRFlowCfg",
    "ContinuousLinearBetaFlow", "ContinuousLinearBetaFlowCfg",
    "get_flow",
    "register_flow"
]
