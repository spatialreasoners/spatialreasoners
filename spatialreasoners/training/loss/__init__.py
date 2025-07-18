from spatialreasoners.denoising_model.flow import Flow
from spatialreasoners.registry import Registry

from .loss import Loss, LossCfg

_loss_registry = Registry(Loss, LossCfg)


def get_loss(
    cfg: LossCfg,
    flow: Flow
) -> Loss:
    return _loss_registry.build(cfg, flow)


register_loss = _loss_registry.register


from .cosine import Cosine, CosineCfg
from .mse import MSE, MSECfg
from .nll import NLL, NLLCfg
from .vlb import VLB, VLBCfg

__all__ = [
    "Loss", "LossCfg",
    "Cosine", "CosineCfg",
    "MSE", "MSECfg",
    "NLL", "NLLCfg",
    "VLB", "VLBCfg",
    "get_loss",
    "register_loss"
]
