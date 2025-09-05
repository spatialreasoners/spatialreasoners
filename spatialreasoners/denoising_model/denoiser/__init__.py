from spatialreasoners.registry import Registry

from ..tokenizer import Tokenizer
from .denoiser import Denoiser, DenoiserCfg

_denoiser_registry = Registry(Denoiser, DenoiserCfg)
register_denoiser = _denoiser_registry.register

def get_denoiser(
    cfg: DenoiserCfg,
    tokenizer: Tokenizer,
    num_classes: int | None = None
) -> Denoiser:
    denoiser: Denoiser = _denoiser_registry.build(
        cfg, tokenizer, num_classes
    )
    denoiser.init_weights()
    denoiser.freeze()
    return denoiser



from .dit.model import DiT, DiTCfg
from .u_vit3d.u_vit3d_pose import UViT3DPose, UViT3DPoseConfig
from .unet.model import UNet, UNetCfg
from .xar.model import xAR, xARCfg

__all__ = [
    "Denoiser", "DenoiserCfg",
    "DiT", "DiTCfg",
    "UNet", "UNetCfg",
    "xAR", "xARCfg",
    "get_denoiser",
    "register_denoiser",
    "UViT3DPose", "UViT3DPoseConfig"
]
