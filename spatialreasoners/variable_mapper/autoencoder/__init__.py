from typing import Sequence

from spatialreasoners.registry import Registry

from .autoencoder import Autoencoder, AutoencoderCfg

_autoencoder_registry = Registry(Autoencoder, AutoencoderCfg)


def get_autoencoder(cfg: AutoencoderCfg, input_shape: Sequence[int]) -> Autoencoder:
    return _autoencoder_registry.build(cfg, input_shape)


register_autoencoder = _autoencoder_registry.register


from .image_autoencoder import ImageAutoencoder, ImageAutoencoderCfg
from .kl_image_autoencoder import KLImageAutoencoder, KLImageAutoencoderCfg

__all__ = [
    "Autoencoder", "AutoencoderCfg",
    "ImageAutoencoder", "ImageAutoencoderCfg",
    "KLImageAutoencoder", "KLImageAutoencoderCfg",
    "get_autoencoder",
    "register_autoencoder"
]
