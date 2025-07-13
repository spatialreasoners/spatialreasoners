from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from diffusers import AutoencoderKL as Model
from diffusers.models.autoencoders.vae import (
    DiagonalGaussianDistribution as LatentDistribution,
)
from jaxtyping import Float
from torch import Tensor

from . import register_autoencoder
from .image_autoencoder import ImageAutoencoder, ImageAutoencoderCfg


@dataclass
class AutoencoderKLArchitectureCfg:
    down_block_types: list[str]
    up_block_types: list[str]
    block_out_channels: list[int]
    layers_per_block: int
    latent_channels: int


@dataclass
class KLImageAutoencoderCfg(ImageAutoencoderCfg):
    scaling_factor: float
    pretrained_model_name: str | None = None
    checkpoint_path: Path | None = None
    latent_stats_path: Path | None = None
    architecture: AutoencoderKLArchitectureCfg | None = None


@register_autoencoder("kl_image", KLImageAutoencoderCfg)
class KLImageAutoencoder(ImageAutoencoder[KLImageAutoencoderCfg]):
    def __init__(
        self, 
        cfg: KLImageAutoencoderCfg,
        input_shape: Sequence[int],
    ) -> None:
        assert (cfg.pretrained_model_name is None) != (cfg.checkpoint_path is None)
        super().__init__(cfg, input_shape=input_shape)
        if cfg.pretrained_model_name is not None:
            self.model: Model = Model.from_pretrained(cfg.pretrained_model_name)
        else:
            assert cfg.architecture is not None
            self.model: Model = Model.from_config(asdict(cfg.architecture))
            self.model.load_state_dict(torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=True))
        self.register_buffer("mean", None, persistent=False)
        self.register_buffer("std", None, persistent=False)
        if cfg.latent_stats_path is not None:
            latent_stats = torch.load(
                cfg.latent_stats_path, map_location="cpu", weights_only=True
            )
            assert set(latent_stats.keys()) <= {"mean", "std"}
            self.mean = latent_stats["mean"]
            self.std = latent_stats["std"]
    
    def encode_deterministic(
        self,
        image: Float[Tensor, "batch 3 height width"]
    ) -> LatentDistribution:
        return self.model.encode(image).latent_dist
    
    def encoding_to_tensor(
        self,
        encoding: LatentDistribution
    ) -> Float[Tensor, "batch ..."]:
        return encoding.parameters

    def tensor_to_encoding(
        self,
        t: Float[Tensor, "batch ..."]
    ) -> LatentDistribution:
        return LatentDistribution(t)

    def sample_latent(
        self,
        encoding: LatentDistribution
    ) -> Float[Tensor, "batch d_latent h_latent w_latent"]:
        latent = encoding.sample()
        if self.mean is not None:
            latent.sub_(self.mean)
        scale = self.cfg.scaling_factor
        if self.std is not None:
            scale = scale / self.std
        return latent.mul_(scale)

    def decode(
        self,
        latent: Float[Tensor, "batch d_latent h_latent w_latent"]
    ) -> Float[Tensor, "batch 3 height width"]:
        scale = self.cfg.scaling_factor
        if self.std is not None:
            scale = scale / self.std
        latent = latent / scale
        if self.mean is not None:
            latent = latent + self.mean
        return self.model.decode(latent).sample

    @property
    def downscale_factor(self) -> int:
        return 2 ** (len(self.model.encoder.down_blocks)-1)

    @property
    def d_latent(self) -> int:
        return self.model.decoder.conv_in.in_channels
    