from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence, TypeVar

from jaxtyping import Float
from torch import Tensor

from .autoencoder import Autoencoder, AutoencoderCfg


@dataclass
class ImageAutoencoderCfg(AutoencoderCfg):
    pass


T = TypeVar("T", bound=ImageAutoencoderCfg)


class ImageAutoencoder(Autoencoder[T], ABC):
    cfg: T

    def __init__(
        self, 
        cfg: T,
        input_shape: Sequence[int],
    ) -> None:
        super().__init__(cfg, input_shape)

    @abstractmethod
    def encode_deterministic(
        self,
        image: Float[Tensor, "batch 3 height width"]
    ) -> Any:
        pass
    
    @abstractmethod
    def encoding_to_tensor(
        self,
        encoding: Any
    ) -> Float[Tensor, "batch ..."]:
        pass

    @abstractmethod
    def tensor_to_encoding(
        self,
        t: Float[Tensor, "batch ..."]
    ) -> Any:
        pass

    @abstractmethod
    def sample_latent(
        self,
        encoding: Any
    ) -> Float[Tensor, "batch d_latent h_latent w_latent"]:
        pass

    def encode(
        self, 
        data: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, "batch d_latent h_latent w_latent"]:
        encoding = self.encode_deterministic(data)
        return self.sample_latent(encoding)

    @abstractmethod
    def decode(
        self,
        latent: Float[Tensor, "batch d_latent h_latent w_latent"]
    ) -> Float[Tensor, "batch 3 height width"]:
        pass
    
    @property
    @abstractmethod
    def downscale_factor(self) -> int:
        pass
    
    @property
    @abstractmethod
    def d_latent(self) -> int:
        pass
    
    @property 
    def latent_shape(self) -> Sequence[int]:
        w = self.input_shape[2] // self.downscale_factor
        h = self.input_shape[1] // self.downscale_factor        
        return (self.d_latent, w, h)
