
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Sequence, TypeVar

from jaxtyping import Float
from torch import Tensor, nn


@dataclass
class AutoencoderCfg:
    pass

T = TypeVar("T", bound=AutoencoderCfg)

class Autoencoder(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(
        self, 
        cfg: T,
        input_shape: Sequence[int],
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.cfg = cfg
        
    
    @abstractmethod
    def encode_deterministic(
        self,
        data: Float[Tensor, "*batch_raw"]
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
    ) -> Float[Tensor, "*batch_latent"]:
        pass

    def encode(
        self, 
        data: Float[Tensor, "*batch_raw"]
    ) -> Float[Tensor, "*batch_latent"]:
        encoding = self.encode_deterministic(data)
        return self.sample_latent(encoding)

    @abstractmethod
    def decode(
        self,
        latent: Float[Tensor, "*batch_latent"]
    ) -> Float[Tensor, "*batch_raw"]:
        pass
    
    @property
    @abstractmethod
    def latent_shape(self) -> Sequence[int]:
        pass
