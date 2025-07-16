from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from jaxtyping import Float
from torch import nn, Tensor

from ..activations import Activation, get_activation


@dataclass
class EmbeddingCfg:
    """
    d_out (int): The dimensionality of the output embedding (after MLP).
    act (str): Name of activation layer. Default: SiLU.
    """
    d_emb: int
    act: Activation = "silu"


T = TypeVar("T", bound=EmbeddingCfg)


class Embedding(nn.Module, Generic[T], ABC):
    """(Time) embedding layer, reference to Two level embedding. First embedding
    (time) by an embedding function, then feed to neural networks.

    Args:
        d_out (int): The dimensionality of the output embedding (after MLP).
    """
    cfg: T

    def __init__(
        self,
        cfg: T,
        d_out: int
    ):
        super(Embedding, self).__init__()
        self.cfg = cfg
        self.d_out = d_out
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_emb, d_out),
            get_activation(cfg.act),
            nn.Linear(d_out, d_out))

    @abstractmethod
    def embed(
        self, 
        x: Float[Tensor, "*batch"]
    ) -> Float[Tensor, "*batch d_emb"]:
        """Create embeddings.

        Args:
            x: Data to embed.

        Returns:
            Intermediate embeddings.
        """
        pass

    def forward(
        self, 
        x: Float[Tensor, "*batch"]
    ) -> Float[Tensor, "*batch d_out"]:
        """Forward function for embedding layer.
        Args:
            x: Input data.

        Returns:
            Data embeddings.

        """
        return self.mlp(self.embed(x))
