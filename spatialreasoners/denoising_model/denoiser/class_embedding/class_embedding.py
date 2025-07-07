from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from jaxtyping import Bool, Float, Int64
import torch
from torch import nn, Tensor


@dataclass
class ClassEmbeddingCfg:
    dropout_prob: float = 0.0


T = TypeVar("T", bound=ClassEmbeddingCfg)


class ClassEmbedding(nn.Module, Generic[T], ABC):
    """Class embedding layer

    Args:
        num_classes (int): The number of classes
        d_out (int): The dimensionality of the output embedding.
    """

    def __init__(
        self,
        cfg: T,
        d_out: int,
        num_classes: int
    ):
        assert 0 <= cfg.dropout_prob <= 1, "Dropout probability has to be in between 0 and 1"
        super(ClassEmbedding, self).__init__()
        self.cfg = cfg
        self.d_out = d_out
        self.num_classes = num_classes

    def drop(
        self, 
        labels: Int64[Tensor, "batch"],
        drop_mask: Bool[Tensor, "batch"] | None = None
    ) -> Int64[Tensor, "batch"]:
        """
        Drops labels to enable classifier-free guidance.
        """
        if drop_mask is None:
            drop_mask = torch.rand(labels.shape[0], device=labels.device) < self.cfg.dropout_prob
        labels = torch.where(drop_mask, self.num_classes, labels)     # empty class with class ID n_classes
        return labels

    @abstractmethod
    def embed(
        self, 
        labels: Int64[Tensor, "batch"]
    ) -> Float[Tensor, "batch d_out"]:
        """Create class embeddings."""
        pass

    def forward(
        self, 
        labels: Int64[Tensor, "batch"],  
        drop_mask: Bool[Tensor, "batch"] | None = None
    ) -> Float[Tensor, "batch d_out"]:
        if (self.training and self.cfg.dropout_prob > 0) or (drop_mask is not None):
            labels = self.drop(labels, drop_mask)
        emb = self.embed(labels)
        return emb

    def init_weights(self) -> None:
        pass
