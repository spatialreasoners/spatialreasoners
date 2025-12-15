from dataclasses import dataclass

from jaxtyping import Float, Int64
from torch import nn, Tensor

from . import register_class_embedding
from .class_embedding import ClassEmbedding, ClassEmbeddingCfg


@dataclass
class ClassEmbeddingParametersCfg(ClassEmbeddingCfg):
    init_std: float = 0.02


@register_class_embedding("parameters", ClassEmbeddingParametersCfg)
class ClassEmbeddingParameters(ClassEmbedding[ClassEmbeddingParametersCfg]):
    def __init__(
        self,
        cfg: ClassEmbeddingParametersCfg,
        d_out: int,
        num_classes: int
    ):
        super(ClassEmbeddingParameters, self).__init__(cfg, d_out, num_classes)
        self.emb = nn.Embedding(num_classes+1, d_out)  # +1 for empty class

    def embed(
        self, 
        labels: Int64[Tensor, "batch"]
    ) -> Float[Tensor, "batch d_out"]:
        """Create class embeddings."""
        return self.emb(labels)

    def init_weights(self) -> None:
        nn.init.normal_(self.emb.weight, std=self.cfg.init_std)
