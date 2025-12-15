from dataclasses import dataclass, fields

from jaxtyping import Float
from torch import Tensor, nn


@dataclass
class PatchEmbeddingCfg:
    bias: bool = True


class PatchEmbedding(nn.Module):
    """ anyD Patch Embedding"""
    # TODO add configurable norm layer after proj?

    def __init__(
        self,
        d_in: int,
        d_out: int,
        bias: bool = True
    ):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out, bias=bias)

    @classmethod
    def from_config(
        cls: type["PatchEmbedding"], 
        config: PatchEmbeddingCfg, 
        d_in: int,
        d_out: int,
    ) -> "PatchEmbedding":
        return cls(d_in, d_out, **{f.name: getattr(config, f.name) for f in fields(config)})

    def forward(
        self, 
        x: Float[Tensor, "batch patch d_in"]
    ) -> Float[Tensor, "batch patch d_out"]:
        return self.proj(x)

    def init_weights(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0)
