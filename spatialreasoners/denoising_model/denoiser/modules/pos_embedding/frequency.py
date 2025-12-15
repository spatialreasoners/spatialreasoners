from collections.abc import Sequence
from dataclasses import dataclass

from jaxtyping import Float
import torch
from torch import Tensor

from . import register_pos_embedding
from .pos_embedding import PosEmbedding, PosEmbeddingCfg


@dataclass
class FrequencyPosEmbeddingCfg(PosEmbeddingCfg):
    max_period: int = 10_000


@register_pos_embedding("frequency", FrequencyPosEmbeddingCfg)
class FrequencyPosEmbedding(PosEmbedding[FrequencyPosEmbeddingCfg]):
    def __init__(
        self,
        cfg: FrequencyPosEmbeddingCfg,
        dim: int,
        grid_size: Sequence[int]
    ):
        super(FrequencyPosEmbedding, self).__init__(cfg, dim, grid_size)
        assert dim % (2 * self.num_spatial_dims) == 0
        self.emb_dim_factor = dim // (2 * self.num_spatial_dims) # split emb_dim over spatial dims and sin/cos
        # equivalent to np.linspace(..., endpoint=False)
        omega = torch.linspace(0, 1, self.emb_dim_factor+1)[:-1] # [emb_dim_factor]
        self.register_buffer("omega", 1. / self.cfg.max_period ** omega, persistent=False)
        self.register_buffer("emb", None, persistent=False)

    def embed(
        self, 
        grid: Float[Tensor, "*batch spatial"]
    ) -> Float[Tensor, "*batch dim"]:
        # [..., num_spatial_dims, emb_dim_factor], outer product
        out = grid.unsqueeze(-1) * self.omega
        emb_sin = torch.sin(out)
        emb_cos = torch.cos(out)
        emb = torch.cat((emb_sin, emb_cos), dim=-1)
        emb = emb.flatten(-2)
        return emb

    def set_state(
        self,
        pos_xy: Float[Tensor, "*batch token spatial"],
        pos_ij: Float[Tensor, "*batch token spatial"]
    ) -> None:
        pos = pos_xy if self.cfg.indexing == "xy" else pos_ij
        self.emb = self.embed(pos)

    def del_state(self) -> None:
        self.emb = None

    def modulate(
        self,
        x: Float[Tensor, "*batch token dim"],
    ) -> Float[Tensor, "*batch token dim"]:
        return x + self.emb
