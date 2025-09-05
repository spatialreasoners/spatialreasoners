from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from einops import rearrange, repeat
from jaxtyping import Float
import torch
from torch import Tensor

from . import register_pos_embedding
from .pos_embedding import PosEmbedding, PosEmbeddingCfg


@dataclass
class RotaryPosEmbeddingCfg(PosEmbeddingCfg):
    """TODO  LightningDiT uses ij here; make this consistent with frequency?"""
    indexing: Literal["ij", "xy"] = "ij"
    max_period: int = 10_000
    max_freq: int = 10
    num_freqs: int = 1


@register_pos_embedding("rotary", RotaryPosEmbeddingCfg)
class RotaryPosEmbedding(PosEmbedding[RotaryPosEmbeddingCfg]):
    def __init__(
        self,
        cfg: RotaryPosEmbeddingCfg,
        dim: int,
        grid_size: Sequence[int]
    ) -> None:
        super(RotaryPosEmbedding, self).__init__(cfg, dim, grid_size)
        assert dim % (2 * self.num_spatial_dims) == 0
        # split emb_dim over spatial dims and real/imaginary
        self.emb_dim_factor = dim // (2 * self.num_spatial_dims)
        # TODO possibly allow other omega definitions
        omega = torch.linspace(0, 1, self.emb_dim_factor+1)[:-1]
        self.register_buffer("omega", 1. / self.cfg.max_period ** omega, persistent=False)
        self.register_buffer("freqs_sin", None, persistent=False)
        self.register_buffer("freqs_cos", None, persistent=False)

    @staticmethod
    def rotate_half(
        x: Float[Tensor, "*batch dim"]
    ) -> Float[Tensor, "*batch dim"]:
        x = rearrange(x, '... (d r) -> ... d r', r=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return rearrange(x, '... d r -> ... (d r)')

    def embed(
        self, 
        grid: Float[Tensor, "*batch spatial"]
    ) -> tuple[
        Float[Tensor, "*batch dim"],
        Float[Tensor, "*batch dim"]
    ]:
        # [..., num_spatial_dims, emb_dim_factor], outer product
        freqs = grid.unsqueeze(-1) * self.omega
        freqs_sin = repeat(torch.sin(freqs), "... s d -> ... (s d r)", r=2)
        freqs_cos = repeat(torch.cos(freqs), "... s d -> ... (s d r)", r=2)
        return freqs_sin, freqs_cos

    def set_state(
        self,
        pos_xy: Float[Tensor, "*batch token spatial"],
        pos_ij: Float[Tensor, "*batch token spatial"]
    ) -> None:
        pos = pos_xy if self.cfg.indexing == "xy" else pos_ij
        self.freqs_sin, self.freqs_cos = self.embed(pos)

    def del_state(self) -> None:
        self.freqs_sin = None
        self.freqs_cos = None

    @torch.amp.autocast("cuda", enabled=False)
    def modulate(
        self,
        x: Float[Tensor, "*batch token dim"],
    ) -> Float[Tensor, "*batch token dim"]:
        dtype = x.dtype
        out = x * self.freqs_cos + self.rotate_half(x) * self.freqs_sin
        return out.type(dtype)
