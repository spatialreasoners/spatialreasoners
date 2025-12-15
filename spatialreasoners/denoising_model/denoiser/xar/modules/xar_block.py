from dataclasses import dataclass, field, fields

from jaxtyping import Bool, Float
from torch import Tensor
import torch.nn as nn

from ...modules.attention import AttentionCfg, AttnMask, ScoreMod
from ...modules.feed_forward import FeedForwardCfg
from ...modules.norm_1d import Norm1dCfg, LayerNormCfg
from ...modules.pos_embedding import PosEmbedding
from ...modules.vit_block import ViTBlock, ViTBlockCfg
from ...utils import modulate


@dataclass
class xARBlockCfg(ViTBlockCfg):
    # Used in original DiT paper
    norm_layer: Norm1dCfg = field(
        default_factory=lambda: LayerNormCfg(eps=1.e-6, elementwise_affine=False)
    )
    

class xARBlock(ViTBlock):
    # NOTE this block is the same as the DiTBlock except for the changed order of adaLN_modulation outputs
    def __init__(
        self,
        dim: int,
        c_dim: int,
        attention: AttentionCfg,
        feed_forward: FeedForwardCfg,
        norm_layer: Norm1dCfg,
        mlp_ratio: float = 4.
    ):
        super(xARBlock, self).__init__(
            dim,
            attention,
            feed_forward,
            norm_layer,
            mlp_ratio
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(c_dim, 6 * dim, bias=True)
        )

    @classmethod
    def from_config(
        cls: type["xARBlock"], 
        config: xARBlockCfg, 
        dim: int,
        c_dim: int,
    ) -> "xARBlock":
        return cls(dim, c_dim, **{f.name: getattr(config, f.name) for f in fields(config)})

    def forward(
        self, 
        x: Float[Tensor, "*batch token dim"],
        c: Float[Tensor, "*#batch #token c_dim"],
        attn_mask: AttnMask | None = None,
        score_mod: ScoreMod | None = None,
        rel_pos_emb: PosEmbedding | None = None,
        # *#batch token original shape, while *batch for x, c can be empty and token dim with all tokens of batch elements in sparse format
        cache_mask: Bool[Tensor, "..."] | None = None,
        use_cache: bool = False,
        update_cache: bool = False
    ) -> Float[Tensor, "*batch token dim"]:
        # NOTE this order is the only difference compared to the DiTBlock
        gate_msa, gate_mlp, scale_msa, scale_mlp, shift_msa, shift_mlp \
            = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), 
            attn_mask=attn_mask, 
            score_mod=score_mod, 
            rel_pos_emb=rel_pos_emb,
            cache_mask=cache_mask,
            use_cache=use_cache,
            update_cache=update_cache
        )
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
