from dataclasses import dataclass, field, fields

from jaxtyping import Bool, Float
from torch import Tensor
import torch.nn as nn

from .attention import Attention, AttentionCfg, AttnMask, ScoreMod
from .feed_forward import FeedForwardCfg, get_feed_forward, MlpCfg
from .norm_1d import get_norm_1d, Norm1dCfg, LayerNormCfg
from .pos_embedding import PosEmbedding


@dataclass
class ViTBlockCfg:
    attention: AttentionCfg = field(default_factory=AttentionCfg)
    feed_forward: FeedForwardCfg = field(default_factory=MlpCfg)
    norm_layer: Norm1dCfg = field(default_factory=LayerNormCfg)
    mlp_ratio: float = 4.
    

class ViTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        attention: AttentionCfg,
        feed_forward: FeedForwardCfg,
        norm_layer: Norm1dCfg,
        mlp_ratio: float = 4.,
    ):
        super().__init__()
        self.norm1 = get_norm_1d(norm_layer, dim)
        self.attn = Attention.from_config(attention, dim)
        self.norm2 = get_norm_1d(norm_layer, dim)
        # NOTE attribute name mlp for backwards compatibility with checkpoints
        self.mlp = get_feed_forward(feed_forward, dim, int(dim * mlp_ratio))

    @classmethod
    def from_config(
        cls: type["ViTBlock"], 
        config: ViTBlockCfg, 
        dim: int,
    ) -> "ViTBlock":
        return cls(dim, **{f.name: getattr(config, f.name) for f in fields(config)})

    def free_cache(self) -> None:
        self.attn.free_cache()

    def forward(
        self, 
        x: Float[Tensor, "*batch token dim"],
        attn_mask: AttnMask | None = None, 
        score_mod: ScoreMod | None = None,
        rel_pos_emb: PosEmbedding | None = None,
        # *#batch token original shape, while *batch for x, c can be empty and token dim with all tokens of batch elements in sparse format
        cache_mask: Bool[Tensor, "..."] | None = None,
        use_cache: bool = False,
        update_cache: bool = False
    ) -> Float[Tensor, "*batch token dim"]:
        x = x + self.attn(
            self.norm1(x), attn_mask=attn_mask, score_mod=score_mod, rel_pos_emb=rel_pos_emb, \
                cache_mask=cache_mask, use_cache=use_cache, update_cache=update_cache
        )
        x = x + self.mlp(self.norm2(x))
        return x
