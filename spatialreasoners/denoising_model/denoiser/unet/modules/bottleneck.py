from dataclasses import dataclass, fields

from jaxtyping import Float
from torch import Tensor, nn

from spatialreasoners.misc.nn_module_tools import constant_init

from ...embedding import Embedding
from .embed_sequential import EmbedSequential
from .multi_head_attention import MultiHeadAttention, MultiHeadAttentionCfg
from .res_block import ResBlock, ResBlockCfg


@dataclass
class BottleneckCfg:
    res_block_cfg: ResBlockCfg
    attention_cfg: MultiHeadAttentionCfg


class Bottleneck(nn.Module):
    """the bottom part of Unet"""
    cfg: BottleneckCfg

    def __init__(
        self,
        in_dim: int,
        t_emb: Embedding,
        res_block_cfg: ResBlockCfg,
        attention_cfg: MultiHeadAttentionCfg
    ) -> None:

        super(Bottleneck, self).__init__()
        self.blocks = EmbedSequential(
            ResBlock.from_config(
                res_block_cfg,
                in_dim,
                t_emb,
            ),
            MultiHeadAttention.from_config(attention_cfg, in_dim),
            ResBlock.from_config(
                res_block_cfg,
                in_dim,
                t_emb
            )
        )
    
    @classmethod
    def from_config(
        cls: type["Bottleneck"], 
        config: BottleneckCfg, 
        dim: int, 
        t_emb: Embedding
    ) -> "Bottleneck":
        return cls(dim, t_emb, **{f.name: getattr(config, f.name) for f in fields(config)})

    def init_weights(self):
        """Init weights for models.
        We just use the initialization method proposed in the original paper.
        """
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d) and 'conv_2' in n:
                constant_init(m, 0)
            if isinstance(m, nn.Conv1d) and 'proj' in n:
                constant_init(m, 0)

    def forward(
        self, 
        x: Float[Tensor, "batch dim height width"], 
        t: Float[Tensor, "batch 1 orig_height orig_width"],
        c_emb: Float[Tensor, "#batch d_hid"] | None = None
    ) -> Float[Tensor, "batch dim height width"]:
        return self.blocks(x, t, c_emb)
