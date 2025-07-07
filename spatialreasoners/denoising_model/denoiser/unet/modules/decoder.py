from copy import deepcopy
from dataclasses import dataclass, fields

import torch
from jaxtyping import Float
from torch import Tensor, nn

from spatialreasoners.misc.nn_module_tools import constant_init
from spatialreasoners.misc.utils import to_list

from ...activations import Activation, get_activation
from ...embedding import Embedding
from ...norm_layers import Norm, get_norm
from . import EmbedSequential, MultiHeadAttention, ResBlock, Upsample
from .embed_sequential import EmbedSequential
from .multi_head_attention import MultiHeadAttention, MultiHeadAttentionCfg
from .res_block import ResBlock, ResBlockCfg
from .upsample import Upsample, UpsampleCfg


@dataclass
class DecoderCfg:
    hid_dims: list[int]
    attention: bool | list[bool]
    num_blocks: int | list[int]
    res_block_cfg: ResBlockCfg
    attention_cfg: MultiHeadAttentionCfg
    upsample_cfg: UpsampleCfg
    out_norm: Norm = "group"
    out_act: Activation = "silu"


class Decoder(nn.Module):
    def __init__(
        self,
        in_dim: list[int],
        t_emb: Embedding,
        out_dim: int,
        hid_dims: list[int],
        attention: bool | list[bool],
        num_blocks: int | list[int],
        res_block_cfg: ResBlockCfg,
        attention_cfg: MultiHeadAttentionCfg,
        upsample_cfg: UpsampleCfg,
        out_norm: Norm = "group",
        out_act: Activation = "silu"
    ) -> None:
        super(Decoder, self).__init__()
        attention = to_list(attention, len(hid_dims))
        num_blocks = to_list(num_blocks, len(hid_dims))
        # construct the decoder part of Unet
        in_dim = deepcopy(in_dim)
        cur_dim = in_dim[-1]
        self.blocks = nn.ModuleList()
        for level, hid_dim in enumerate(hid_dims[::-1]):
            for idx in range(num_blocks[len(num_blocks) - 1 - level] + 1):
                layers = [ResBlock.from_config(
                    res_block_cfg,
                    cur_dim + in_dim.pop(),
                    t_emb=t_emb,
                    out_channels=hid_dim
                )]
                cur_dim = hid_dim
                if attention[len(attention) - 1 - level]:
                    layers.append(MultiHeadAttention.from_config(
                        attention_cfg,
                        cur_dim,
                    ))
                if level != len(hid_dims) - 1 and idx == num_blocks[len(num_blocks) - 1 - level]:
                    layers.append(Upsample.from_config(
                        upsample_cfg,
                        cur_dim,
                    ))
                self.blocks.append(EmbedSequential(*layers))

        self.out = nn.Sequential(
            get_norm(out_norm, cur_dim),
            get_activation(out_act),
            nn.Conv2d(cur_dim, out_dim, 3, 1, padding=1)
        )

    @classmethod
    def from_config(
        cls: type["Decoder"], 
        config: DecoderCfg,
        in_dim: list[int],
        t_emb: Embedding,
        out_dim: int,
    ) -> "Decoder":
        return cls(in_dim, t_emb, out_dim, **{f.name: getattr(config, f.name) for f in fields(config)})

    def init_weights(self):
        """Init weights for models.
        We just use the initialization method proposed in the original paper.
        """
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d) and ('conv_2' in n or 'out' in n):
                constant_init(m, 0)
            if isinstance(m, nn.Conv1d) and 'proj' in n:
                constant_init(m, 0)

    def forward(
        self,
        h: Float[Tensor, "batch d_in height width"],
        hs: list[Float[Tensor, "batch _ _ _"]], 
        t: Float[Tensor, "batch 1 orig_height orig_width"],
        c_emb: Float[Tensor, "#batch d_hid"] | None = None
    ) -> Float[Tensor, "batch d_out orig_height orig_width"]:
        for block in self.blocks:
            h = torch.cat((h, hs.pop()), dim=1)
            h = block(h, t, c_emb, hs[-1].shape[-2:] if hs else None)
        out = self.out(h)
        return out
