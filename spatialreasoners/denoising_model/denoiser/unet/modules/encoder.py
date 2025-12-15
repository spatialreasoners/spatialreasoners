from collections.abc import Sequence
from dataclasses import dataclass, fields
from math import ceil

from jaxtyping import Float
from torch import Tensor, nn

from spatialreasoners.misc.nn_module_tools import constant_init
from spatialreasoners.misc.utils import to_list

from ...embedding import Embedding
from .downsample import Downsample, DownsampleCfg
from .embed_sequential import EmbedSequential
from .multi_head_attention import MultiHeadAttention, MultiHeadAttentionCfg
from .res_block import ResBlock, ResBlockCfg


@dataclass
class EncoderCfg:
    hid_dims: list[int]
    attention: bool | list[bool]
    num_blocks: int | list[int]
    res_block_cfg: ResBlockCfg
    attention_cfg: MultiHeadAttentionCfg
    downsample_cfg: DownsampleCfg


class Encoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        t_emb: Embedding,
        hid_dims: list[int],
        attention: bool | list[bool],
        num_blocks: int | list[int],
        res_block_cfg: ResBlockCfg,
        attention_cfg: MultiHeadAttentionCfg,
        downsample_cfg: DownsampleCfg
    ) -> None:
        super(Encoder, self).__init__()
        self.blocks = nn.ModuleList([
            EmbedSequential(
                nn.Conv2d(
                    in_dim, 
                    hid_dims[0], 
                    3, 
                    1, 
                    padding=1
                )
            )
        ])
        self.channels_list = [hid_dims[0]]
        
        num_blocks = to_list(num_blocks, len(hid_dims))
        attention = to_list(attention, len(hid_dims))
        cur_dim = hid_dims[0]
        self.num_downsamples = 0
        for level, hid_dim in enumerate(hid_dims):
            for _ in range(num_blocks[level]):
                layers = [ResBlock.from_config(
                    res_block_cfg,
                    cur_dim,
                    t_emb=t_emb,
                    out_channels=hid_dim
                )]
                cur_dim = hid_dim
                if attention[level]:
                    layers.append(MultiHeadAttention.from_config(
                        attention_cfg,
                        cur_dim
                    ))
                self.blocks.append(EmbedSequential(*layers))
                self.channels_list.append(cur_dim)
            
            if level != len(hid_dims) - 1:
                self.blocks.append(
                    EmbedSequential(
                        Downsample.from_config(downsample_cfg, cur_dim)
                    )
                )
                self.num_downsamples += 1
                self.channels_list.append(cur_dim)
        self.out_dim = cur_dim

    @classmethod
    def from_config(
        cls: type["Encoder"], 
        config: EncoderCfg,
        dim: int,
        t_emb: Embedding
    ) -> "Encoder":
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

    def get_output_spatial_size(
        self,
        input_spatial_size: Sequence[int]
    ) -> tuple[int, ...]:
        res = []
        for s in input_spatial_size:
            for _ in range(self.num_downsamples):
                s = ceil(s / 2)
            res.append(s)
        return tuple(res)

    def forward(
        self, 
        x: Float[Tensor, "batch dim height width"], 
        t: Float[Tensor, "batch 1 height width"],
        c_emb: Float[Tensor, "#batch d_hid"] | None = None
    ) -> tuple[
        Float[Tensor, "batch d_out out_height out_width"], 
        list[Float[Tensor, "batch _ _ _"]]
    ]:
        h, hs = x, []
        # forward downsample blocks
        for block in self.blocks:
            h = block(h, t, c_emb)
            hs.append(h)
        return h, hs
