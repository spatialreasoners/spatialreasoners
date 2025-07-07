from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass, field, fields

from jaxtyping import Bool, Float, Int64
from torch import Tensor, nn

from spatialreasoners.misc.nn_module_tools import freeze

from ...tokenizer.unet_tokenizer.unet_tokenizer import UNetTokenizer
from .. import register_denoiser
from ..activations import Activation
from ..denoiser import Denoiser, DenoiserCfg
from ..embedding import EmbeddingCfg, EmbeddingSinusodialCfg, get_embedding
from ..norm_layers import Norm
from .modules import (
    Bottleneck,
    Decoder,
    DownsampleCfg,
    Encoder,
    MultiHeadAttentionCfg,
    ResBlockCfg,
    UpsampleCfg,
)
from .type_extensions import UNetModelInputs, UNetModelOutput


@dataclass(frozen=True, kw_only=True)
class UNetCfg(DenoiserCfg):
    time_embedding: EmbeddingCfg = field(default_factory=EmbeddingSinusodialCfg)
    hid_dims: list[int]
    attention: bool | list[bool]
    num_blocks: int | list[int] = 2
    res_block_cfg: ResBlockCfg = field(default_factory=ResBlockCfg)
    attention_cfg: MultiHeadAttentionCfg = field(default_factory=MultiHeadAttentionCfg)
    downsample_cfg: DownsampleCfg = field(default_factory=DownsampleCfg)
    upsample_cfg: UpsampleCfg = field(default_factory=UpsampleCfg)
    out_norm: Norm = "group"
    out_act: Activation = "silu"
    time_embedding_dim: int = 256
    has_bottleneck: bool = True

@register_denoiser("unet", UNetCfg)
class UNet(Denoiser[UNetCfg, UNetModelInputs, UNetModelOutput]):
    def __init__(
        self,
        cfg: UNetCfg,
        tokenizer: UNetTokenizer,
        num_classes: int     | None = None
    ) -> None:
        super(UNet, self).__init__(cfg, tokenizer, num_classes)
        
        self.t_emb = get_embedding(cfg.time_embedding, self.d_time_embedding)
        
        self.encoder = Encoder(
            in_dim=self.tokenizer.in_channels,
            t_emb=self.t_emb,
            hid_dims=self.cfg.hid_dims,
            attention=self.cfg.attention,
            num_blocks=self.cfg.num_blocks,
            res_block_cfg=self.cfg.res_block_cfg,
            attention_cfg=self.cfg.attention_cfg,
            downsample_cfg=self.cfg.downsample_cfg
        )
        
        self.bottleneck = Bottleneck(
            in_dim=self.encoder.out_dim,
            t_emb=self.t_emb,
            res_block_cfg=self.cfg.res_block_cfg,
            attention_cfg=self.cfg.attention_cfg
        ) if self.cfg.has_bottleneck else None

        self.decoder = Decoder(
            in_dim=self.encoder.channels_list,
            t_emb=self.t_emb,
            out_dim=self.tokenizer.out_channels,
            hid_dims=self.cfg.hid_dims,
            attention=self.cfg.attention,
            num_blocks=self.cfg.num_blocks,
            res_block_cfg=self.cfg.res_block_cfg,
            attention_cfg=self.cfg.attention_cfg,
            upsample_cfg=self.cfg.upsample_cfg
        )
        self.init_weights()
        
    def freeze_time_embedding(self) -> None:
        freeze(self.t_emb, eval=False)

    def init_weights(self) -> None:
        self.encoder.init_weights()
        if self.bottleneck is not None:
            self.bottleneck.init_weights()
        self.decoder.init_weights()

    def forward(
        self, 
        model_inputs: UNetModelInputs,
        sample: bool = False
    ) -> UNetModelOutput:
        c_emb = self.embed_conditioning(model_inputs.label, model_inputs.label_mask)
        h, hs = self.encoder(model_inputs.z_t, model_inputs.t, c_emb)
        if self.bottleneck is not None:
            h = self.bottleneck(h, model_inputs.t, c_emb)
        out = self.decoder(h, hs, model_inputs.t, c_emb)
        return out
    
    @property
    def d_conditioning(self) -> int:
        return self.cfg.time_embedding_dim
    
    @property
    def d_time_embedding(self) -> int:
        return self.cfg.time_embedding_dim
