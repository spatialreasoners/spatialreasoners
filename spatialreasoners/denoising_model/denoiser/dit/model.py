from dataclasses import dataclass, field

import torch
from torch import nn

from spatialreasoners.misc.nn_module_tools import freeze

from ...denoiser import Denoiser, DenoiserCfg
from ..modules.pos_embedding import (
    FrequencyPosEmbeddingCfg,
    PosEmbeddingCfg,
    get_pos_embedding,
)
from ...tokenizer import DiTTokenizer
from .. import register_denoiser
from ..embedding import EmbeddingCfg, EmbeddingSinusodialCfg, get_embedding
from .modules import (
    DiTBlock,
    DiTBlockCfg,
    OutputLayer,
    OutputLayerCfg,
    PatchEmbedding,
    PatchEmbeddingCfg,
)
from .type_extensions import DiTModelInputs, DiTModelOutput


@dataclass(frozen=True, kw_only=True)
class DiTCfg(DenoiserCfg):
    time_embedding: EmbeddingCfg = field(default_factory=EmbeddingSinusodialCfg)
    block: DiTBlockCfg
    d_hid: int
    depth: int
    patch_embedding: PatchEmbeddingCfg = field(default_factory=PatchEmbeddingCfg)
    pos_embedding: PosEmbeddingCfg = field(default_factory=FrequencyPosEmbeddingCfg)
    rel_pos_embedding: PosEmbeddingCfg | None = None
    out_layer: OutputLayerCfg = field(default_factory=OutputLayerCfg)


@register_denoiser("dit", DiTCfg)
class DiT(Denoiser[DiTCfg, DiTModelInputs, DiTModelOutput]):
    
    def __init__(
        self, 
        cfg: DiTCfg,
        tokenizer: DiTTokenizer,
        num_classes: int | None = None
    ) -> None:
        super().__init__(cfg, tokenizer, num_classes)
        self.t_emb = get_embedding(cfg.time_embedding, self.d_time_embedding)
        
        self.patch_emb = PatchEmbedding.from_config(cfg.patch_embedding, self.d_in, cfg.d_hid)
        self.pos_emb = get_pos_embedding(cfg.pos_embedding, cfg.d_hid, grid_size=self.tokenizer.token_grid_size)
        
        self.blocks = nn.ModuleList([
            DiTBlock.from_config(config=cfg.block, dim=cfg.d_hid, c_dim=self.t_emb.d_out) 
            for _ in range(cfg.depth)
        ])
        first_block: DiTBlock = self.blocks[0]
        self.rel_pos_emb = get_pos_embedding(cfg.rel_pos_embedding, first_block.attn.dim_head, grid_size=self.tokenizer.token_grid_size) \
            if cfg.rel_pos_embedding is not None else None
        self.out_layer = OutputLayer.from_config(cfg.out_layer, cfg.d_hid, self.d_out, self.t_emb.d_out)
        
        self.init_weights()
        
    def freeze_time_embedding(self) -> None:
        freeze(self.t_emb, eval=False)

        
    @property
    def d_in(self) -> int:
        return self.tokenizer.model_d_in

    @property
    def d_out(self) -> int:
        return self.tokenizer.model_d_out
        
    @property
    def d_hid(self) -> int:
        return self.cfg.d_hid
    
    @property
    def d_conditioning(self) -> int:
        return self.d_hid
    
    @property
    def depth(self) -> int:
        return self.cfg.depth

    def init_weights(self) -> None:
        super().init_weights()
        
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        self.patch_emb.init_weights()

        # Zero-out adaLN modulation layers in DiT blocks:
        block: DiTBlock
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            
            
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_emb.mlp[2].weight, std=0.02)

        # Zero-out output layers:
        nn.init.constant_(self.out_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.out_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.out_layer.linear.weight, 0)
        nn.init.constant_(self.out_layer.linear.bias, 0)

    def forward(
        self,
        model_inputs: DiTModelInputs,
        sample: bool = False, # ignored in DiT
    ) -> DiTModelOutput:
        z_t = model_inputs.z_t # [B, token, d_in]
        t = model_inputs.t # [B, token] 
        pos_xy = model_inputs.token_coordinates_xy
        pos_ij = model_inputs.token_coordinates_ij
        
        c_emb = self.embed_conditioning(model_inputs.label, model_inputs.label_mask) # [B, d_c]
        t_emb = self.t_emb(t)
        
        if c_emb is not None:
            t_emb = t_emb + c_emb.unsqueeze(1)
        
        z_t = self.patch_emb(z_t) # [B, token, d_hid]
        z_t = self.pos_emb(z_t, pos_xy, pos_ij)
        
        if self.rel_pos_emb is not None:
            self.rel_pos_emb.set_state(pos_xy.unsqueeze(1), pos_ij.unsqueeze(1))    # add head dimension

        block: DiTBlock
        for block in self.blocks:
            z_t = block(z_t, t_emb, rel_pos_emb=self.rel_pos_emb)

        if self.rel_pos_emb is not None:
            self.rel_pos_emb.del_state()

        z_t = self.out_layer(z_t, t_emb) 
        
        return z_t
