from dataclasses import dataclass, field
from functools import cache
from math import prod

from jaxtyping import Bool
import torch
from torch import nn, Tensor
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from ...denoiser import Denoiser, DenoiserCfg
from ..modules.norm_1d import Norm1dCfg, get_norm_1d, RMSNormCfg
from ..modules.pos_embedding import PosEmbeddingCfg, get_pos_embedding, RotaryPosEmbeddingCfg
from ...tokenizer import ImageDiTTokenizer
from .. import register_denoiser
from .type_extensions import xARModelInputs, xARModelOutput

from .modules import xARBlock, xARBlockCfg


@dataclass(frozen=True, kw_only=True)
class xARCfg(DenoiserCfg):
    encoder_block: xARBlockCfg
    decoder_block: xARBlockCfg
    d_hid: int
    encoder_depth: int
    decoder_depth: int
    encoder_rel_pos_embedding: PosEmbeddingCfg | None = field(default_factory=RotaryPosEmbeddingCfg)
    decoder_rel_pos_embedding: PosEmbeddingCfg | None = field(default_factory=RotaryPosEmbeddingCfg)
    input_norm: Norm1dCfg = field(default_factory=RMSNormCfg)
    encoder_norm: Norm1dCfg = field(default_factory=RMSNormCfg)
    decoder_norm: Norm1dCfg = field(default_factory=RMSNormCfg)


@register_denoiser("xar", xARCfg)
class xAR(Denoiser[xARCfg, xARModelInputs, xARModelOutput]):
    
    def __init__(
        self, 
        cfg: xARCfg,
        tokenizer: ImageDiTTokenizer,
        num_classes: int | None = None
    ) -> None:
        super().__init__(cfg, tokenizer, num_classes)

        # Construct encoder
        self.z_proj = nn.Linear(self.d_in, cfg.d_hid, bias=True)
        self.z_proj_ln = get_norm_1d(cfg.input_norm, cfg.d_hid)
        self.encoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, cfg.d_hid))

        use_flex_encoder_attention = cfg.encoder_block.attention.backends is not None \
            and cfg.encoder_block.attention.backends[0] == "flex_attention"
        use_flex_decoder_attention = cfg.decoder_block.attention.backends is not None \
            and cfg.decoder_block.attention.backends[0] == "flex_attention"
        assert use_flex_encoder_attention == use_flex_decoder_attention, \
            "Expected to either use flex_attention for encoder and decoder or not at all"
        self.use_flex_attention = use_flex_encoder_attention & use_flex_decoder_attention

        self.encoder_blocks = nn.ModuleList([
            xARBlock.from_config(config=cfg.encoder_block, dim=cfg.d_hid, c_dim=self.t_emb.d_out) 
            for _ in range(cfg.encoder_depth)
        ])
        first_encoder_block: xARBlock = self.encoder_blocks[0]
        self.encoder_rel_pos_emb = get_pos_embedding(
            cfg.encoder_rel_pos_embedding, first_encoder_block.attn.dim_head, grid_size=self.tokenizer.token_grid_size
        ) if cfg.encoder_rel_pos_embedding is not None else None
        self.encoder_norm = get_norm_1d(cfg.encoder_norm, cfg.d_hid)

        # Construct decoder
        self.decoder_embed = nn.Linear(cfg.d_hid, cfg.d_hid, bias=True)
        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, cfg.d_hid))

        self.decoder_blocks = nn.ModuleList([
            xARBlock.from_config(config=cfg.decoder_block, dim=cfg.d_hid, c_dim=self.t_emb.d_out) 
            for _ in range(cfg.decoder_depth)
        ])
        first_decoder_block: xARBlock = self.decoder_blocks[0]
        self.decoder_rel_pos_emb = get_pos_embedding(
            cfg.decoder_rel_pos_embedding, first_decoder_block.attn.dim_head, grid_size=self.tokenizer.token_grid_size
        ) if cfg.decoder_rel_pos_embedding is not None else None
        self.decoder_norm = get_norm_1d(cfg.decoder_norm, cfg.d_hid)

        # Out layer
        self.pred = nn.Linear(cfg.d_hid, self.d_out)

        # self.register_buffer("token_cache_mask", None, persistent=False)
        # self.register_buffer("x_cache", None, persistent=False)
        self.init_weights()
        
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
        return self.cfg.encoder_depth + self.cfg.decoder_depth

    @property
    def seq_len(self) -> int:
        return prod(self.tokenizer.token_grid_size)

    @property
    def num_clusters(self) -> int:
        return self.tokenizer.variable_mapper.num_variables

    def init_weights(self) -> None:
        super().init_weights()
        
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)

         # initialize nn.Linear and nn.LayerNorm
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                # we use xavier_uniform following official JAX ViT:
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)
        self.apply(_basic_init)
    
    def create_block_mask(
        self, 
        device: torch.device
    ) -> BlockMask:
        cluster_size = self.seq_len // self.num_clusters
        def mask_mod(b, h, q_idx, kv_idx):
            # block- / cluster-wise causal masking
            return (q_idx // cluster_size) >= (kv_idx // cluster_size)
        return create_block_mask(
            mask_mod, B=None, H=None, Q_LEN=self.seq_len, KV_LEN=self.seq_len, device=device
        )
    
    @cache
    def create_attn_mask(
        self, 
        device: torch.device
    ) -> BlockMask | Bool[Tensor, "batch #head token token"]:
        """Attention within sets and from denoising to conditioning tokens"""
        if self.use_flex_attention:
            return self.create_block_mask()
        
        mask = torch.tril(torch.ones(2 * (self.num_clusters,), dtype=torch.bool, device=device))
        mask = torch.kron(mask, torch.ones(2 * (self.seq_len // self.num_clusters,), dtype=torch.bool, device=device))
        mask = mask[None, None] # add batch and head dimensions
        return mask.contiguous()

    # TODO implement this for caching
    #def free_cache(self) -> None:
    #    self.token_cache_mask = None
    #    self.x_cache = None
    #    block: DiTBlock
    #    for block in self.encoder_blocks:
    #        block.free_cache()
    #    for block in self.decoder_blocks:
    #        block.free_cache()

    #def on_sampling_start(self) -> None:
    #    self.free_cache()

    #def on_sampling_end(self) -> None:
    #    self.free_cache()

    def forward_encoder(self, x, condition, mask, pos_xy, pos_ij):
        encoder_pos_embed_learned=self.encoder_pos_embed_learned
        # TODO adapt this for token caching
        #if self.training:
        #    encoder_pos_embed_learned=self.encoder_pos_embed_learned
        #else:
        #    encoder_pos_embed_learned =self.encoder_pos_embed_learned[:, (scale_index+1)*self.seq_len//self.num_clusters-x.shape[1]:(scale_index+1)*self.seq_len//self.num_clusters]
        #if (not self.training) and (mask is not None):
        #    mask = mask[:,:,(scale_index+1)*self.seq_len//self.num_clusters-x.shape[1]:(scale_index+1)*self.seq_len//self.num_clusters, :]

        x = x + encoder_pos_embed_learned
        x = self.z_proj_ln(x)

        if self.encoder_rel_pos_emb is not None:
            self.encoder_rel_pos_emb.set_state(pos_xy.unsqueeze(1), pos_ij.unsqueeze(1))    # add head dimension

        for block in self.encoder_blocks:
            x = block(x, condition, mask, rel_pos_emb=self.encoder_rel_pos_emb)

        if self.encoder_rel_pos_emb is not None:
            self.encoder_rel_pos_emb.del_state()

        x = self.encoder_norm(x)
        return x

    def forward_decoder(self, x, condition, mask, pos_xy, pos_ij):
        x = self.decoder_embed(x)
        
        decoder_pos_embed_learned=self.decoder_pos_embed_learned
        # TODO adapt this for token caching
        #if self.training:
        #    decoder_pos_embed_learned=self.decoder_pos_embed_learned
        #else:
        #    decoder_pos_embed_learned=self.decoder_pos_embed_learned[:, (scale_index+1)*self.seq_len//self.num_clusters-x.shape[1]:(scale_index+1)*self.seq_len//self.num_clusters]
        #if (not self.training) and (mask is not None):
        #    mask = mask[:,:,(scale_index+1)*self.seq_len//self.num_clusters-x.shape[1]:(scale_index+1)*self.seq_len//self.num_clusters, :(scale_index+1)*self.seq_len//self.num_clusters]
        
        x = x + decoder_pos_embed_learned

        if self.decoder_rel_pos_emb is not None:
            self.decoder_rel_pos_emb.set_state(pos_xy.unsqueeze(1), pos_ij.unsqueeze(1))    # add head dimension

        for block in self.decoder_blocks:
            x = block(x, condition, mask, rel_pos_emb=self.decoder_rel_pos_emb)

        if self.decoder_rel_pos_emb is not None:
            self.decoder_rel_pos_emb.del_state()

        x = self.decoder_norm(x)
        x = self.pred(x)
        return x

    def forward(
        self,
        model_inputs: xARModelInputs,
        sample: bool = False, # TODO use this for token caching!
    ) -> xARModelOutput:
        z_t = model_inputs.z_t # [B, token, d_in]
        t = model_inputs.t # [B, token]
        pos_xy = model_inputs.token_coordinates_xy
        pos_ij = model_inputs.token_coordinates_ij

        t = (1000 * t).floor()
        t_emb = self.t_emb(t)

        c_emb = self.embed_conditioning(model_inputs.label, model_inputs.label_mask) # [B, d_c]
        if c_emb is not None:
            t_emb = t_emb + c_emb.unsqueeze(1)

        x = self.z_proj(z_t)
        attn_mask = self.create_attn_mask(x.device)
        x = self.forward_encoder(x, t_emb, attn_mask, pos_xy, pos_ij)
        x = self.forward_decoder(x, t_emb, attn_mask, pos_xy, pos_ij)
        return x
