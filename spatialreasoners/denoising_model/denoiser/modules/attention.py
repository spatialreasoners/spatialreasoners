from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, fields
from functools import cache
from math import sqrt
from typing import Literal, Union

from einops import rearrange, repeat
from jaxtyping import Bool, Float, Int32
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from .norm_1d import get_norm_1d, Norm1dCfg
from .pos_embedding import PosEmbedding


Backend = Literal[
    "math", 
    "flash_attention", 
    "flex_attention",
    "efficient_attention", 
    "cudnn_attention", 
    "naive",
]

AttnMask = Union[
    BlockMask,
    Bool[Tensor, "*#batch #head target source"],
    Float[Tensor, "*#batch #head target source"],
]
ScoreMod = Callable[
    [
        Float[Tensor, ""],
        Int32[Tensor, ""],
        Int32[Tensor, ""],
        Int32[Tensor, ""],
        Int32[Tensor, ""]
    ], 
    Float[Tensor, ""]
]


@cache
def compile_flex_attention():
    # If used, should always be compiled
    return torch.compile(flex_attention, dynamic=True)


def naive_scaled_dot_product_attention(
    query: Float[Tensor, "*#batch h_query target d_key"], 
    key: Float[Tensor, "*#batch h_key source d_key"], 
    value: Float[Tensor, "*#batch h_key source d_val"], 
    attn_mask: Union[
        Bool[Tensor, "*#batch #h_query target source"],
        Float[Tensor, "*#batch #h_query target source"],
        None
    ] = None, 
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False
) -> Float[Tensor, "*#batch h_query target d_val"]:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / sqrt(query.size(-1)) if scale is None else scale
    device, dtype = query.device, query.dtype
    
    attn_bias = None
    if is_causal:
        assert attn_mask is None
        attn_mask = torch.ones(L, S, device=device, dtype=torch.bool).tril(diagonal=0)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias = torch.zeros(attn_mask.shape, device=device, dtype=dtype)
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask.to(dtype=dtype, device=device)
    
    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ rearrange(key, "... s d -> ... d s") * scale_factor
    if attn_bias is not None:
        attn_weight += attn_bias
    attn_weight = F.softmax(attn_weight, dim=-1)
    if dropout_p > 0:
        attn_weight = F.dropout(attn_weight, dropout_p, training=True)
    return attn_weight @ value


def scaled_dot_product_attention(
    query: Float[Tensor, "*#batch h_query target d_key"], 
    key: Float[Tensor, "*#batch h_key source d_key"], 
    value: Float[Tensor, "*#batch h_key source d_val"], 
    attn_mask: AttnMask | None = None, 
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    backends: list[Backend] | None = None,
    score_mod: ScoreMod | None = None
) -> Float[Tensor, "*#batch h_query target d_val"]:
    if backends and "naive" in backends:
        assert len(backends) == 1
        return naive_scaled_dot_product_attention(
            query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
        )
    if backends and "flex_attention" in backends:
        assert len(backends) == 1
        assert all(t.ndim == 4 for t in (query, key, value))
        if attn_mask is not None:
            assert isinstance(attn_mask, BlockMask)
        return compile_flex_attention()(
            query, key, value, score_mod=score_mod, block_mask=attn_mask, scale=scale, enable_gqa=enable_gqa
        )
    with sdpa_kernel([getattr(SDPBackend, backend.upper()) for backend in backends]) if backends else nullcontext():
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask, dropout_p, is_causal, scale=scale, enable_gqa=enable_gqa
        )


@dataclass
class AttentionCfg:
    num_heads: int | None = None
    dim_head: int | None = None
    qkv_bias: bool = True   # NOTE this is True for DiT & MAR but not for ViT
    qk_norm: Norm1dCfg | None = None
    attn_drop: float = 0.
    out_drop: float = 0.
    scale: float | None = None
    backends: list[Backend] | None = None


class Attention(nn.Module):
    def __init__(
        self, 
        dim: int, 
        num_heads: int | None = None,
        dim_head: int | None = None,
        qkv_bias: bool = False,
        qk_norm: Norm1dCfg | None = None,
        attn_drop: float = 0.,
        out_drop: float = 0.,
        scale: float | None = None,
        self_attn: bool = True, 
        kv_dim: int | None = None,
        backends: list[Backend] | None = None
    ):
        super().__init__()
        if dim_head is None:
            if num_heads is None:
                num_heads = 1
            assert dim % num_heads == 0
            dim_head = dim // num_heads
        else:
            assert dim % dim_head == 0
            num_heads = dim // dim_head
        self.inner_dim = dim_head * num_heads
        self.dim_head = dim_head
        self.heads = num_heads
        self.attn_drop = attn_drop
        self.scale = scale
        self.self_attn = self_attn
        self.backends = backends
        if backends and "flex_attention" in backends:
            # Compile flex attention at instantiation, not runtime
            compile_flex_attention()

        if self_attn:
            self.qkv = nn.Linear(dim, self.inner_dim * 3, bias=qkv_bias)
        else:
            self.q_linear = nn.Linear(dim, self.inner_dim, bias=qkv_bias)
            self.kv_linear = nn.Linear(kv_dim, self.inner_dim * 2, bias=qkv_bias)
        # TODO possibly make type of norm configurable
        self.q_norm = get_norm_1d(qk_norm, dim_head) if qk_norm is not None else nn.Identity()
        self.k_norm = get_norm_1d(qk_norm, dim_head) if qk_norm is not None else nn.Identity()
        self.proj = nn.Linear(self.inner_dim, dim)
        self.proj_drop = nn.Dropout(out_drop)
        self.register_buffer("k_cache", None, persistent=False)
        self.register_buffer("v_cache", None, persistent=False)

    @classmethod
    def from_config(
        cls: type["Attention"], 
        config: AttentionCfg, 
        dim: int, 
        self_attn: bool = True, 
        kv_dim: int | None = None
    ) -> "Attention":
        return cls(dim, self_attn=self_attn, kv_dim=kv_dim, **{f.name: getattr(config, f.name) for f in fields(config)})

    def free_cache(self) -> None:
        self.k_cache = None
        self.v_cache = None

    def forward(
        self, 
        x: Float[Tensor, "*target_batch target dim"], 
        z: Float[Tensor, "*source_batch source kv_dim"] | None = None,  # TODO can we do better than this for the shapes?
        # *#batch #head token source original shape
        attn_mask: BlockMask | Bool[Tensor, "..."] | Float[Tensor, "..."] | None = None,
        score_mod: ScoreMod | None = None,
        rel_pos_emb: PosEmbedding | None = None,
        # *#batch token original shape, while *batch for x, c can be empty and token dim with all tokens of batch elements in sparse format
        cache_mask: Bool[Tensor, "..."] | None = None,
        use_cache: bool = False,
        update_cache: bool = False
    ) -> Float[Tensor, "*target_batch target dim"]:
        """
        If cross attention (not self.self_attn) and cache_mask is given, it will be used for padded batching of queries
        """

        if self.self_attn:
            q, k, v = rearrange(self.qkv(x), "... n (e h d) -> e ... h n d", e=3, h=self.heads).unbind()
        else:
            assert z is not None, "No self_attn but no z for cross attention given"
            q = rearrange(self.q_linear(x), "... n (h d) -> ... h n d", h=self.heads)
            k, v = rearrange(self.kv_linear(z), "... n (e h d) -> e ... h n d", e=2, h=self.heads).unbind()

        q = self.q_norm(q)
        k = self.k_norm(k)

        if rel_pos_emb is not None:
            q = rel_pos_emb.modulate(q)
            k = rel_pos_emb.modulate(k)

        if use_cache:
            assert cache_mask is not None, "use_cache but cache_mask is None"
            cache_mask = repeat(cache_mask, "... n -> h ... n", h=self.heads)

            if self.self_attn:
                if update_cache:
                    # Cache keys and values according to mask
                    self.k_cache = rearrange(k, "... h n d -> h ... n d")[cache_mask]
                    self.v_cache = rearrange(v, "... h n d -> h ... n d")[cache_mask]
                else:
                    assert attn_mask is None, "attn_mask expected to be None if use cached keys and values"
                    assert self.k_cache is not None, "use_cache and not update_cache but k_cache is None"
                    # Compose keys and values from current ones and cache according to mask
                    k_new = torch.empty((*cache_mask.shape, self.dim_head), dtype=k.dtype, device=k.device)
                    k_new[cache_mask] = self.k_cache
                    k_new[~cache_mask] = k.flatten(0, 1)
                    k = rearrange(k_new, "h ... n d -> ... h n d")
                    
                    assert self.v_cache is not None, "use_cache and not update_cache but v_cache is None"
                    v_new = torch.empty((*cache_mask.shape, self.dim_head), dtype=v.dtype, device=v.device)
                    v_new[cache_mask] = self.v_cache
                    v_new[~cache_mask] = v.flatten(0, 1)
                    v = rearrange(v_new, "h ... n d -> ... h n d")

            if not update_cache:
                # This q_mask could also be precomputed once for all blocks
                # And sparse attention could possibly further speed this up
                batch_dims = cache_mask.shape[1:-1]
                num_cached = cache_mask[0].sum(-1)
                num_queries = cache_mask.size(-1) - num_cached
                max_queries = num_queries.max().item()
                if (num_queries == max_queries).all():
                    q_mask = None
                    q = q.unflatten(-2, batch_dims + (max_queries,))
                else:
                    q_mask = torch.arange(max_queries, device=num_cached.device).expand(*batch_dims, -1) < num_queries.unsqueeze(-1)
                    q_new: Tensor = torch.zeros((self.heads, *batch_dims, max_queries, self.dim_head), dtype=q.dtype, device=q.device)
                    q_new.masked_scatter_(q_mask[None, ..., None], q)
                    q = q_new
                q = rearrange(q, "h ... n d -> ... h n d")

        out = scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop if self.training else 0., 
            scale=self.scale, 
            backends=self.backends,
            score_mod=score_mod
        )

        out = rearrange(out, "... h n d -> ... n (h d)")
        if use_cache and not update_cache:
            if q_mask is None:
                out = out.flatten(0, -2)
            else:
                out = out[q_mask]
        out = self.proj(out)
        out = self.proj_drop(out)
        return out
