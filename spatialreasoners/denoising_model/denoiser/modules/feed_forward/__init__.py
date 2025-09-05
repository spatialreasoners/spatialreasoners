from spatialreasoners.registry import Registry
from .feed_forward import FeedForward, FeedForwardCfg


_feed_forward_registry = Registry(FeedForward, FeedForwardCfg)


def get_feed_forward(
    cfg: FeedForwardCfg, 
    d_in: int,
    d_hid: int | None = None,
    d_out: int | None = None
) -> FeedForward:
    return _feed_forward_registry.build(cfg, d_in, d_hid, d_out)


register_feed_forward = _feed_forward_registry.register


from .mlp import Mlp, MlpCfg
from .lightningdit_swiglu import LightningDiTSwiGLU, LightningDiTSwiGLUCfg
from .timm_swiglu import TimmSwiGLU, TimmSwiGLUCfg


__all__ = [
    "FeedForward", "FeedForwardCfg",
    "Mlp", "MlpCfg",
    "LightningDiTSwiGLU", "LightningDiTSwiGLUCfg",
    "TimmSwiGLU", "TimmSwiGLUCfg",
    "get_feed_forward",
    "register_feed_forward"
]
