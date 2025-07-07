from dataclasses import dataclass

from jaxtyping import Float
from torch.nn.functional import silu
from torch import nn, Tensor

from . import FeedForward, FeedForwardCfg, register_feed_forward


@dataclass
class LightningDiTSwiGLUCfg(FeedForwardCfg):
    bias: bool = True


@register_feed_forward("lightningdit_swiglu", LightningDiTSwiGLUCfg)
class LightningDiTSwiGLU(FeedForward[LightningDiTSwiGLUCfg]):
    def __init__(
        self,
        cfg: LightningDiTSwiGLUCfg,
        d_in: int,
        d_hid: int | None = None,
        d_out: int | None = None
    ):
        super(LightningDiTSwiGLU, self).__init__(cfg, d_in, d_hid, d_out)
        self.w12 = nn.Linear(self.d_in, 2 * self.d_hid, bias=cfg.bias)
        self.w3 = nn.Linear(self.d_hid, self.d_out, bias=cfg.bias)

    def forward(
        self, 
        input: Float[Tensor, "*batch d_in"]
    ) -> Float[Tensor, "*batch d_out"]:
        x12: Tensor = self.w12(input)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = silu(x1) * x2
        return self.w3(hidden)
