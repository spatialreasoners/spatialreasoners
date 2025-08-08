from dataclasses import dataclass

from jaxtyping import Float
from torch import nn, Tensor

from spatialreasoners.misc.utils import to_tuple

from ...activations import Activation, get_activation
from ..norm_1d import Norm1dCfg, get_norm_1d
from . import FeedForward, FeedForwardCfg, register_feed_forward


@dataclass
class TimmSwiGLUCfg(FeedForwardCfg):
    act_layer: Activation = "silu"
    norm_layer: Norm1dCfg | None = None
    bias: bool | tuple[bool, bool] = True
    drop: float | tuple[float | float] = 0.


@register_feed_forward("timm_swiglu", TimmSwiGLUCfg)
class TimmSwiGLU(FeedForward[TimmSwiGLUCfg]):
    def __init__(
        self,
        cfg: TimmSwiGLUCfg,
        d_in: int,
        d_hid: int | None = None,
        d_out: int | None = None
    ):
        super(TimmSwiGLU, self).__init__(cfg, d_in, d_hid, d_out)
        bias = to_tuple(cfg.bias, 2)
        drop = to_tuple(cfg.drop, 2)
        self.fc1_g = nn.Linear(self.d_in, self.d_hid, bias=bias[0])
        self.fc1_x = nn.Linear(self.d_in, self.d_hid, bias=bias[0])
        self.act = get_activation(cfg.act_layer)
        self.drop1 = nn.Dropout(drop[0])
        self.norm = get_norm_1d(cfg.norm_layer, self.d_hid) if cfg.norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(self.d_hid, self.d_out, bias=bias[1])
        self.drop2 = nn.Dropout(drop[1])
        self.init_weights() # TODO move this call somewhere else (e.g. in block)

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        if self.fc1_g.bias is not None:
            nn.init.ones_(self.fc1_g.bias)
        nn.init.normal_(self.fc1_g.weight, std=1e-6)

    def forward(
        self, 
        input: Float[Tensor, "*batch d_in"]
    ) -> Float[Tensor, "*batch d_out"]:
        x_gate = self.fc1_g(input)
        x = self.fc1_x(input)
        x = self.act(x_gate) * x
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
