from collections.abc import Sequence
from dataclasses import dataclass

from jaxtyping import Float
from torch import nn, Tensor

from . import FeedForward, FeedForwardCfg, register_feed_forward
from ...activations import Activation, get_activation
from ...modules.norm_1d import Norm1dCfg, get_norm_1d


@dataclass
class MlpCfg(FeedForwardCfg):
    activation: Activation = "gelu"
    norm: Norm1dCfg | None = None
    bias: bool | Sequence[bool] = True
    drop: float | Sequence[float] = 0.


@register_feed_forward("mlp", MlpCfg)
class Mlp(FeedForward[MlpCfg]):
    def __init__(
        self,
        cfg: MlpCfg,
        d_in: int,
        d_hid: int | None = None,
        d_out: int | None = None
    ):
        super(Mlp, self).__init__(cfg, d_in, d_hid, d_out)
        bias = tuple(cfg.bias) if isinstance(cfg.bias, Sequence) else 2 * (cfg.bias,)
        drop = tuple(cfg.drop) if isinstance(cfg.drop, Sequence) else 2 * (cfg.drop,)

        self.fc1 = nn.Linear(self.d_in, self.d_hid, bias=bias[0])
        self.act = get_activation(cfg.activation)
        self.drop1 = nn.Dropout(drop[0])
        self.norm = get_norm_1d(cfg.norm, self.d_hid) if cfg.norm is not None else nn.Identity()
        self.fc2 = nn.Linear(self.d_hid, self.d_out, bias=bias[1])
        self.drop2 = nn.Dropout(drop[1])

    def forward(
        self, 
        input: Float[Tensor, "*batch d_in"]
    ) -> Float[Tensor, "*batch d_out"]:
        x = self.fc1(input)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
