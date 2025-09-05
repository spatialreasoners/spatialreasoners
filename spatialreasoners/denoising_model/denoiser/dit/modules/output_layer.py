from dataclasses import dataclass, field

from jaxtyping import Float
from torch import Tensor, nn

from ...modules.norm_1d import get_norm_1d, Norm1dCfg, LayerNormCfg
from ...utils import modulate


@dataclass
class OutputLayerCfg:
    # Used in original DiT paper
    norm: Norm1dCfg = field(
        default_factory=lambda: LayerNormCfg(eps=1.e-6, elementwise_affine=False)
    )


class OutputLayer(nn.Module):
    def __init__(
        self, 
        d_in: int,
        d_out: int,
        d_c: int,
        norm: Norm1dCfg = LayerNormCfg(eps=1.e-6, elementwise_affine=False)
    ):
        super().__init__()
        # NOTE name norm_final (instead of norm) for checkpoint compatibility
        self.norm_final = get_norm_1d(norm, d_in)
        self.linear = nn.Linear(d_in, d_out, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_c, 2 * d_in, bias=True)
        )

    @classmethod
    def from_config(
        cls: type["OutputLayer"], 
        config: OutputLayerCfg, 
        d_in: int,
        d_out: int,
        d_c: int,
    ) -> "OutputLayer":
        return cls(d_in, d_out, d_c, **config.__dict__)

    def forward(
        self, 
        x: Float[Tensor, "*batch d_in"], 
        c: Float[Tensor, "*#batch d_c"]
    ) -> Float[Tensor, "*batch d_out"]:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        # NOTE splitting dim into first patch size and then out channels 
        # for compatibility with DiT checkpoints
        x = self.linear(x)
        return x
