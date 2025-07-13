from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor
from torch.nn.functional import cosine_similarity

from spatialreasoners.denoising_model.flow import Flow

from . import Loss, LossCfg, register_loss


@dataclass
class CosineCfg(LossCfg):
    pass


@register_loss("cosine", CosineCfg)
class Cosine(Loss[CosineCfg]):
    def __init__(
        self,
        cfg: CosineCfg,
        flow: Flow
    ) -> None:
        assert flow.parameterization == "ut", \
            f"Cosine (velocity direction) loss expects flow (ut) \
                parameterization but got {flow.parameterization}"
        super().__init__(cfg, flow)

    def unweighted_loss(
        self,
        t: Float[Tensor, "*#batch #dim"],
        eps: Float[Tensor, "*batch dim"],
        x: Float[Tensor, "*batch dim"],
        z_t: Float[Tensor, "*batch dim"],
        mean_target: Float[Tensor, "*batch dim"],
        mean_pred: Float[Tensor, "*batch dim"],
        v_pred: Float[Tensor, "*#batch #dim"] | None,
        sigma_pred: Float[Tensor, "*#batch #dim"] | None,
    ) -> Float[Tensor, "*batch #dim"]:
        """NOTE expects feature dimension to be the last one"""
        return 1 - cosine_similarity(mean_pred, mean_target, dim=-1).unsqueeze(-1)
