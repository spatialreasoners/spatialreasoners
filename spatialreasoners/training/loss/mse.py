from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor
from torch.nn.functional import mse_loss

from . import Loss, LossCfg, register_loss


@dataclass
class MSECfg(LossCfg):
    pass


@register_loss("mse", MSECfg)
class MSE(Loss[MSECfg]):
    def unweighted_loss(
        self,
        t: Float[Tensor, "*#batch"],
        eps: Float[Tensor, "*batch"],
        x: Float[Tensor, "*batch"],
        z_t: Float[Tensor, "*batch"],
        mean_target: Float[Tensor, "*batch"],
        mean_pred: Float[Tensor, "*batch"],
        v_pred: Float[Tensor, "*#batch"] | None,
        sigma_pred: Float[Tensor, "*#batch"] | None,
    ) -> Float[Tensor, "*batch"]:
        return mse_loss(mean_pred, mean_target, reduction="none")
