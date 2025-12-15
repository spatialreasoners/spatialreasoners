from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from spatialreasoners.denoising_model.flow.diagonal_gaussian import DiagonalGaussian

from . import Loss, LossCfg, register_loss


@dataclass
class NLLCfg(LossCfg):
    detach_mean: bool = True


@register_loss("nll", NLLCfg)
class NLL(Loss[NLLCfg]):
    def unweighted_loss(
        self,
        t: Float[Tensor, "*#batch"],
        eps: Float[Tensor, "*batch"],
        x: Float[Tensor, "*batch"],
        z_t: Float[Tensor, "*batch"],
        mean_target: Float[Tensor, "*batch"],
        mean_pred: Float[Tensor, "*batch"],
        v_pred: Float[Tensor, "*#batch"] | None,
        sigma_pred: Float[Tensor, "*#batch"],
    ) -> Float[Tensor, "*batch"]:
        pred_theta = DiagonalGaussian(
            mean_pred.detach() if self.cfg.detach_mean else mean_pred, 
            std=sigma_pred
        )
        sigma_loss = pred_theta.nll(mean_target)
        return sigma_loss
