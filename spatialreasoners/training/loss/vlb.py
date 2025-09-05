from dataclasses import dataclass
from math import log

import torch
from jaxtyping import Float
from torch import Tensor

from . import Loss, LossCfg, register_loss


@dataclass
class VLBCfg(LossCfg):
    detach_mean: bool = True
    time_step_size: float = 1.e-3


@register_loss("vlb", VLBCfg)
class VLB(Loss[VLBCfg]):
    def unweighted_loss(
        self,
        t: Float[Tensor, "*#batch"],
        eps: Float[Tensor, "*batch"],
        x: Float[Tensor, "*batch"],
        z_t: Float[Tensor, "*batch"],
        mean_target: Float[Tensor, "*batch"],
        mean_pred: Float[Tensor, "*batch"],
        v_pred: Float[Tensor, "*#batch"],
        sigma_pred: Float[Tensor, "*#batch"] | None,
    ) -> Float[Tensor, "*batch"]:
        t_next = t - self.cfg.time_step_size
        nll_mask = t_next <= 0
        t_next[nll_mask] = 0
        p_theta = self.flow.conditional_p(
            mean_pred.detach() if self.cfg.detach_mean else mean_pred,
            z_t, 
            t, 
            t_next, 
            alpha=1, 
            v_theta=v_pred
        )
        q = self.flow.conditional_q(x, eps, t, t_next, alpha=1)
        kl = q.kl(p_theta)
        nll = -p_theta.discretized_log_likelihood(x)
        vlb_loss = torch.where(nll_mask, nll, kl) / log(2.0)
        return vlb_loss
