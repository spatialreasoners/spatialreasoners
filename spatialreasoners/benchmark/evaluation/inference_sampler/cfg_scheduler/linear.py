from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor

from . import CFGScheduler, CFGSchedulerCfg, register_cfg_scheduler


@dataclass
class LinearCfg(CFGSchedulerCfg):
    threshold: float | None = 1.e-5


@register_cfg_scheduler("linear", LinearCfg)
class Linear(CFGScheduler[LinearCfg]):
    def __call__(
        self,
        t: Float[Tensor, "*batch dim"],
    ) -> Float[Tensor, "*batch"]:
        if self.cfg.scale == 1:
            return torch.ones_like(t[..., 0])
        num_given = t.shape[-1] - t.sum(dim=-1) if self.cfg.threshold is None \
            else (t <= self.cfg.threshold).sum(dim=-1)
        return 1 + (self.cfg.scale - 1) / t.shape[-1] * num_given
