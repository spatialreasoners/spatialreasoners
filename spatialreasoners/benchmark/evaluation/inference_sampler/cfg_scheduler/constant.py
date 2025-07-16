from dataclasses import dataclass

from jaxtyping import Float
import torch
from torch import Tensor

from . import CFGScheduler, CFGSchedulerCfg, register_cfg_scheduler


@dataclass
class ConstantCfg(CFGSchedulerCfg):
    t_start: float | int = 1


@register_cfg_scheduler("constant", ConstantCfg)
class Constant(CFGScheduler[ConstantCfg]):
    def __call__(
        self,
        t: Float[Tensor, "*batch dim"],
    ) -> Float[Tensor, "*batch"]:
        t_mean = t.mean(dim=-1)
        return torch.where(t_mean <= self.cfg.t_start, self.cfg.scale, 1)
