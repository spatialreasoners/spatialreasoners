from dataclasses import dataclass

from jaxtyping import Float
import numpy as np
from numpy import ndarray

from . import BetaSchedule, BetaScheduleCfg, register_beta_schedule


@dataclass
class LinearCfg(BetaScheduleCfg):
    num_ref_timesteps: int = 1000
    start: float = 1e-4
    end: float = 2e-2


@register_beta_schedule("linear", LinearCfg)
class Linear(BetaSchedule[LinearCfg]):
    def __call__(
        self,
        num_timesteps: int
    ) -> Float[ndarray, "num_timesteps"]:
        scale = 1000 / num_timesteps
        return np.linspace(
            scale * self.cfg.start, 
            scale * self.cfg.end, 
            num_timesteps, 
            dtype=np.float64
        )
