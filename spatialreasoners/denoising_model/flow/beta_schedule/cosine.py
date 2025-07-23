from dataclasses import dataclass

from jaxtyping import Float
import numpy as np
from numpy import ndarray

from . import BetaSchedule, BetaScheduleCfg, register_beta_schedule


@dataclass
class CosineCfg(BetaScheduleCfg):
    max_beta: float = 0.999
    skew: float = 0.008


@register_beta_schedule("cosine", CosineCfg)
class Cosine(BetaSchedule[CosineCfg]):
    def __call__(
        self,
        num_timesteps: int
    ) -> Float[ndarray, "num_timesteps"]:
        def f(t, T, s):
            return np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2

        betas = []
        for t in range(num_timesteps):
            alpha_bar_t = f(t + 1, num_timesteps, self.cfg.skew)
            alpha_bar_t_1 = f(t, num_timesteps, self.cfg.skew)
            betas_t = 1 - alpha_bar_t / alpha_bar_t_1
            betas.append(min(betas_t, self.cfg.max_beta))
        return np.array(betas)
