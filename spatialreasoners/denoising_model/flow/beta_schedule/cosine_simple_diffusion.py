from dataclasses import dataclass

import numpy as np
from jaxtyping import Float
from numpy import ndarray

from . import BetaSchedule, BetaScheduleCfg, register_beta_schedule


@dataclass
class CosineSimpleDiffusionCfg(BetaScheduleCfg):
    logsnr_min: float = -15.0
    logsnr_max: float = 15.0
    shifted: float = 0.125
    interpolated: bool = False


@register_beta_schedule("cosine_simple_diffusion", CosineSimpleDiffusionCfg)
class CosineSimpleDiffusion(BetaSchedule[CosineSimpleDiffusionCfg]):    
    def __call__(
        self,
        num_timesteps: int
    ) -> Float[ndarray, "num_timesteps"]:
        """
        Computes beta schedule based on Simple Diffusion's cosine logSNR parameterization.

        Args:
            num_timesteps: number of timesteps
            logsnr_min: minimum log SNR (usually ~ -20)
            logsnr_max: maximum log SNR (usually ~ 20)
            shifted: shift factor for resolution changes (set to 1.0 for none)
            interpolated: interpolate between original and shifted logSNRs

        Returns:
            betas: Tensor of shape [num_timesteps] with beta_t values
        """
        t_min = np.arctan(np.exp(-0.5 * np.array(self.cfg.logsnr_max, dtype=np.float64)))
        t_max = np.arctan(np.exp(-0.5 * np.array(self.cfg.logsnr_min, dtype=np.float64)))

        t = np.linspace(0, 1, num_timesteps, dtype=np.float64)
        logsnr = -2 * np.log(np.tan(t_min + t * (t_max - t_min)))

        if self.cfg.shifted != 1.0:
            shifted_logsnr = logsnr + 2 * np.log(np.array(self.cfg.shifted, dtype=np.float64))
            logsnr = t * logsnr + (1 - t) * shifted_logsnr if self.cfg.interpolated else shifted_logsnr

        # Compute alpha_cumprod (ᾱₜ) from logSNR
        alpha_cumprod = 1 / (1 + np.exp(-logsnr))

        # Compute alpha_t from ᾱₜ
        alpha_cumprod_prev = np.concatenate([np.array([1.0], dtype=np.float64), alpha_cumprod[:-1]])
        alpha_t = alpha_cumprod / alpha_cumprod_prev

        # Final beta schedule
        beta_t = 1.0 - alpha_t
        return beta_t