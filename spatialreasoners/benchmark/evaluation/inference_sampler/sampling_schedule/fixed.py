from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from . import SamplingSchedule, SamplingScheduleCfg, register_sampling_schedule


@dataclass(kw_only=True)
class FixedCfg(SamplingScheduleCfg):
    num_steps: int
    timestep_shift: float | int = 1
    
    @property
    def num_noise_level_values(self) -> int:
        return self.num_steps + 1 # +1 for the initial noise level of 1
    

@register_sampling_schedule("fixed", FixedCfg)
class Fixed(SamplingSchedule[FixedCfg]):
    def _validate_config(self):
        assert self.cfg.num_steps > 0, "num_steps must be greater than 0"
    
    def _init_batch_state(self):
        self.noise_levels = torch.linspace(1, 0, self.cfg.num_steps + 1, device=self.device)
        if self.cfg.timestep_shift != 1:
            self.noise_levels = self.cfg.timestep_shift * self.noise_levels \
                / (1 + (self.cfg.timestep_shift - 1) * self.noise_levels)
    
    def _get_next_t_and_denoise_mask(
        self,
        t: Float[Tensor, "batch_relevant total_patches"],
        sigma_theta: Float[Tensor, "batch_relevant total_patches"] | None = None,
    ) -> tuple[
        Float[Tensor, "batch_relevant total_patches"],
        Bool[Tensor, "batch_relevant total_patches"],
    ]:  
        next_noise_level = self.noise_levels[self.current_step + 1]
        next_t = torch.full(t.shape, next_noise_level, device=self.device)
        
        if self.mask is not None:
            next_t = next_t * self.mask
        
        should_denoise = ~torch.isclose(next_t, torch.tensor(0.0, device=self.device))
                
        return next_t, should_denoise
    
    @property
    def is_unfinished_mask(self) -> Bool[Tensor, "batch"]:
        should_continue = self.current_step < self.cfg.num_steps
        return torch.full((self.batch_size,), should_continue, dtype=torch.bool, device=self.device)
