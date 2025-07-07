from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from . import SamplingSchedule, SamplingScheduleCfg, register_sampling_schedule


@dataclass(kw_only=True)
class RasterCfg(SamplingScheduleCfg):
    num_steps: int
    
    @property
    def num_noise_level_values(self) -> int:
        return self.num_steps + 1 # +1 for the initial noise level of 1
    

@register_sampling_schedule("raster", RasterCfg)
class Raster(SamplingSchedule[RasterCfg]):
    def _validate_config(self):
        assert self.cfg.num_steps > 0, "num_steps must be greater than 0"
    
    def _init_batch_state(self):
        self.noise_levels = torch.linspace(1, 0, self.cfg.num_steps + 1, device=self.device)
    
    def _get_next_t_and_denoise_mask(
        self,
        t: Float[Tensor, "batch_relevant num_variables"],
        sigma_theta: Float[Tensor, "batch_relevant num_variables"] | None = None,
    ) -> tuple[
        Float[Tensor, "batch_relevant num_variables"],
        Bool[Tensor, "batch_relevant num_variables"],
    ]:
        variable_idx = self.current_step // self.cfg.num_steps
        noise_level_step = self.current_step % self.cfg.num_steps
        next_t = t.clone()
        next_t[:, variable_idx] = self.noise_levels[noise_level_step + 1]
        
        if self.mask is not None:
            next_t = next_t * self.mask
        
        should_denoise = ~torch.isclose(next_t, torch.tensor(0.0, device=self.device))
                
        return next_t, should_denoise
    
    @property
    def is_unfinished_mask(self) -> Bool[Tensor, "batch"]:
        should_continue = self.current_step < (self.cfg.num_steps * self.variable_mapper.num_variables)
        return torch.full((self.batch_size,), should_continue, dtype=torch.bool, device=self.device)
