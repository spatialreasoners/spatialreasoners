from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from ..sampling_schedule import SamplingSchedule, SamplingScheduleCfg


@dataclass(kw_only=True)
class AdaptiveCfg(SamplingScheduleCfg, ABC):
    max_steps: int
    alpha: float
    
    @property
    def num_noise_level_values(self) -> None:
        return None # there are infinite possible noise levels
    
T = TypeVar("T", bound=AdaptiveCfg)
    
class Adaptive(SamplingSchedule[T], ABC):
    @property
    @abstractmethod
    def finished_threshold(self) -> float:
        pass
    
    @abstractmethod
    def _calculate_next_t_and_denoise_mask(
        self,
        t: Float[Tensor, "batch total_patches"] | None = None,
        sigma_theta: Float[Tensor, "batch total_patches"] | None = None,
    ) -> tuple[
        Float[Tensor, "batch total_patches"], # t_next
        Bool[Tensor, "batch total_patches"],  # is_selectable_mask
    ]:
        pass
    
    def _get_next_t_and_denoise_mask(
        self,
        t: Float[Tensor, "batch total_patches"] | None = None,
        sigma_theta: Float[Tensor, "batch total_patches"] | None = None,
    ) -> tuple[
        Float[Tensor, "batch total_patches"], # t_next
        Bool[Tensor, "batch total_patches"],  # is_selectable_mask
    ]:
        if self.current_step >= self.cfg.max_steps - 1:
            t_next = torch.zeros_like(t)
            denoise_mask = t > self.finished_threshold
        else:
            t_next, denoise_mask = self._calculate_next_t_and_denoise_mask(t, sigma_theta)
            
        self.last_t_next = t_next
        return t_next, denoise_mask
    
    @property
    def is_unfinished_mask(self) -> Bool[Tensor, "batch"]:
        if self.current_step == 0:
            return torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        
        should_denoise_mask = self.last_t_next > self.finished_threshold # ones for patches that are not finished
        return should_denoise_mask.any(dim=1) # ones for batches that are not finished