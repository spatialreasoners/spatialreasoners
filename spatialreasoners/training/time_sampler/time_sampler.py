from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, Optional, TypeVar

import torch
from jaxtyping import Bool, Float
from torch import Tensor, device as torch_device

from spatialreasoners.variable_mapper import VariableMapper

from .time_weighting import TimeWeightingCfg, get_time_weighting
from .normalization_weighing import NormalizationWeighingCfg, get_normalization_weighing, InverseHistogramPdfWeighingCfg

@dataclass
class TimeSamplerCfg:
    time_weighting: Optional["TimeWeightingCfg"] = None
    normalization_weighing: Optional["NormalizationWeighingCfg"] = field(
        default_factory=InverseHistogramPdfWeighingCfg
    )
    add_zeros: bool = False


T = TypeVar("T", bound=TimeSamplerCfg)


class TimeSampler(Generic[T], ABC):
    def __init__(
        self,
        cfg: T,
        variable_mapper: VariableMapper,
    ) -> None:
        self.cfg = cfg
        self.variable_mapper = variable_mapper
        self.normalization_weights = None
        self.time_weighting = (
            get_time_weighting(cfg.time_weighting)
            if cfg.time_weighting is not None
            else None
        )
        
    @property
    def num_variables(self) -> int:
        return self.variable_mapper.num_variables
    
    def get_normalization_weights(
        self, 
        t: Float[Tensor, "*batch"],
        mask: Bool[Tensor, "*#batch"] | None = None
    ) -> Float[Tensor, "*#batch"]:
        if self.cfg.normalization_weighing is not None and self.normalization_weights is None:
            
            self.normalization_weights =(
                get_normalization_weighing(self.cfg.normalization_weighing, self.get_time, t.device)
                if self.cfg.normalization_weighing is not None
                else None
            )
        
        if self.normalization_weights is None:
            return torch.ones_like(t)
        
        return self.normalization_weights(t, mask)

    @abstractmethod
    def get_time(
        self, 
        batch_size: int, 
        num_samples: int = 1,
        device: torch_device | str = "cpu",
    ) -> tuple[
        Float[Tensor, "batch sample num_variables"], # t
        Bool[Tensor, "batch #sample num_variables"] | None # mask
    ]:
        pass

    def __call__(
        self, 
        batch_size: int, 
        num_samples: int = 1,
        device: torch_device | str = "cpu",
    ) -> tuple[
        Float[Tensor, "batch sample num_variables"],        # t
        Float[Tensor, "batch #sample num_variables"],       # weights
        Bool[Tensor, "batch #sample num_variables"] | None, # mask
    ]:
        t, mask = self.get_time(batch_size, num_samples, device)
        weights = self.get_normalization_weights(t, mask)
        weights = weights.to(device)

        if self.time_weighting is not None:
            weights.mul_(self.time_weighting(t))

        if self.cfg.add_zeros:
            # TODO this is so ugly
            t = t.flatten(-2).contiguous()
            weights = weights.flatten(-2).contiguous()
            zero_ratios = torch.rand((batch_size,), device=device)
            zero_mask = torch.linspace(1/self.num_variables, 1, self.num_variables, device=device) < zero_ratios[:, None, None]
            idx = torch.rand_like(t).argsort(dim=-1)
            t[zero_mask] = 0
            weights[zero_mask] = 0
            if mask is not None:
                mask = mask.flatten(-2).contiguous()
                mask[zero_mask] = False
                mask = mask.gather(-1, idx).reshape(batch_size, num_samples, -1)
            t = t.gather(-1, idx).reshape(batch_size, num_samples, -1)
            weights = weights.gather(-1, idx).reshape(batch_size, num_samples, -1)
                        
        return t, weights, mask
