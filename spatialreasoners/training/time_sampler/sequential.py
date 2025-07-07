from dataclasses import dataclass, field
from math import ceil

import torch
from jaxtyping import Bool, Float
from torch import Tensor, device

from spatialreasoners.variable_mapper import VariableMapper

from . import register_time_sampler
from .mask_ratio_sampler import (
    MaskRatioSamplerCfg,
    TruncNormCfg,
    get_mask_ratio_sampler,
)
from .two_stage_time_sampler import TwoStageTimeSampler, TwoStageTimeSamplerCfg


@dataclass
class SequentialCfg(TwoStageTimeSamplerCfg):
    mask_ratio_sampler: MaskRatioSamplerCfg = field(default_factory=TruncNormCfg)
    mask_full_noise: bool = False
    num_parallel: int = 1   # ignored if mask_full_noise == False
    # whether to sample the same or independent order (given by t == 0 and possibly t == 1) 
    # for all samples w.r.t. one batch element
    independent_order: bool = False
    # whether to sample independent times for all patches of one batch element or a shared single one
    # TODO possibly instead allow another time_sampler for this?
    independent_time: bool = True


@register_time_sampler("sequential", SequentialCfg)
class Sequential(TwoStageTimeSampler[SequentialCfg]):
    """
    NOTE 
    will always sample the same number of clean and noisy elements:
    if mask_full_noise: 
        Use mask_ratio_sampler for sampling the ratio of 1s (full noise) 
        and use num_parallel as number of elements with 0 < t < 1
    else:
        use mask_ratio_sampler for sampling ratio of elements with 0 < t
        and ignore num_parallel
    """

    def __init__(
        self,
        cfg: SequentialCfg,
        variable_mapper: VariableMapper,
    ) -> None:
        assert cfg.num_parallel >= 1
        super().__init__(cfg, variable_mapper)
        
        self.mask_ratio_sampler = get_mask_ratio_sampler(cfg.mask_ratio_sampler)
        if cfg.mask_full_noise:
            assert self.num_variables % cfg.num_parallel == 0

    def get_time(
        self, 
        batch_size: int, 
        num_samples: int = 1,
        device: device | str = "cpu",
    ) -> tuple[
        Float[Tensor, "batch sample num_variables"], # t
        Bool[Tensor, "batch #sample num_variables"]  # mask
    ]:
        mask_ratio = self.mask_ratio_sampler()
        num_masked = ceil(mask_ratio * self.num_variables)
        t = torch.zeros((batch_size, num_samples, self.num_variables), device=device)
        mask = torch.zeros((batch_size, 1, self.num_variables), dtype=torch.bool, device=device)
        mask[..., :num_masked] = True
        if self.cfg.mask_full_noise:
            t[..., :num_masked] = 1
            t[..., -self.cfg.num_parallel:] = self.scalar_time_sampler(
                (batch_size, num_samples, self.cfg.num_parallel if self.cfg.independent_time else 1), 
                device=device
            )
            mask[..., -self.cfg.num_parallel:] = True
        else:
            t[..., :num_masked] = self.scalar_time_sampler(
                (batch_size, num_samples, num_masked if self.cfg.independent_time else 1), 
                device=device
            )
        # Batched shuffle
        if self.cfg.independent_order:
            idx = torch.rand((batch_size, num_samples, self.num_variables), device=device).argsort()
            mask = mask.expand_as(idx).gather(-1, idx).reshape(batch_size, num_samples, self.num_variables)
        else:
            idx = torch.rand((batch_size, 1, self.num_variables), device=device).argsort()
            mask = mask.gather(-1, idx).reshape(batch_size, 1, self.num_variables)
            idx = idx.expand(-1, num_samples, -1)

        t = t.gather(-1, idx).reshape(batch_size, num_samples, self.num_variables)
        return t, mask
    
    def get_normalization_weights(
        self, 
        t: Float[Tensor, "*batch"],
        mask: Bool[Tensor, "*#batch"]
    ) -> Float[Tensor, "*#batch"]:
        if self.cfg.mask_full_noise:
            mask = mask.logical_and(t < 1)  # TODO find a better alternative?
        return torch.where(mask, mask.numel() / mask.sum(), 0.0)
