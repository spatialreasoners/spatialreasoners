import warnings
from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from spatialreasoners.misc.tensor import unsqueeze_as

from .. import register_sampling_schedule
from .adaptive import Adaptive, AdaptiveCfg


@dataclass(kw_only=True)
class AdaptiveInverseCfg(AdaptiveCfg):
    div_threshold: float | None = None
    max_approx_error: float | None = 5.0e-4
    num_search_steps: int | None = None
    finished_threshold: float = 1.0e-3
    min_alpha: float = 1.0e-5
    min_step_size: float | None = None


@register_sampling_schedule("adaptive_inverse", AdaptiveInverseCfg)
class AdaptiveInverse(Adaptive[AdaptiveInverseCfg]):
    def _validate_config(self):
        assert (
            self.cfg.min_step_size is None or self.cfg.min_step_size > 0
        ), "min_step_size must be greater than zero"
        assert (
            self.cfg.num_search_steps is None or self.cfg.num_search_steps > 0
        ), "num_search_steps must be greater than zero"
        assert (
            self.cfg.min_alpha > 0
        ), "min_alpha must be greater than zero to avoid division by zero in kl_simple"

    def _init_batch_state(self):
        if self.cfg.min_alpha > self.cfg.alpha:
            warnings.warn(
                "Found alpha smaller than min_alpha, continuing with min_alpha"
            )
            self.cfg.alpha = self.cfg.min_alpha

    @property
    def finished_threshold(self) -> float:
        return self.cfg.finished_threshold

    def _calculate_next_t_and_denoise_mask(
        self,
        t: Float[Tensor, "batch_relevant total_patches"],
        sigma_theta: Float[Tensor, "batch_relevant total_patches"] | None = None,
    ) -> tuple[
        Float[Tensor, "batch total_patches"],  # t_next
        Bool[Tensor, "batch total_patches"],  # should_denoise_mask
    ]:
        assert sigma_theta is not None
        if self.cfg.div_threshold is None:
            # Compute the threshold such that every variable does a step of size >= min_step_size
            t_next_max = (1 - 1 / (self.cfg.max_steps - self.current_step)) * t
            divergence = self.model_flow.divergence_simple(
                sigma_theta,
                t.clamp_min(self.cfg.min_step_size),
                t_next_max.clamp_min(self.cfg.min_step_size),
                self.cfg.alpha,
            )
            threshold = torch.amax(divergence, dim=(-1))
        else:
            threshold = torch.tensor([self.cfg.div_threshold], device=t.device)

        t_next = self.model_flow.inverse_divergence_simple(
            sigma_theta,
            t,
            self.cfg.alpha,
            unsqueeze_as(threshold, t),
            self.cfg.max_approx_error,
            self.cfg.num_search_steps,
        )

        # ones for patches that are not finished
        should_denoise_mask = t > self.finished_threshold

        return t_next, should_denoise_mask
