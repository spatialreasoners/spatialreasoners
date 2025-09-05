from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float
from torch import Tensor
from torch.nn.functional import softmax

from .. import register_sampling_schedule
from .adaptive import Adaptive, AdaptiveCfg


@dataclass(kw_only=True)
class AdaptiveSoftmaxCfg(AdaptiveCfg):
    scale: float = 0.1
    max_clip_iter: int = 8
    finished_threshold: float | None = None  # If None, use 1 / (max_steps)
    epsion: float = 1.0e-6

    @property
    def num_noise_level_values(self) -> None:
        return None  # there are infinite possible noise levels


@register_sampling_schedule("adaptive_softmax", AdaptiveSoftmaxCfg)
class AdaptiveSoftmax(Adaptive[AdaptiveSoftmaxCfg]):
    def _validate_config(self):
        assert self.cfg.scale > 0, "scale must be greater than zero"
        assert (
            self.cfg.finished_threshold is None or 0 <= self.cfg.finished_threshold <= 1
        ), "finished_threshold must be in [0, 1]"

    def _init_batch_state(self):
        if self.cfg.finished_threshold is None:
            self._finished_threshold = 1 / self.cfg.max_steps

        else:
            self._finished_threshold = self.cfg.finished_threshold

    @property
    def finished_threshold(self) -> float:
        return self._finished_threshold

    def _calculate_next_t_and_denoise_mask(
        self,
        t: Float[Tensor, "batch total_patches"],
        sigma_theta: Float[Tensor, "batch total_patches"],
    ) -> tuple[
        Float[Tensor, "batch total_patches"], # t_next
        Bool[Tensor, "batch total_patches"],  # is_selectable_mask
    ]:
        assert t is not None
        assert sigma_theta is not None

        min_step_factor = 1 - 1 / (self.cfg.max_steps - self.current_step)
        t_next = min_step_factor * t

        kl_div = self.model_flow.divergence_simple(
            sigma_theta, t, t_next, self.cfg.alpha
        )

        t_next_norm = t_next.sum(dim=-1, keepdim=True)
        t_next = softmax(
            torch.log(t_next + self.cfg.epsion) + self.cfg.scale * kl_div, dim=-1
        ).mul_(t_next_norm)

        # Normalize such that t_next <= t while preserving the t_next_norm
        for _ in range(self.cfg.max_clip_iter):
            t_next.clamp_max_(t)
            sub_idx = (t_next == t).long()
            sub_norms = torch.zeros(
                (t_next_norm.shape[0], 2), dtype=t.dtype, device=t.device
            )
            sub_norms.scatter_add_(1, sub_idx, t_next)
            if (sub_norms[:, 1] < 1.0e-5).all():
                break
            t_next_sub_norm = t_next_norm - sub_norms[:, 1]
            t_next.mul_(t_next_sub_norm / sub_norms[:, 0])

        t_next = t_next.clamp_max_(t)
        should_denoise_mask = (
            t > self.finished_threshold
        )  # ones for patches that are not finished

        return t_next, should_denoise_mask
