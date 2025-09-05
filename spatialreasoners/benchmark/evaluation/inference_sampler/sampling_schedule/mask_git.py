from dataclasses import dataclass
from math import cos, floor, pi

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from . import SamplingSchedule, SamplingScheduleCfg, register_sampling_schedule


@dataclass
class MaskGITCfg(SamplingScheduleCfg):
    num_steps_per_group: int # also the number of discrete steps for OriginalDiffusion
    num_groups: int

    @property
    def num_noise_level_values(self) -> int:
        # +1 for the initial noise level of 1
        return self.num_groups * self.num_steps_per_group + 1


@register_sampling_schedule("mask_git", MaskGITCfg)
class MaskGIT(SamplingSchedule[MaskGITCfg]):
    def _validate_config(self):
        assert (
            self.cfg.num_steps_per_group > 0
        ), "num_steps_per_group must be greater than 0"
        assert self.cfg.num_groups > 0, "num_groups must be greater than 0"
        assert self.mask is None, "mask currently not supported"

    def _init_batch_state(self):
        # NOTE interprets max_steps as num_iter in MAR, i.e., number of "sequential denoising groups"
        self.num_steps_per_group = self.cfg.num_steps_per_group
        self.group_counter = 0
        self.in_group_counter = 0

        self.variable_schedule = torch.linspace(
            1, 0, steps=self.cfg.num_steps_per_group + 1, device=self.device
        )
        self.patch_order = torch.rand(
            (self.batch_size, self.num_patches), device=self.device
        ).argsort()
        self.cur_clean = 0  # current number of unmasked, i.e., clean patches
        self.is_unfinished_patch = torch.ones(
            (self.batch_size, self.num_patches), dtype=torch.bool, device=self.device
        )

        self.mask_len = self.calculate_mask_len()
        self.next_clean = self.num_patches - self.mask_len
        self.group_patch_idx = self.patch_order[:, 0 : self.next_clean]

    def _init_new_group(self):
        self.group_counter += 1
        self.in_group_counter = 0

        self.cur_clean = self.next_clean
        self.is_unfinished_patch = torch.ones(
            (self.batch_size, self.num_patches), dtype=torch.bool, device=self.device
        )

        if self.group_counter < self.cfg.num_groups - 1:
            # Compute number of masked patches for next iteration
            self.mask_len = self.calculate_mask_len()
            self.next_clean = self.num_patches - self.mask_len
        else:
            self.next_clean = self.num_patches

        self.group_patch_idx = self.patch_order[:, self.cur_clean : self.next_clean]

    def calculate_mask_len(self) -> int:
        mask_ratio = cos(pi / 2.0 * (self.group_counter + 1) / self.cfg.num_groups)
        mask_len = floor(self.num_patches * mask_ratio)

        return max(1, min(self.num_patches - self.cur_clean, mask_len))

    def _get_next_t_and_denoise_mask(
        self,
        t: Float[Tensor, "batch_relevant total_patches"],
        sigma_theta: Float[Tensor, "batch_relevant total_patches"] | None = None,
    ) -> tuple[
        Float[Tensor, "batch_relevant total_patches"],
        Bool[Tensor, "batch_relevant total_patches"],
    ]:

        # Check if group finished
        if self.in_group_counter + 1 > self.num_steps_per_group:
            self.is_unfinished_patch.scatter_(1, self.group_patch_idx, value=False)
            self._init_new_group()

        next_t = t.scatter(
            1,
            self.group_patch_idx,
            self.variable_schedule[self.in_group_counter + 1].expand_as(
                self.group_patch_idx
            ),
        )

        self.in_group_counter += 1
        return (next_t, self.is_unfinished_patch)

    @property
    def is_unfinished_mask(self) -> Bool[Tensor, "batch"]:
        is_last_group = self.group_counter >= self.cfg.num_groups - 1
        is_last_step = self.in_group_counter >= self.num_steps_per_group

        return torch.full(
            (self.batch_size,),
            not (is_last_group and is_last_step),
            device=self.device,
            dtype=torch.bool,
        )
