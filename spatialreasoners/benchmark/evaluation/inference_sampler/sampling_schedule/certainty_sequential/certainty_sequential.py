from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar

import torch
from jaxtyping import Bool, Float, Int32, Int64
from torch import Tensor

from ..sampling_schedule import SamplingSchedule, SamplingScheduleCfg


@dataclass(kw_only=True)
class CertaintySequentialCfg(SamplingScheduleCfg, ABC):
    max_steps: int
    num_parallel: int = 1  # previously top-k
    overlap: float | None = 0.1 # Use either overlap or num_steps_per_group
    num_steps_per_group: int | None = None
    epsilon: float = 1e-6
    reverse_certainty: bool = False
    
    @property
    def num_noise_level_values(self) -> int | None:
        assert (
            (self.overlap is not None) != (self.num_steps_per_group is not None)
        ), "You have to set the overlap or the num_steps_per_group, but not both"
        
        if self.overlap is not None:
            return None # infinite possible noise levels
        
        return self.num_steps_per_group + 1  # +1 for the initial noise level of 1

T = TypeVar("T", bound=CertaintySequentialCfg)


class CertaintySequential(SamplingSchedule[T], ABC):
    def _validate_config(self):
        assert self.cfg.epsilon > 0
        if self.cfg.overlap is not None:
            assert self.cfg.overlap >= 0 and self.cfg.overlap <= 1
            assert self.cfg.num_steps_per_group is None
        else:
            assert self.cfg.num_steps_per_group is not None
            assert self.cfg.num_steps_per_group > 0
            
        assert self.cfg.num_parallel > 0

        assert (
            self.num_variables % self.cfg.num_parallel == 0
        ), "num_parallel must divide total_patches"

    def _init_batch_state(self):
        # Set up scheduling
        selectable = (
            self.mask > 1 - self.cfg.epsilon
            if self.mask is not None
            else torch.ones(
                self.batch_size, self.num_variables, device=self.device, dtype=torch.bool
            )
        )  # [batch_size, total_patches]
        scheduling_matrix = torch.ones(
            [self.cfg.max_steps + 1, self.batch_size, self.num_variables], device=self.device
        )

        # Zero out known regions
        scheduling_matrix *= selectable.unsqueeze(0)
        # [max_steps, batch_size, total_patches]

        num_unknown_patches = selectable.sum(dim=1).long()
        # [batch_size]

        num_inference_blocks = torch.ceil(
            num_unknown_patches / self.cfg.num_parallel
        ).int()
        
        if self.cfg.overlap is not None:
            ideal_block_lengths = self._get_inference_lengths(num_inference_blocks)
            block_lengths = ideal_block_lengths.ceil().int()  # [batch_size]
            self.block_starts = (
                torch.arange(num_inference_blocks.max() + 1, device=self.device).unsqueeze(0)
                * ideal_block_lengths.unsqueeze(1)
                * (1 - self.cfg.overlap)
            ).floor()
        else:
            last_block_start = self.cfg.max_steps - self.cfg.num_steps_per_group
            block_lengths = torch.full(
                [self.batch_size], self.cfg.num_steps_per_group, device=self.device, dtype=torch.int32)
            
            self.block_starts = torch.arange(
                0, last_block_start, num_inference_blocks.max() + 1, device=self.device
            ).floor()

        unique_counts = torch.unique(self.block_starts, dim=1, return_counts=True)[1]
        assert (
            unique_counts.max() == 1
        ), "Block starts must be unique for each batch element -- possibly too high overlap"
        
        self.block_lengths = block_lengths
        self.num_inference_blocks = num_inference_blocks
        
        self.scheduling_matrix = scheduling_matrix
        self.selectable = selectable
        self.block_starts[:, -1] = -1  # This extra block should never be used!
        
        self.block_counters = torch.zeros(
            self.batch_size, device=self.device, dtype=torch.int64
        )
        self.step_targets = torch.zeros(
            self.batch_size, device=self.device, dtype=torch.int64
        )
        self.prototypes = self._get_schedule_prototypes(block_lengths)
        self.same_step_offset = 0
        
        # Calculate the finishing step for each batch element
        rows = torch.arange(self.num_inference_blocks.shape[0], device=self.device) 
        self.finishing_step = self.block_starts[rows,self.num_inference_blocks]
        self.finishing_step[self.finishing_step == -1] = self.cfg.max_steps
        
    @abstractmethod
    def _get_uncertainty(
        self,
        t: Float[Tensor, "batch total_patches"],
        sigma_theta: Float[Tensor, "batch total_patches"],
    ) -> Float[Tensor, "batch total_patches"]:
        pass

    @property
    def is_unfinished_mask(self) -> Bool[Tensor, "batch"]:        
        # We need to check if the current step is less than the finishing step
        return self.combined_step < self.finishing_step

    @property
    def combined_step(self) -> int:
        return self.current_step + self.same_step_offset
        
    def _get_inference_lengths(
        self, num_inference_blocks: Int32[Tensor, "batch_size"]
    ) -> Float[Tensor, "batch_size"]:
        if self.cfg.num_steps_per_group is not None:
            return torch.full(
                [self.batch_size], self.cfg.num_steps_per_group, device=self.device, dtype=torch.float
            )
            
        ideal_lengths = self.cfg.max_steps / (
            (num_inference_blocks - 1) * (1 - self.cfg.overlap) + 1
        )

        return ideal_lengths

    def _get_schedule_prototypes(
        self, prototype_lengths: Int32[Tensor, "batch_size"]
    ) -> Float[Tensor, "max_prototype_length batch_size"]:
        batch_size = prototype_lengths.size(0)
        device = prototype_lengths.device

        max_prototype_length = prototype_lengths.max()
        assert prototype_lengths.min() > 0

        # We add one to the prototype length to include the timestep with just zeros
        prototype_base = torch.linspace(
            max_prototype_length, 0, max_prototype_length + 1, device=device
        )

        prototypes = prototype_base.unsqueeze(0).expand(batch_size, -1)

        # shift down based on the prototype lengths
        prototypes = prototypes - (max_prototype_length - prototype_lengths).unsqueeze(1)
        # [batch_size, max_prototype_length + 1]

        # scale to batch_max = 1
        prototypes = prototypes / prototype_lengths.unsqueeze(1)
        assert prototypes.max() <= 1 + self.cfg.epsilon

        prototypes.clamp_(0, 1)    # [batch_size, max_prototype_length + 1]
        prototypes = prototypes.T  # [max_prototype_length + 1, batch_size]
        return prototypes[:-1]     # [max_prototype_length, batch_size] skip trailing zeros

    def _get_next_patch_ids(
        self,
        patch_uncertainty: Float[Tensor, "batch relevant_patches"],
        selectable: Bool[Tensor, "batch relevant_patches"],
    ) -> Int64[Tensor, "batch top_k"]:
        if self.cfg.reverse_certainty:
            patch_uncertainty = patch_uncertainty * selectable

            selected_indices = torch.topk(
                patch_uncertainty, self.cfg.num_parallel, largest=True
            ).indices
        else:
            # Get K non-masked regions with lowest uncertainty for each batch element
            known_shift = patch_uncertainty.max() + 1  # to avoid known regions
            patch_uncertainty = patch_uncertainty + ~selectable * known_shift

            selected_indices = torch.topk(
                patch_uncertainty, self.cfg.num_parallel, largest=False
            ).indices

        return selected_indices

    def _expand_masked_to_zeros(
        self,
        unfinished_tensor: Float[Tensor, "batch_relevant total_patches"],
        mask: Bool[Tensor, "batch"],
    ) -> Float[Tensor, "batch total_patches"]:
        
        expanded = torch.zeros([self.batch_size, self.num_variables], device=self.device)
        expanded[mask] = unfinished_tensor

        return expanded

    def _get_next_t_and_denoise_mask(
        self,
        t: Float[Tensor, "batch_relevant total_patches"],
        sigma_theta: Float[Tensor, "batch_relevant total_patches"] | None = None,
    ) -> tuple[
        Float[Tensor, "batch_relevant total_patches"],
        Bool[Tensor, "batch_relevant total_patches"],
    ]:
        relevant_mask = self.is_unfinished_mask
        t = self._expand_masked_to_zeros(t, relevant_mask)
        if sigma_theta is not None:
            sigma_theta = self._expand_masked_to_zeros(sigma_theta, relevant_mask)
        
        uncertainty = self._get_uncertainty(t, sigma_theta)
        uncertainty.squeeze_(1)
        should_start = self.step_targets == self.combined_step

        # Only start new blocks if there are still patches to denoise
        should_start &= relevant_mask

        if self.selectable.sum() > self.cfg.epsilon and should_start.any():
            self.block_counters += should_start.int()
            self.step_targets = self.block_starts[
                torch.arange(self.batch_size, device=self.device), self.block_counters
            ]

            should_start_batch_ids = should_start.nonzero(as_tuple=True)[0]

            uncertainty_relevant = uncertainty[should_start_batch_ids]
            selectable_relevant = self.selectable[should_start_batch_ids]
            prototypes_relevant = self.prototypes[:, should_start_batch_ids]

            next_ids = self._get_next_patch_ids(
                uncertainty_relevant, selectable_relevant
            )  # This might include patches that are already known for K > 1

            repeat_batch_ids = torch.repeat_interleave(
                should_start_batch_ids, repeats=next_ids.shape[1]
            )

            repeat_prototypes = torch.repeat_interleave(
                prototypes_relevant, repeats=next_ids.shape[1], dim=1
            )

            flat_next_ids = next_ids.flatten()
            self.selectable[repeat_batch_ids, flat_next_ids] = False

            length_to_consider = min(
                repeat_prototypes.shape[0], self.cfg.max_steps - self.combined_step
            )

            # Paste the prototype into the scheduling matrix
            # We need the torch.minimum because for K>1, we might have chosen a patch that is already known
            self.scheduling_matrix[
                self.combined_step : self.combined_step + length_to_consider,
                repeat_batch_ids,
                flat_next_ids,
            ] = torch.minimum(
                repeat_prototypes[:length_to_consider],
                self.scheduling_matrix[
                    self.combined_step : self.combined_step + length_to_consider,
                    repeat_batch_ids,
                    flat_next_ids,
                ],
            )

            if self.combined_step + length_to_consider < self.cfg.max_steps:
                self.scheduling_matrix[
                    self.combined_step + length_to_consider :,
                    repeat_batch_ids,
                    flat_next_ids,
                ] = 0

            self.scheduling_matrix[-1] = 0

        t_new = self.scheduling_matrix[self.combined_step + 1]
        
        if torch.allclose(t_new, t, atol=self.cfg.epsilon):
            self.same_step_offset += 1
            return self._get_next_t_and_denoise_mask(t, sigma_theta) # recursive call
        
        t_new_relevant = t_new[relevant_mask]
        t_relevalnt = t[relevant_mask]
        
        return t_new_relevant, self._get_should_denoise_mask(t_relevalnt, self.cfg.epsilon)
