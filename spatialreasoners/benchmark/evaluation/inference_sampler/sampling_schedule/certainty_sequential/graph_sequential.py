from dataclasses import dataclass
from functools import cache

import torch
from jaxtyping import Float
from torch import Tensor

from .. import register_sampling_schedule
from .certainty_sequential import CertaintySequential, CertaintySequentialCfg


@dataclass(kw_only=True)
class GraphSequentialCfg(CertaintySequentialCfg):
    max_order: int = 1
    certainty_decay: float = 0.5


@register_sampling_schedule("graph_sequential", GraphSequentialCfg)
class GraphSequential(CertaintySequential[GraphSequentialCfg]):
    def _validate_config(self):
        super()._validate_config()
        assert self.cfg.max_order >= 0
        assert self.cfg.certainty_decay > 0, "Certainty decay can't be smaller than 0"
        assert self.cfg.certainty_decay < 1, "Certainty decay can't be higher than 1"
    
    def _init_batch_state(self):
        super()._init_batch_state()
        self.weighted_adjacency_matrix = self._get_weighted_adjacency_matrix()

    @cache
    def _get_weighted_adjacency_matrix(
        self,
    ) -> Float[Tensor, "num_variables num_variables"]:
        dependency_matrix = self.variable_mapper.get_dependency_matrix()
        if self.cfg.max_order == 0:
            return torch.eye(
                dependency_matrix.shape[0]
            )

        weighted_adjacency = dependency_matrix.clone()
        dependency_power = weighted_adjacency.clone()
        weight = 1
        for _ in range(self.cfg.max_order - 1):
            dependency_power @= dependency_matrix
            dependency_power.div_(dependency_power.sum(dim=1))
            weight *= self.cfg.certainty_decay
            weighted_adjacency.add_(weight * dependency_power)
        return weighted_adjacency

    def _get_uncertainty(
        self,
        t: Float[Tensor, "batch total_patches"],
        sigma_theta: Float[Tensor, "batch total_patches"] | None = None,
    ) -> Float[Tensor, "batch total_patches"]:
        assert t is not None, f"t must be provided for {self.cfg}"

        certainty = self._propagate_certainty(1 - t)  # [batch total_patches]
        return 1 - certainty

    def _propagate_certainty(
        self, certainty: Float[Tensor, "batch num_variables"]
    ) -> Float[Tensor, "batch num_variables"]:
        if certainty.sum() < self.cfg.epsilon or self.cfg.max_order == 0:
            return certainty.clone()

        certainty = certainty @ self.weighted_adjacency_matrix.to(
            device=certainty.device
        )
        certainty.div_(certainty.max())
        return certainty
