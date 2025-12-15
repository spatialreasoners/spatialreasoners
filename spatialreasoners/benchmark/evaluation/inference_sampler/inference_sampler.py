from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import torch
from jaxtyping import Bool, Float, Int64, Shaped
from torch import Tensor

from spatialreasoners.denoising_model import DenoisingModel
from spatialreasoners.variable_mapper import VariableMapper

from .cfg_scheduler import CFGSchedulerCfg, ConstantCfg, get_cfg_scheduler
from .type_extensions import FinalInferenceBatchSample, IntermediateInferenceBatchSample


@dataclass(kw_only=True, frozen=True)
class InferenceSamplerCfg:
    alpha: float | int = 0
    temperature: float | int = 1
    use_ema: bool = True  # NOTE ignored if model does not have EMA
    label_cfg_scheduler: CFGSchedulerCfg = field(default_factory=ConstantCfg)


T = TypeVar("T", bound=InferenceSamplerCfg)


class InferenceSampler(Generic[T], ABC):
    def __init__(
        self,
        cfg: T,
        variable_mapper: VariableMapper,
    ) -> None:
        super().__init__()
        self.variable_mapper = variable_mapper        
        self.cfg = cfg
        self.label_cfg_scheduler = get_cfg_scheduler(cfg.label_cfg_scheduler)
        
    @property
    def num_variables(self) -> int | None:
        return self.variable_mapper.num_variables

    @property
    @abstractmethod
    def num_noise_level_values(self) -> int | None:
        pass

    @staticmethod
    def _prepare_classifier_free_guidance_split(
        batch_size: int,
        device: torch.device,
        *args: Shaped[Tensor, "batch *_"] | None,
    ) -> tuple[
        Bool[Tensor, "2*batch"], tuple[Shaped[Tensor, "2*batch *_"] | None, ...]
    ]:
        label_mask = (
            torch.tensor([False, True], device=device)
            .unsqueeze(1)
            .expand(-1, batch_size)
            .flatten()
        )
        res = tuple(
            arg.repeat(2, *((arg.ndim - 1) * (1,))) if arg is not None else None
            for arg in args
        )
        return label_mask, res

    def _merge_classifier_free_guidance(
        self,
        label_cfg_scale: Float[Tensor, "batch *_ #num_variables"],
        z_t: Float[Tensor, "2*batch *_ num_variables dim"],
        t: Float[Tensor, "2*batch *_ num_variables"],
        mean_theta: Float[Tensor, "2*batch *_ num_variables dim"],
        v_theta: Float[Tensor, "2*batch *_ num_variables dim"] | None,
        sigma_theta: Float[Tensor, "2*batch *_ #num_variables"] | None,
    ) -> tuple[
        Float[Tensor, "batch *_ num_variables dim"],  # z_t
        Float[Tensor, "batch *_ num_variables"],  # t
        Float[Tensor, "batch *_ num_variables dim"],  # mean_theta
        Float[Tensor, "batch *_ num_variables dim"] | None,  # v_theta
        Float[Tensor, "batch *_ #num_variables"] | None,  # sigma_theta
    ]:
        z_t, _ = z_t.chunk(2)
        t, _ = t.chunk(2)
        cond_mean, uncond_mean = mean_theta.chunk(2)
        mean_theta = uncond_mean + label_cfg_scale.unsqueeze(-1) * (cond_mean - uncond_mean)

        if v_theta is not None:
            v_theta, _ = v_theta.chunk(2)  # take the variance for the conditional model

        if sigma_theta is not None:
            cond_sigma, uncond_sigma = sigma_theta.chunk(2)
            uncond_var = uncond_sigma**2
            sigma_theta = torch.sqrt(
                uncond_var + label_cfg_scale.unsqueeze(-1)**2 * (cond_sigma**2 + uncond_var)
            )

        return z_t, t, mean_theta, v_theta, sigma_theta

    @abstractmethod
    def _sample(
        self,
        denoising_model: DenoisingModel,
        z_t: Float[Tensor, "batch num_variables dim"],
        t: Float[Tensor, "batch num_variables"],
        mask: Float[Tensor, "batch num_variables"] | None = None,
        x_masked: Float[Tensor, "batch num_variables dim"] | None = None,
        label: Int64[Tensor, "batch"] | None = None,
        fixed_conditioning_fields: Any | None = None,
        return_intermediate: bool = False,
        return_time: bool = False,
        return_sigma: bool = False,
        return_x_pred: bool = False,
    ) -> Iterator[IntermediateInferenceBatchSample | FinalInferenceBatchSample]:
        pass

    def __call__(
        self,
        denoising_model: DenoisingModel,
        z_t: Float[Tensor, "batch num_variables dim"],
        t: Float[Tensor, "batch num_variables"],
        mask: Float[Tensor, "batch num_variables"] | None = None,
        x_masked: Float[Tensor, "batch num_variables dim"] | None = None,
        label: Int64[Tensor, "batch"] | None = None,
        fixed_conditioning_fields: Any | None = None,
        return_intermediate: bool = False,
        return_sigma: bool = False,
        return_x_pred: bool = False,
    ) -> Iterator[IntermediateInferenceBatchSample | FinalInferenceBatchSample]:
    
        # TODO can we find a spot where this doesn't have to be called for every batch?
        denoising_model.on_sampling_start(self.num_noise_level_values)
        
        res = self._sample(
            denoising_model=denoising_model,
            z_t=z_t,
            t=t,
            mask=mask,
            x_masked=x_masked,
            label=label,
            fixed_conditioning_fields=fixed_conditioning_fields,
            return_intermediate=return_intermediate,
            return_sigma=return_sigma,
            return_x_pred=return_x_pred,
        )
        
        denoising_model.on_sampling_end()
        return res
