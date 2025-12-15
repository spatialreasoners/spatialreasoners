from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from jaxtyping import Bool, Float
from torch import Tensor, device

from spatialreasoners.denoising_model.flow import Flow
from spatialreasoners.variable_mapper import VariableMapper


@dataclass
class SamplingScheduleCfg(ABC):
    pass
    
    @property
    @abstractmethod
    def num_noise_level_values(self) -> int | None:
        pass


T = TypeVar("T", bound=SamplingScheduleCfg)


class SamplingSchedule(Generic[T], ABC):
    def __init__(
        self,
        cfg: T,
        batch_size: int,
        device: device,
        model_flow: Flow,
        variable_mapper: VariableMapper,
        mask: Float[Tensor, "batch num_variables"] | None = None,
        
    ) -> None:
        self.batch_size = batch_size
        self.device = device
        self.mask = mask
        self.cfg = cfg
        self.model_flow = model_flow
        self.variable_mapper = variable_mapper
        self.current_step = 0

        self._validate_config()
        self._init_batch_state()
        
    @property
    def num_variables(self) -> int:
        return self.variable_mapper.num_variables

    @abstractmethod
    def _validate_config(self):
        pass
    
    @abstractmethod
    def _init_batch_state(self):
        pass

    @property
    @abstractmethod
    def is_unfinished_mask(self) -> Bool[Tensor, "batch"]:
        """Should be checked before __call__ to see if a given batch element should be denoised"""
        pass
        
    @abstractmethod
    def _get_next_t_and_denoise_mask(
        self,
        t: Float[Tensor, "batch_relevant num_variables"],
        sigma_theta: Float[Tensor, "batch_relevant num_variables"] | None = None,
    ) -> tuple[
        Float[Tensor, "batch_relevant num_variables"],
        Bool[Tensor, "batch_relevant num_variables"],
    ]:
        pass

    def __call__(
        self,
        t: Float[Tensor, "batch_relevant num_variables"],
        sigma_theta: Float[Tensor, "batch_relevant num_variables"] | None = None,
    ) -> tuple[
        Float[Tensor, "batch_relevant num_variables"],
        Bool[Tensor, "batch_relevant num_variables"],
    ]:
        next_t, should_denoise = self._get_next_t_and_denoise_mask(t, sigma_theta)
        if self.mask is not None:
            next_t = next_t * self.mask

        self.current_step += 1
        return next_t, should_denoise
    
    @staticmethod
    def _get_should_denoise_mask(
        t: Float[Tensor, "batch num_variables"],
        threshold: float,
    ) -> Bool[Tensor, "batch num_variables"]:
        return (t > threshold)
