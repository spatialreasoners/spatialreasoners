import copy
from dataclasses import dataclass
from typing import Any, Literal, NotRequired, TypedDict

import torch
from jaxtyping import Float, Int64
from torch import Tensor

FullPrecision = Literal[32, 64, "32-true", "64-true", "32", "64"]
HalfPrecision = Literal[16, "16-true", "16-mixed", "bf16-true", "bf16-mixed", "bf16", "16"]
Stage = Literal["train", "val", "test"]

Parameterization = Literal["eps", "ut", "x0", "v"]


# NOTE conditioning both required for model and dataset
# therefore define here to avoid circular dependencies
@dataclass(slots=True, frozen=True, kw_only=True)
class ConditioningCfg:
    label: bool = False
    mask: bool = False


class UnstructuredExample(TypedDict, total=True):
    """ Represents the raw example from the dataset
    """
    in_dataset_index: int
    path: NotRequired[str]
    z_t: NotRequired[Float[Tensor, "..."]] # Data of the example (clean or noisy)
    cached_latent: NotRequired[Float[Tensor, "..."]]
    # TODO is_latent: bool = False
    label: NotRequired[int]
    fixed_conditioning_fields: NotRequired[dict[str, Any]]
    
    # if mask == 1: needs to be inpainted
    mask: NotRequired[Float[Tensor, "..."]]


class BatchUnstructuredExample(TypedDict, total=True):
    """ Represents a batch of examples from the dataset
    """
    in_dataset_index: Int64[Tensor, "batch"]
    path: NotRequired[list[str]]
    z_t: NotRequired[Float[Tensor, "batch ..."]] # Data of the example (clean or noisy)
    cached_latent: NotRequired[Float[Tensor, "batch ..."]]
    # TODO is_latent: bool = False
    label: NotRequired[Int64[Tensor, "batch"]]
    fixed_conditioning_fields: NotRequired[dict[str, Any]]
    
    mask: NotRequired[Float[Tensor, "batch ..."]]
    
    sigma: NotRequired[Float[Tensor, "batch ..."]]  # Can be used in inference for uncertainty visualization
    t: NotRequired[Float[Tensor, "batch ..."]]      # Can be used in inference for noise level visualization
    x_pred: NotRequired[Float[Tensor, "batch ..."]] # Can be used in inference for clean prediction visualization
    
    num_steps: NotRequired[Int64[Tensor, "batch"]] # Number of steps in the diffusion process
    


@dataclass(slots=True, frozen=True, kw_only=True)
class BatchVariables:
    """ Represents a batch of variables from the dataset
    """
    z_t: Float[Tensor, "batch num_variables dim"]  # Data of the example (clean or noisy)
    t: Float[Tensor, "batch num_variables"]        # Noise level (zeros for clean data)

    in_dataset_index: Int64[Tensor, "batch"]    # Index of the example in the dataset
    path: list[str] | None = None               # Path to the example (if exists)
    
    label: Int64[Tensor, "batch"] | None = None                 # Conditioning label
    fixed_conditioning_fields: dict[str, Any] | None = None
    
    mask: Float[Tensor, "batch num_variables"] | None = None    # Mask of the example, 1=inpainted, 0=clean

    sigma_pred: Float[Tensor, "batch num_variables"] | None = None # Uncertainty level
    x_pred: Float[Tensor, "batch num_variables dim"] | None = None # Predicted clean data
    
    num_steps: Int64[Tensor, "batch"] | None = None # Number of steps in the diffusion process

    @property
    def batch_size(self) -> int:
        return self.z_t.shape[0]
    
    @property
    def device(self) -> torch.device:
        return self.z_t.device
    
    @property
    def num_variables(self) -> int:
        return self.z_t.shape[1]
    
    @property
    def num_features(self) -> int:
        return self.z_t.shape[2]

    @staticmethod
    def _clone_tensor(tensor: Tensor | None) -> Tensor | None:
        if tensor is None:
            return None
        return tensor.clone().detach().to(tensor.device)

    def __deepcopy__(self, memo):
        return BatchVariables(
            in_dataset_index=self._clone_tensor(self.in_dataset_index),
            path=copy.deepcopy(self.path, memo),
            z_t=self._clone_tensor(self.z_t),
            t=self._clone_tensor(self.t),
            label=self._clone_tensor(self.label),
            mask=self._clone_tensor(self.mask),
            sigma_pred=self._clone_tensor(self.sigma_pred),
            x_pred=self._clone_tensor(self.x_pred),
        )
        
    def clone(self) -> "BatchVariables":
        return self.__deepcopy__(memo={})