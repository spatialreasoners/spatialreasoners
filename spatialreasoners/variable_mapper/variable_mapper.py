from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cache
from typing import Generic, Sequence, TypeVar

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from spatialreasoners.misc.nn_module_tools import freeze
from spatialreasoners.type_extensions import BatchUnstructuredExample, BatchVariables

from .autoencoder import AutoencoderCfg, get_autoencoder


@dataclass(kw_only=True, frozen=True)
class VariableMapperCfg:
    """
    Configuration class for the representation adapter.
    """
    autoencoder: AutoencoderCfg | None = field(default=None)


T = TypeVar("T", bound=VariableMapperCfg)
    
class VariableMapper(nn.Module, Generic[T], ABC):
    """
    Base class for representation adapters.
    """
    
    def __init__(
        self,
        cfg: VariableMapperCfg,
        unstructured_sample_shape: Sequence[int],
    ) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.autoencoder is not None:    
            self.autoencoder = get_autoencoder(cfg.autoencoder, unstructured_sample_shape)
            freeze(self.autoencoder)
            self.unstructured_sample_shape = self.autoencoder.latent_shape
        else:
            self.autoencoder = None
            self.unstructured_sample_shape = unstructured_sample_shape
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # Don't save the autoencoder 
        return {k: v for k, v in super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars).items() \
                if not k.startswith(("autoencoder",))}

    @property
    @abstractmethod
    def num_variables(self) -> int:
        """
        Number of variables in the variable representation.
        """
        pass
    
    @property
    @abstractmethod
    def num_features(self) -> int:
        """
        Number of features in the variable representation.
        """
        pass

    @abstractmethod
    def unstructured_tensor_to_variables(self, x: Float[Tensor, "batch *dims"]) -> Float[Tensor, "batch num_variables features"]:
        """
        Transform input tensor to variable representation.
        
        Args:
            x: Input tensor with shape (batch, *dims)
        
        Returns:
            Transformed tensor with shape (batch, num_variables, data)
            
        Requires:
            prod(dims) == num_variables * data
        """
        pass
    
    @abstractmethod
    def variables_tensor_to_unstructured(self, x: Float[Tensor, "batch num_variables features"]) -> Float[Tensor, "batch *dims"]:
        """
        Transform variable representation to Unstruct.
        
        Args:
            x: Input tensor with shape (batch, num_variables, data)
        
        Returns:
            Transformed tensor with shape (batch, *dims)
            
        Requires:
            prod(dims) == num_variables * data
        """
        pass
    
    
    @abstractmethod
    def mask_unstructured_tensor_to_variables(
        self,
        mask: Float[Tensor, "batch *dims"] | Bool[Tensor, "batch *dims"]
    ) -> Float[Tensor, "batch num_variables"] | Bool[Tensor, "batch num_variables"]:
        """
        Transform input mask to variable representation.
        
        Args:
            mask: Mask tensor with shape (batch, *dims)
        
        Returns:
            Transformed mask tensor with shape (batch, num_variables, data)
            
        """
        pass

    
    @abstractmethod
    def mask_variables_tensor_to_unstructured(
        self,
        mask: Float[Tensor, "batch num_variables"] | Bool[Tensor, "batch num_variables"]
    ) -> Float[Tensor, "batch *dims"] | Bool[Tensor, "batch *dims"]:
        """
        Transform variable representation to Unstruct.
        
        Args:
            mask: Mask tensor with shape (batch, num_variables)
        
        Returns:
            Transformed mask tensor with shape (batch, *dims)
            
        """
        pass
    
    def _calculate_dependency_matrix(self) -> Float[Tensor, "num_variables num_variables"]:
        """
        Calculate the dependency matrix for the variables.
        The dependency matrix is a square matrix of shape (num_variables, num_variables)
        where each element (i, j) represents the dependency between variable i and variable j.
        In this case, we assume that all variables are independent, so the matrix is an identity matrix.
        
        Returns:
            Dependency matrix with shape (num_variables, num_variables)
        """
        return torch.eye(self.num_variables, dtype=torch.float32)
        
    @cache
    def get_dependency_matrix(
        self, device: torch.device | str = "cpu"
    ) -> Float[Tensor, "batch num_variables num_variables"]:
        return self._calculate_dependency_matrix().to(device)

    @torch.no_grad()
    def unstructured_to_variables(self, batch_unstr: BatchUnstructuredExample) -> BatchVariables:
        """
        Transform input tensor to variable representation.
        
        Args:
            x: Input tensor with shape (batch, *dims)
        
        Returns:
            Transformed tensor with shape (batch, num_variables, data)
            
        Requires:
            prod(dims) == num_variables * data
        """
        
         # Autoencoder encoding
        if self.autoencoder is not None:
            # latent diffusion
            if "cached_latent" in batch_unstr:
                encoding = self.autoencoder.tensor_to_encoding(batch_unstr["cached_latent"])
                z_t = self.autoencoder.sample_latent(encoding)
            else:
                z_t = self.autoencoder.encode(batch_unstr["z_t"])
            
        else:
            # pixel space diffusion
            z_t = batch_unstr["z_t"]

        variables_z_t = self.unstructured_tensor_to_variables(z_t)
                
        # Mask conversion
        if "mask" in batch_unstr:
            assert self.autoencoder is None, "Latent masking not supported"
            mask = self.mask_unstructured_tensor_to_variables(batch_unstr["mask"])
        else:
            mask = None
            
        return BatchVariables(
            in_dataset_index=batch_unstr["in_dataset_index"],
            z_t=variables_z_t,
            fixed_conditioning_fields=batch_unstr["fixed_conditioning_fields"] if "fixed_conditioning_fields" in batch_unstr else None,
            label=batch_unstr["label"] if "label" in batch_unstr else None,
            path=batch_unstr["path"] if "path" in batch_unstr else None,
            mask=mask,
            t=torch.zeros(variables_z_t.shape[:2], device=variables_z_t.device), # Initially no noise
            num_steps=batch_unstr["num_steps"] if "num_steps" in batch_unstr else None,
        )    
    
    @torch.no_grad()
    def variables_to_unstructured(self, batch_variables: BatchVariables) -> BatchUnstructuredExample:
        # Variables to unstructured
        z_t = self.variables_tensor_to_unstructured(batch_variables.z_t)
        
        # Autoencoder decoding
        if self.autoencoder is not None:
            z_t = self.autoencoder.decode(z_t)

        # Masking
        mask = None
        if batch_variables.mask is not None:
            assert self.autoencoder is None, "Latent masking not supported"
            mask = self.mask_variables_tensor_to_unstructured(batch_variables.mask)
        
        res = BatchUnstructuredExample(
            in_dataset_index=batch_variables.in_dataset_index,
            num_steps=batch_variables.num_steps,
            fixed_conditioning_fields=batch_variables.fixed_conditioning_fields,
            z_t=z_t,
            label=batch_variables.label,
            path=batch_variables.path,
            mask=mask,  
            x_pred=self.x_pred_to_unstructured(batch_variables.x_pred),
            t=self.t_to_unstructured(batch_variables.t),
            sigma=self.sigma_to_unstructured(batch_variables.sigma_pred),
        )
        if batch_variables.num_steps is not None:
            res["num_steps"] = batch_variables.num_steps
        if batch_variables.label is not None:
            res["label"] = batch_variables.label
        if batch_variables.path is not None:
            res["path"] = batch_variables.path
        if mask is not None:
            res["mask"] = mask
        if batch_variables.x_pred is not None:
            res["x_pred"] = self.x_pred_to_unstructured(batch_variables.x_pred)
        if batch_variables.t is not None:
            res["t"] = self.t_to_unstructured(batch_variables.t)
        if batch_variables.sigma_pred is not None:
            res["sigma"] = self.sigma_to_unstructured(batch_variables.sigma_pred)

        return res
        
        
    def x_pred_to_unstructured(self, x_pred: Float[Tensor, "batch num_variables dim"] | None) -> Float[Tensor, "batch *dims"] | None:
        """
        Transform predicted clean data to unstructured tensor.
        """
        if x_pred is None:
            return None
        
        x_pred = self.variables_tensor_to_unstructured(x_pred)
        if self.autoencoder is not None:
            x_pred = self.autoencoder.decode(x_pred)
        return x_pred
    
    def t_to_unstructured(self, t: Float[Tensor, "batch num_variables"] | None) -> Float[Tensor, "batch *dims"] | None:
        """
        Transform noise level to unstructured tensor.
        """
        return self.mask_variables_tensor_to_unstructured(t) if t is not None else None
    
    def sigma_to_unstructured(self, sigma: Float[Tensor, "batch num_variables"] | None) -> Float[Tensor, "batch *dims"] | None:
        """
        Transform uncertainty to unstructured tensor.
        """
        return self.mask_variables_tensor_to_unstructured(sigma) if sigma is not None else None
