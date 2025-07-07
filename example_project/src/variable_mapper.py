"""
Spiral Variable Mapper Implementation for SpatialReasoners
"""

from dataclasses import dataclass
import torch
from jaxtyping import Float
from torch import Tensor

import spatialreasoners as sr


@dataclass(frozen=True, kw_only=True)
class SpiralVariablMapperCfg(sr.variable_mapper.VariableMapperCfg):
    pass


@sr.variable_mapper.register_variable_mapper("spiral", SpiralVariablMapperCfg)
class SpiralVariablMapper(sr.variable_mapper.VariableMapper[SpiralVariablMapperCfg]):
    num_variables = 3  # x, y, r (color)
    num_features = 1
        
    def unstructured_tensor_to_variables(self, x: Float[Tensor, "batch x_y_c"]) -> Float[Tensor, "batch num_variables features"]:
        # Input second dimension is x, y, and color
        # Treat the current dimension as variables and just add a feature dimension
        return x.unsqueeze(-1)
    
    def variables_tensor_to_unstructured(self, x: Float[Tensor, "batch num_variables features"]) -> Float[Tensor, "batch *dims"]:
        # Remove the feature dimension
        return x.squeeze(-1)
    
    def mask_variables_tensor_to_unstructured(self, mask: Float[Tensor, "batch num_variables"]) -> Float[Tensor, "batch *dims"]:
        # Masks already don't have feature dimension, so return as is
        return mask 
    
    def mask_unstructured_tensor_to_variables(self, mask: Float[Tensor, "batch *dims"]) -> Float[Tensor, "batch num_variables"]:
        return mask
    
    def _calculate_dependency_matrix(self) -> Float[Tensor, "num_variables num_variables"]:
        dependency_matrix = torch.tensor([
            [1, 0, 1],
            [0, 1, 1],
            [0, 0, 1],
        ]).float()

        return dependency_matrix
