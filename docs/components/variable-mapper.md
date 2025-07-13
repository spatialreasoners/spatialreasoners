# What are Variables?

[SRMs]() TODO operate on *Variables* -- pieces of a sample that in full define the state of the sample. *Variables* might not only directly partition the data, but could also be some latent encoding of the sample. The only constraint we have, is that the shape of the *Variables* remains constant throughout training.


# What do you most likely need?
A simple `VariableMapper` needs to implement four methods

`unstructured_tensor_to_variables` -- maps batch of *Unstructured* datapoints to the *Variables* format, with shape `(batch_size, num_variables, num_features)`.

`variables_tensor_to_unstructured` -- does the inverse of the previous function, transforms the *Variables* format tensor to the *Unstructured* (the same as in your dataset). This method is used after inference, when you want to visualize the generated sample. 

`mask_unstructured_tensor_to_variables` -- if your dataset contains masks, here you need to define how to map them to variables. Note that one variable's mask is represented by a single floating point value -- (for now) we don't support sub-variable masking. 

`mask_variables_tensor_to_unstructured` -- takes the mask in the *Variables* format and transforms that back to the *Unstructured*. This method can be used in your `Evaluation` where you might want to visualize the region your SRM was inpainting. By default, the same method is applied to transform the noise levels of each variable into the *Unstructured* format.

It also needs to define the `num_variables` and `num_features` -- note that those could be `@property` methods, eg. depending on the config. 

### Example: Spiral Variable Mapper

```python
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
    
    def variables_tensor_to_unstructured(self, x: Float[Tensor, "batch num_variables features"]) -> Float[Tensor, "batch 3"]:
        # Remove the feature dimension
        return x.squeeze(-1)
    
    def mask_variables_tensor_to_unstructured(self, mask: Float[Tensor, "batch num_variables"]) -> Float[Tensor, "batch 3"]:
        # Masks already don't have feature dimension, so return as is
        return mask 
    
    def mask_unstructured_tensor_to_variables(self, mask: Float[Tensor, "batch 3"]) -> Float[Tensor, "batch num_variables"]:
        return mask
```


<!-- # What about generation in the latent space? 
[todo] -->

# What if I know the dependency structure of my data?


If you want to exploit the known causal structure of your variables when choosing the order of your inference, you need to define the `dependency_matrix`. You can do so, by implementing the `_calculate_dependency_matrix` method in the `VariableMapper`. The matrix defines for each variable (column) on which variables (row) does it depend. There are no restrictions about the acyclicity of the dependency matrices. 

Below is an example implementation for the *Spirals* toy dataset

```python
def _calculate_dependency_matrix(self) -> Float[Tensor, "num_variables num_variables"]:
    # color is the third variable, and it depends both on x and y
    dependency_matrix = torch.tensor([
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 1],
    ]).float()

    return dependency_matrix
```