from dataclasses import dataclass

from jaxtyping import Bool, Float, Int64
from torch import Tensor


@dataclass(frozen=True, slots=True)
class DiTModelInputs:
    z_t: Float[Tensor, "batch num_tokens d_in"]
    t: Float[Tensor, "batch num_tokens"]
    token_coordinates_xy: Float[Tensor, "batch num_tokens num_dims"] | None = None
    token_coordinates_ij: Float[Tensor, "batch num_tokens num_dims"] | None = None
    label: Int64[Tensor, "batch"] | None = None
    label_mask: Bool[Tensor, "batch"] | None = None
    
    
DiTModelOutput = Float[Tensor, "batch num_tokens d_out"]