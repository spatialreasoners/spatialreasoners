from dataclasses import dataclass

from jaxtyping import Bool, Float, Int64
from torch import Tensor


@dataclass(frozen=True, slots=True)
class UNetModelInputs:
    z_t: Float[Tensor, "batch channels height width"]
    t: Float[Tensor, "batch 1 width height"]
    label: Int64[Tensor, "batch"] | None = None
    label_mask: Bool[Tensor, "batch"] | None = None
    
UNetModelOutput = Float[Tensor, "batch channels height width"]