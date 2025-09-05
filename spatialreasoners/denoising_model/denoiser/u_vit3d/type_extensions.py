from dataclasses import dataclass

from jaxtyping import Bool, Float
from torch import Tensor


@dataclass(frozen=True, slots=True)
class UViT3DInputs:
    x: Float[Tensor, "batch frames channels height width"]
    noise_levels: Float[Tensor, "batch frames"]
    external_cond: Float[Tensor, "batch frames channels_conditioning height width"]
    external_cond_mask: Bool[Tensor, "batch frames"] | None = None


UViT3DOutputs = Float[Tensor, "batch frames channels height width"]
