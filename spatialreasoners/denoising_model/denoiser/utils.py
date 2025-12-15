from jaxtyping import Float
from torch import Tensor


def modulate(
    x: Float[Tensor, "*batch"], 
    shift: Float[Tensor, "*#batch"], 
    scale: Float[Tensor, "*#batch"]
) -> Float[Tensor, "*batch"]:
    return x * (1 + scale) + shift
