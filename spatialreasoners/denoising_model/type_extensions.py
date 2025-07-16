from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor


@dataclass(slots=True, frozen=True)
class TokenizerOutputs:
    """ Standardized output of Tokenizer that DenoisingModel will convert to ModelOutputs.
    """
    mean_theta: Float[Tensor, "batch time num_variables features"] # Before parametrization-specifc transformation
    logvar_theta: Float[Tensor, "batch time num_variables"] | None # Before parametrization-specifc transformation
    variance_theta: Float[Tensor, "batch time num_variables features"] | None


@dataclass(slots=True, frozen=True)
class ModelOutputs:
    """ Standardized output of DenoisingModel.forward to which Tokenizer should convert Denoiser's outputs.
    """
    mean_theta: Float[Tensor, "batch time num_variables features"]
    variance_theta: Float[Tensor, "batch time num_variables features"] | None
    sigma_theta: Float[Tensor, "batch time num_variables"] | None # Uncertainty level