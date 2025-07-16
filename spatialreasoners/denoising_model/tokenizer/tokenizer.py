from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from spatialreasoners.variable_mapper import VariableMapper

from ..type_extensions import TokenizerOutputs


@dataclass(frozen=True, kw_only=True)
class TokenizerCfg:
    pass


T_CFG = TypeVar("T_CFG", bound=TokenizerCfg)

T_INPUT = TypeVar("T_INPUT", bound=dict[str, Any])  # Tokens format
T_OUTPUT = TypeVar("T_OUTPUT")  # Tokens format


class Tokenizer(nn.Module, Generic[T_CFG, T_INPUT, T_OUTPUT], ABC):
    """
    Base class for tokenizers.
    """

    def __init__(
        self, 
        cfg: TokenizerCfg, 
        variable_mapper: VariableMapper,
        predict_uncertainty: bool = False,
        predict_variance: bool = False,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.variable_mapper = variable_mapper
        self.predict_uncertainty = predict_uncertainty
        self.predict_variance = predict_variance

    @abstractmethod
    def variables_to_model_inputs(
        self, 
        z_t: Float[Tensor, "batch time num_variables dim"],
        t: Float[Tensor, "batch time num_variables"],
        should_denoise: Bool[Tensor, "batch time num_variables"] | None = None,
        mask: Float[Tensor, "batch num_variables"] | None = None,
        x_masked: Float[Tensor, "batch num_variables dim"] | None = None,
        label: Int[Tensor, "batch"] | None = None,
        label_mask: Bool[Tensor, "batch"] | None = None,
        fixed_conditioning_fields: Any | None = None,
    ) -> T_INPUT:  # Any shape the denoiser might need
        """
        Convert variables to tokens.
        """
        pass

    @abstractmethod
    def model_outputs_to_variable_predictions(
        self, 
        model_outputs: T_OUTPUT, 
        batch_size: int, 
        num_times: int,
    ) -> TokenizerOutputs:
        pass
