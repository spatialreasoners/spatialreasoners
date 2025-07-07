from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence, TypeVar

from jaxtyping import Float
from torch import Tensor, nn

from ...denoiser.dit.type_extensions import DiTModelInputs, DiTModelOutput
from ...type_extensions import TokenizerOutputs
from ..tokenizer import Tokenizer, TokenizerCfg


@dataclass(frozen=True, kw_only=True)
class DiTTokenizerCfg(TokenizerCfg):
    pass


T_CFG = TypeVar("T_CFG", bound=DiTTokenizerCfg)


class DiTTokenizer(Tokenizer[T_CFG, DiTModelInputs, DiTModelOutput], ABC):
    """
    Base class for DiT tokenizers.
    """
    
    @abstractmethod
    def assemble_tokens_to_variables(
        self, 
        mean_pred: Float[Tensor, "batch time num_tokens features"], # Before parametrization-specifc transformation
        logvar_pred: Float[Tensor, "batch time num_tokens"] | None, # Before parametrization-specifc transformation
        v_pred: Float[Tensor, "batch time num_tokens features"] | None,
    ) -> TokenizerOutputs:
        pass

    def model_outputs_to_variable_predictions(
        self, 
        model_outputs: DiTModelOutput, 
        batch_size: int, 
        num_times: int,
    ) -> TokenizerOutputs:
        model_outputs = model_outputs.unflatten(0, (batch_size, num_times))
        
        mean_theta = model_outputs[:,:, :, :self.model_d_data]
        logvar_theta = model_outputs[:, :, :, -1] if self.predict_uncertainty else None
        
        if self.predict_v:
            v_theta = model_outputs[:, :, :, self.model_d_data:2 * self.model_d_data]
            v_theta = (v_theta + 1) / 2 # Normalize to [0, 1]
        else:
            v_theta = None
        
        return self.assemble_tokens_to_variables(
            mean_pred=mean_theta,
            logvar_pred=logvar_theta,
            v_pred=v_theta
        )
    
    @property
    @abstractmethod
    def model_d_in(self) -> int:
        pass
    
    @property
    @abstractmethod
    def model_d_out(self) -> int:
        pass
    
    @property
    @abstractmethod
    def token_grid_size(self) -> Sequence[int]:
        pass
