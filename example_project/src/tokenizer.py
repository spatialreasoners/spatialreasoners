"""
Spiral Tokenizer Implementation for SpatialReasoners
"""

from dataclasses import dataclass
from typing import Any
import torch
from jaxtyping import Float, Bool, Int
from torch import Tensor

import spatialreasoners as sr
from spatialreasoners.denoising_model import tokenizer
from spatialreasoners.denoising_model.type_extensions import TokenizerOutputs


# Type definitions for the denoiser
SpatialDenoiserInputs = Float[Tensor, "batch 3 2"]  # batch, (x,y,c), (value, time)
SpatialDenoiserOutputs = Float[Tensor, "batch 3 num_values"]  # batch, (x,y,c), (value, ?variance, ?uncertainty)


@dataclass(frozen=True, kw_only=True)
class SpiralTokenizerCfg(tokenizer.TokenizerCfg):
    pass


@tokenizer.register_tokenizer("spiral", SpiralTokenizerCfg)
class SpiralTokenizer(tokenizer.Tokenizer[SpiralTokenizerCfg, SpatialDenoiserInputs, SpatialDenoiserOutputs]):
    
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
    ) -> SpatialDenoiserInputs:
        """Convert variables to tokens."""
        # Remove time dimension and flatten
        z_t = z_t.reshape(z_t.shape[0], -1)  # [batch, time*num_variables*dim]
        t = t.reshape(t.shape[0], -1)        # [batch, time*num_variables]
        
        # Ensure both tensors have the same dtype
        t = t.to(z_t.dtype)
        
        # Concatenate z_t and t along the last dimension
        inputs = torch.cat([z_t, t], dim=1)  # [batch, time*num_variables*dim + time*num_variables]
        return inputs

    def model_outputs_to_variable_predictions(
        self, 
        model_outputs: SpatialDenoiserOutputs, 
        batch_size: int, 
        num_times: int,
    ) -> TokenizerOutputs:
        
        # Add the time dimension back
        outputs = model_outputs.reshape(batch_size, num_times, *model_outputs.shape[1:])
        
        if self.predict_uncertainty:
            logvar_theta = outputs[:, :, :, -1]
            outputs = outputs[:, :, :, :-1]
        else:
            logvar_theta = None
            
        if self.predict_variance:
            channel_dim = outputs.shape[-1]
            half_channel_dim = channel_dim // 2
            
            variance = outputs[:, :, :, half_channel_dim:]
            mean_theta = outputs[:, :, :, :half_channel_dim]
        else:
            mean_theta = outputs
            variance = None
            
        return TokenizerOutputs(
            mean_theta=mean_theta,
            logvar_theta=logvar_theta,
            variance_theta=variance,
        ) 