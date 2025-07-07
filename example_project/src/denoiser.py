"""
Spiral Denoiser Implementation for SpatialReasoners
"""

from dataclasses import dataclass
import torch
from torch import nn

import spatialreasoners as sr
from spatialreasoners.denoising_model import denoiser
from .tokenizer import SpiralTokenizer, SpatialDenoiserInputs, SpatialDenoiserOutputs


@dataclass(frozen=True, kw_only=True)
class SpiralDenoiserCfg(denoiser.DenoiserCfg):
    num_hidden_layers: int = 2
    num_hidden_units: int = 128
    dropout_rate: float = 0.1


@denoiser.register_denoiser("spiral_mlp", SpiralDenoiserCfg)
class SpiralDenoiser(denoiser.Denoiser[SpiralDenoiserCfg, SpatialDenoiserInputs, SpatialDenoiserOutputs]):    
    
    def __init__(
        self,
        cfg: SpiralDenoiserCfg,
        tokenizer: SpiralTokenizer,
        num_classes: int | None = None,
    ):
        super().__init__(cfg, tokenizer, num_classes)
        
        # Calculate input/output dimensions directly
        # Spiral data has 3 variables (x, y, color)
        num_variables = 3
        d_input = num_variables * 2  # value and time
        d_output = num_variables * (1 + tokenizer.predict_uncertainty + tokenizer.predict_variance)
        
        hidden_layers = []
        for _ in range(cfg.num_hidden_layers):
            hidden_layers.append(nn.Linear(cfg.num_hidden_units, cfg.num_hidden_units))
            hidden_layers.append(nn.GELU())
            hidden_layers.append(nn.Dropout(cfg.dropout_rate))
        
        self.mlp = nn.Sequential(
            nn.Linear(d_input, cfg.num_hidden_units),
            nn.GELU(),
            *hidden_layers,
            nn.Linear(cfg.num_hidden_units, d_output)
        )
        
    @property
    def d_input(self) -> int:
        # Spiral data has 3 variables (x, y, color)
        return 3 * 2  # value and time
    
    @property
    def d_output(self) -> int:
        # Spiral data has 3 variables (x, y, color)
        return 3 * (1 + self.tokenizer.predict_uncertainty + self.tokenizer.predict_variance)
    
    def freeze_time_embedding(self) -> None:
        """Freeze time embedding parameters. No-op for MLP denoiser."""
        pass
    
    def forward(self, model_inputs: SpatialDenoiserInputs, sample: bool = False) -> SpatialDenoiserOutputs:
        # model_inputs is already flattened from tokenizer: [batch, time*num_variables*dim + time*num_variables]
        batch_size = model_inputs.shape[0]
        
        # Forward through MLP (model_inputs are already flattened)
        outputs = self.mlp(model_inputs)
        
        # Spiral data has 3 variables (x, y, color)
        num_variables = 3
        output_dim = outputs.shape[1] // num_variables
        
        # Reshape to (batch, num_variables, output_dim)
        outputs = outputs.reshape(batch_size, num_variables, output_dim)
        
        return outputs 