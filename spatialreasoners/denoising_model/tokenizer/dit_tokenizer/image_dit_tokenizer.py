from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from einops import rearrange
from jaxtyping import Bool, Float, Int
from torch import Tensor

from spatialreasoners.variable_mapper import ImageVariableMapper

from ...denoiser.dit.type_extensions import DiTModelInputs
from ...type_extensions import TokenizerOutputs
from .. import register_tokenizer
from .dit_tokenizer import DiTTokenizer, DiTTokenizerCfg


@dataclass(frozen=True, kw_only=True)
class ImageDiTTokenizerCfg(DiTTokenizerCfg):
    num_tokens_per_spatial_dim: int = 4 # How much to subdivide the variable
    concat_mask: bool = False # Whether to concatenate the mask to the input
    

@register_tokenizer("image_dit", ImageDiTTokenizerCfg)
class ImageDiTTokenizer(DiTTokenizer[ImageDiTTokenizerCfg]):
    """
    Image DiT tokenizer.
    """

    def __init__(
        self, 
        cfg: ImageDiTTokenizerCfg, 
        variable_mapper: ImageVariableMapper,
        predict_sigma: bool = False,
        predict_v: bool = False,
        ) -> None:
        super().__init__(cfg, variable_mapper, predict_sigma, predict_v)
        self.cfg = cfg
        self.variable_mapper = variable_mapper
        
        token_coordinates_xy = self.get_grid(self.token_grid_size, indexing="xy")
        token_coordinates_ij = self.get_grid(self.token_grid_size, indexing="ij")
        # Rearrange coordinates consistenly with the contiguous tokens within variables
        # TODO make sure that height and width is actually correct or should be swapped?
        token_coordinates_xy = rearrange(
            token_coordinates_xy, 
            "(h p w q) c -> (h w p q) c",
            h=variable_mapper.grid_shape[0], 
            p=cfg.num_tokens_per_spatial_dim, 
            w=variable_mapper.grid_shape[1], 
            q=cfg.num_tokens_per_spatial_dim
        )
        token_coordinates_ij = rearrange(
            token_coordinates_ij, 
            "(h p w q) c -> (h w p q) c",
            h=variable_mapper.grid_shape[0], 
            p=cfg.num_tokens_per_spatial_dim, 
            w=variable_mapper.grid_shape[1], 
            q=cfg.num_tokens_per_spatial_dim
        )
        self.register_buffer("token_coordinates_xy", token_coordinates_xy.unsqueeze(0), persistent=False)
        self.register_buffer("token_coordinates_ij", token_coordinates_ij.unsqueeze(0), persistent=False)
        
    @staticmethod
    def get_grid(
        grid_size: Sequence[int],
        indexing: Literal["ij", "xy"] = "xy"
    ) -> Float[Tensor, "grid_volume spatial"] :
        """
        Absolute coordinate grid
        NOTE assumes grid_size to be [width, height] in 2D corresponding to xy coordinates
        """
        # TODO this indexing "xy" is not the one used for Rotary positional encoding (e.g. LightningDiT), but "ij"
        # TODO we need a proper optional conversion from pos input in xy format to ij coordinates in the positional embedding class with default conversion for Rotary
        return torch.stack(
            torch.meshgrid(
                *[torch.arange(s, dtype=torch.float) for s in grid_size],
                indexing=indexing
            ),
            dim=0
        ).flatten(1).T
        
    @property
    def model_d_data(self) -> int:
        num_spatial_dims = len(self.variable_mapper.grid_shape) # 2 for images
        
        split_factor = self.cfg.num_tokens_per_spatial_dim ** num_spatial_dims
        assert self.variable_mapper.num_features % split_factor == 0, "num_features must be divisible by num_tokens_per_spatial_dim"
        return self.variable_mapper.num_features // split_factor
    
    @property
    def model_d_in(self) -> int:
        d_in = self.model_d_data # base
        
        if self.cfg.concat_mask:
            d_in *= 2 # x_masked
            d_in += 1 # mask
        
        return d_in
    
    @property
    def model_d_out(self) -> int:
        feature_num = self.model_d_in
        
        if self.predict_v:
            feature_num += (self.variable_mapper.num_features // self.cfg.num_tokens_per_spatial_dim) * len(self.variable_mapper.grid_shape)
        
        if self.predict_uncertainty:
            feature_num += 1 # logvar per token
        
        return feature_num
    
    
    def variables_to_model_inputs(
        self, 
        z_t: Float[Tensor, "batch time num_variables dim"],
        t: Float[Tensor, "batch time num_variables"],
        should_denoise: Bool[Tensor, "batch time num_variables"] | None = None,
        mask: Float[Tensor, "batch num_variables"] | None = None,
        x_masked: Float[Tensor, "batch num_variables dim"] | None = None,
        label: Int[Tensor, "batch"] | None = None,
        label_mask: Bool[Tensor, "batch"] | None = None,
        fixed_conditioning_fields: None = None,
    ) -> DiTModelInputs:
    
        if self.cfg.concat_mask:
            assert mask is not None, "mask must be provided if concat_mask is True"
            assert x_masked is not None, "masked_x must be provided if mask is provided"
            c_cat = torch.cat((mask, x_masked), dim=1)
            z_t = torch.cat((z_t, c_cat), dim=-3)
        
        z_t_rearranged = rearrange(
            z_t,
            "b time num_variables (f h w) -> (b time) (num_variables h w) f",
            f=self.model_d_in,
            h=self.cfg.num_tokens_per_spatial_dim,
            w=self.cfg.num_tokens_per_spatial_dim,
            time=t.shape[1],
            num_variables=self.variable_mapper.num_variables,
        )
        
        
        num_t_repeats = self.cfg.num_tokens_per_spatial_dim ** len(self.variable_mapper.grid_shape)
        t_repeated = t.flatten(0, 1).repeat_interleave(num_t_repeats, dim=1)
        
        token_coordinates_xy = self.token_coordinates_xy.repeat(z_t_rearranged.shape[0], 1, 1)
        token_coordinates_ij = self.token_coordinates_ij.repeat(z_t_rearranged.shape[0], 1, 1)
        
        return DiTModelInputs(
            z_t=z_t_rearranged,
            t=t_repeated,
            token_coordinates_xy=token_coordinates_xy,
            token_coordinates_ij=token_coordinates_ij,
            label=label,
            label_mask=label_mask,
        )
    
    
    def assemble_tokens_to_variables(
        self, 
        mean_pred: Float[Tensor, "batch time num_tokens features"], # Before parametrization-specifc transformation
        logvar_pred: Float[Tensor, "batch time num_tokens"] | None, # Before parametrization-specifc transformation
        v_pred: Float[Tensor, "batch time num_tokens features"] | None,
    ) -> TokenizerOutputs:
        
        mean_pred = rearrange(
            mean_pred,
            "b t (v h w) f -> b t v (f h w)",
            v=self.variable_mapper.num_variables,
            h=self.cfg.num_tokens_per_spatial_dim,
            w=self.cfg.num_tokens_per_spatial_dim,
            f=self.model_d_in,
        )
        
        if v_pred is not None:
            v_pred = rearrange(
                v_pred,
                "b t (v h w) f -> b t v (f h w)",
                v=self.variable_mapper.num_variables,
                h=self.cfg.num_tokens_per_spatial_dim,
                w=self.cfg.num_tokens_per_spatial_dim,
                f=self.model_d_in,
            )
        
        if logvar_pred is not None:
            logvar_pred = rearrange(
                logvar_pred,
                "b t (v h w) -> b t v (h w)",
                v=self.variable_mapper.num_variables,
                h=self.cfg.num_tokens_per_spatial_dim,
                w=self.cfg.num_tokens_per_spatial_dim,
            )
            logvar_pred = logvar_pred.mean(dim=-1) # average over the spatial dimensions within variable
        
        return TokenizerOutputs(
            mean_theta=mean_pred,
            logvar_theta=logvar_pred,
            variance_theta=v_pred
        )
        
        
    @property
    def token_grid_size(self) -> Sequence[int]:
        return [
            spatial_dim * self.cfg.num_tokens_per_spatial_dim 
            for spatial_dim in self.variable_mapper.grid_shape
        ]
