from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
from einops import rearrange
from jaxtyping import Bool, Float, Int
from torch import Tensor

from ...denoiser.u_vit3d.type_extensions import UViT3DInputs, UViT3DOutputs
from .. import register_tokenizer
from ..tokenizer import Tokenizer, TokenizerCfg, TokenizerOutputs


@dataclass(frozen=True, kw_only=True)
class PoseVideoUViTTokenizerCfg(TokenizerCfg):
    pass


@register_tokenizer("pose_video_u_vit", PoseVideoUViTTokenizerCfg)
class PoseVideoUViTTokenizer(
    Tokenizer[PoseVideoUViTTokenizerCfg, UViT3DInputs, UViT3DOutputs]
):
    @property
    def model_input_shape(self) -> Sequence[int]:
        return self.variable_mapper.unstructured_sample_shape

    @property
    def external_conditioning_shape(self) -> Sequence[int]:
        return self.variable_mapper.pose_conditioning_shape

    def variables_to_model_inputs(
        self,
        z_t: Float[Tensor, "batch time num_variables dim"],
        t: Float[Tensor, "batch time num_variables"],
        should_denoise: Bool[Tensor, "batch time num_variables"] | None = None,
        mask: None = None,
        x_masked: None = None,
        label: None = None,
        label_mask: None = None,
        fixed_conditioning_fields: dict[str, Any] | None = None,
    ) -> UViT3DInputs:  # Any shape the denoiser might need
        """
        Convert variables to model inputs.
        """
        assert (
            z_t.shape[1] == 1 and t.shape[1] == 1
        ), "No multi-time inference supported (frames should be in the variable dimension)"

        z_t = rearrange(
            z_t,
            "b 1 frames (c w h) -> b frames c w h",
            frames=self.variable_mapper.num_variables,
            c=self.variable_mapper.num_image_channels,
            w=self.variable_mapper.unstructured_sample_shape[2],
            h=self.variable_mapper.unstructured_sample_shape[3],
        )
        
        t = t.squeeze(1)

        # TODO extra conditioning
        return UViT3DInputs(
            x=z_t,
            noise_levels=t,
            external_cond_mask=None,
            external_cond=fixed_conditioning_fields['pos_encodings'],
        )

    def model_outputs_to_variable_predictions(
        self,
        model_outputs: UViT3DOutputs,
        batch_size: int,
        num_times: int,
    ) -> TokenizerOutputs:
        x = model_outputs
        x_variables = self.variable_mapper.unstructured_tensor_to_variables(x)
        x_variables.unsqueeze_(1) # [B, 1, V, F]
        
        return TokenizerOutputs(
            mean_theta=x_variables,
            variance_theta=None,
            logvar_theta=None,
        )
