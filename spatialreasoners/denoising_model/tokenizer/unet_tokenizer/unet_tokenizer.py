from dataclasses import dataclass
from typing import TypeVar

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from spatialreasoners.denoising_model.denoiser.unet.type_extensions import (
    UNetModelInputs,
    UNetModelOutput,
)

from ...type_extensions import TokenizerOutputs
from .. import register_tokenizer
from ..tokenizer import Tokenizer, TokenizerCfg


@dataclass(frozen=True, kw_only=True)
class UNetTokenizerCfg(TokenizerCfg):
    concat_mask: bool = False
    base_channels: int = 3


T_CFG = TypeVar("T_CFG", bound=UNetTokenizerCfg)


@register_tokenizer("unet", UNetTokenizerCfg)
class UNetTokenizer(Tokenizer[T_CFG, UNetModelInputs, UNetModelOutput]):
    """
    UNet tokenizer.
    """

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
    ) -> UNetModelInputs:  # Any shape the denoiser might need
        batch_size, time, num_variables, dim = z_t.shape

        # Remove time dimension
        z_t = z_t.flatten(0, 1)
        t = t.flatten(0, 1)

        if label is not None:
            label = label.repeat_interleave(time, dim=1)

            if label_mask is not None:
                label_mask = label_mask.repeat_interleave(time, dim=1)

        # Move to image format
        z_t_image = self.variable_mapper.variables_tensor_to_unstructured(z_t)
        t_image = self.variable_mapper.mask_variables_tensor_to_unstructured(t)

        if self.cfg.concat_mask:
            assert mask is not None and x_masked is not None, "mask and x_masked must be provided if concat_mask is True"
            mask = mask.repeat_interleave(time, dim=0)
            mask = self.variable_mapper.mask_variables_tensor_to_unstructured(mask)

            x_masked = x_masked.repeat_interleave(time, dim=0)
            x_masked = self.variable_mapper.variables_tensor_to_unstructured(x_masked)

            z_t_image = torch.cat([z_t_image, mask, x_masked], dim=1)  # concat in channel dimension

        return UNetModelInputs(
            z_t=z_t_image,
            t=t_image,
            label=label,
            label_mask=label_mask,
        )

    def model_outputs_to_variable_predictions(
        self, 
        model_outputs: UNetModelOutput,
        batch_size: int,
        num_times: int,
    ) -> TokenizerOutputs:
        num_image_channels = self.variable_mapper.num_image_channels

        mean_theta_image = model_outputs[:, :num_image_channels]
        mean_theta = self.variable_mapper.unstructured_tensor_to_variables(
            mean_theta_image
        )
        mean_theta = mean_theta.reshape(batch_size, num_times, *mean_theta.shape[1:])
        if self.predict_variance:
            v_theta_image = model_outputs[
                :, num_image_channels : 2 * num_image_channels
            ]
            v_theta = self.variable_mapper.unstructured_tensor_to_variables(
                v_theta_image
            )
            v_theta = v_theta.reshape(batch_size, num_times, *v_theta.shape[1:])

        if self.predict_uncertainty:
            logvar_theta_image = model_outputs[:, -1].unsqueeze(1)
            logvar_theta = self.variable_mapper.unstructured_tensor_to_variables(
                logvar_theta_image
            )
            logvar_theta = logvar_theta.mean(dim=-1) # average over the spatial dimensions within variable
            logvar_theta = logvar_theta.reshape(batch_size, num_times, *logvar_theta.shape[1:])

        return TokenizerOutputs(
            mean_theta=mean_theta,
            variance_theta=v_theta,
            logvar_theta=logvar_theta,
        )

    @property
    def in_channels(self) -> int:
        num_channels = self.variable_mapper.num_image_channels

        if self.cfg.concat_mask:
            num_channels *= 2  # x_masked
            num_channels += 1  # mask

        return num_channels

    @property
    def out_channels(self) -> int:
        num_channels = self.variable_mapper.num_image_channels

        if self.predict_variance:
            num_channels *= 2  # output, v

        if self.predict_uncertainty:
            num_channels += 1

        return num_channels
