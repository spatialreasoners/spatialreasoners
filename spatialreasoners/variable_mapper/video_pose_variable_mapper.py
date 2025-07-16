import math
from dataclasses import dataclass, field
from typing import Sequence, TypeVar

import torch
from einops import rearrange
from jaxtyping import Bool, Float
from torch import Tensor
from torch.nn import functional as F

from . import register_variable_mapper
from .variable_mapper import VariableMapper, VariableMapperCfg


@dataclass(kw_only=True, frozen=True)
class VideoPoseVariableMapperCfg(VariableMapperCfg):
    # Might make sense to increase sigma for larger resolutions
    dependency_matrix_sigma: float = 2.0
    autoencoder: None = None
    ray_encoding_channels: int = 180


T = TypeVar("T", bound=VideoPoseVariableMapperCfg)


@register_variable_mapper("video_pose", VideoPoseVariableMapperCfg)
class VideoPoseVariableMapper(VariableMapper[T]):
    def __init__(self, cfg: T, unstructured_sample_shape: Sequence[int]) -> None:
        super().__init__(cfg, unstructured_sample_shape)

        assert (
            cfg.autoencoder is None
        ), "Autoencoder not supported for video pose variable mapper"

        assert (
            len(unstructured_sample_shape) == 4
        ), f"Unstructured sample shape must be frames, channels, height, width, but got {unstructured_sample_shape}"

    def unstructured_tensor_to_variables(
        self, x: Float[Tensor, "batch frames channels height width"]
    ) -> Float[Tensor, "batch frames features"]:
        return x.flatten(start_dim=2)

    def variables_tensor_to_unstructured(
        self, x: Float[Tensor, "batch frames features"]
    ) -> Float[Tensor, "batch frames channels height width"]:
        return x.reshape(x.shape[0], *self.unstructured_sample_shape)

    def mask_unstructured_tensor_to_variables(
        self,
        mask: (
            Float[Tensor, "batch frames 1 height width"]
            | Bool[Tensor, "batch frames 1 height width"]
        ),
    ) -> Float[Tensor, "batch frames"] | Bool[Tensor, "batch frames"]:
        mask_rearranged = rearrange(
            mask,
            "b f 1 h w -> b f (h w)",
        )

        return (
            mask_rearranged.any(dim=-1)
            if mask.dtype == torch.bool
            else mask_rearranged.mean(dim=-1)
        )

    def mask_variables_tensor_to_unstructured(
        self,
        mask: (
            Float[Tensor, "batch num_variables"] | Bool[Tensor, "batch num_variables"]
        ),
    ) -> (
        Float[Tensor, "batch frames 1 height width"]
        | Bool[Tensor, "batch frames 1 height width"]
    ):
        # Reshape to b frames 1 height width
        reshaped = mask.reshape(
            *mask.shape[:2], 1, 1, 1
        )  # Add dummy dimensions for channels, height and width
        expanded = reshaped.repeat(1, 1, 1, *self.unstructured_sample_shape[2:])
        return expanded

    @property
    def num_variables(self) -> int:
        return self.unstructured_sample_shape[0]  # num frames

    @property
    def num_features(self) -> int:
        return math.prod(self.unstructured_sample_shape[1:])

    @property
    def num_image_channels(self) -> int:
        return self.unstructured_sample_shape[1]

    @property
    def pose_conditioning_shape(self) -> torch.Size:
        return torch.Size(
            [
                self.num_variables,
                self.cfg.ray_encoding_channels,
                self.unstructured_sample_shape[2],
                self.unstructured_sample_shape[3],
            ]
        )

    def _calculate_dependency_matrix(
        self,
    ) -> Float[Tensor, "num_variables num_variables"]:
        N = self.num_variables
        indices = torch.arange(N)
        abs_diff = torch.abs(indices[:, None] - indices[None, :])
        return torch.exp(-self.cfg.dependency_matrix_sigma * abs_diff)
        # "The Default dependency matrix is based on the locality assumption"
        # dep_matrix = torch.eye(self.num_variables)  # Initialy all patches are independent

        # # Then we blur the dependency matrix

        # # Create a 1D Gaussian kernel
        # kernel_size = 2 * self.cfg.dependency_matrix_sigma + 1
        # x = torch.linspace(-self.cfg.dependency_matrix_sigma, self.cfg.dependency_matrix_sigma, kernel_size)
        # kernel = torch.exp(-x**2 / (2 * self.cfg.dependency_matrix_sigma**2))
        # kernel = kernel / kernel.sum()

        # blurred = F.conv1d(dep_matrix, kernel[None], padding="same")
        # return blurred


