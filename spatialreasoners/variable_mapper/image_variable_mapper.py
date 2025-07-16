from dataclasses import dataclass, field
from typing import Sequence, TypeVar

import torch
from einops import rearrange
from jaxtyping import Bool, Float
from torch import Tensor
from torch.nn import functional as F

from . import register_variable_mapper
from .autoencoder import AutoencoderCfg
from .variable_mapper import VariableMapper, VariableMapperCfg


@dataclass(kw_only=True, frozen=True)
class ImageVariableMapperCfg(VariableMapperCfg):
    variable_patch_size: int = 4
    # Might make sense to increase sigma for larger resolutions
    dependency_matrix_sigma: float = 2.0
    # TODO this should be ImageAutoencoderCfg but the autoencoder registry relies on AutoencoderCfg type for dacite type hooks!
    autoencoder: AutoencoderCfg | None = None


T = TypeVar("T", bound=ImageVariableMapperCfg)


@register_variable_mapper("image", ImageVariableMapperCfg)
class ImageVariableMapper(VariableMapper[T]):
    def __init__(
        self, cfg: T, unstructured_sample_shape: Sequence[int]
    ) -> None:
        super().__init__(cfg, unstructured_sample_shape)

        assert (
            len(self.unstructured_sample_shape) == 3
        ), f"Unstructured sample shape must be channels, height, width, but got {self.unstructured_sample_shape}"

        assert (
            self.unstructured_sample_shape[1] % self.cfg.variable_patch_size == 0
        ), f"Height {self.unstructured_sample_shape[1]} must be divisible by patch size {self.cfg.variable_patch_size}"
        assert (
            self.unstructured_sample_shape[2] % self.cfg.variable_patch_size == 0
        ), f"Width {self.unstructured_sample_shape[2]} must be divisible by patch size {self.cfg.variable_patch_size}"

        self.grid_shape = (
            self.unstructured_sample_shape[1] // self.cfg.variable_patch_size,
            self.unstructured_sample_shape[2] // self.cfg.variable_patch_size,
        )

    def unstructured_tensor_to_variables(
        self, x: Float[Tensor, "batch channels height width"]
    ) -> Float[Tensor, "batch num_variables features"]:
        return rearrange(
            x,
            "b c (h p1) (w p2) -> b (h w) (c p1 p2)",
            h=self.grid_shape[0],
            w=self.grid_shape[1],
            p1=self.cfg.variable_patch_size,
            p2=self.cfg.variable_patch_size,
        )

    def variables_tensor_to_unstructured(
        self, x: Float[Tensor, "batch num_variables features"]
    ) -> Float[Tensor, "batch channels height width"]:
        return rearrange(
            x,
            "b (hg wg) (c p1 p2) -> b c (hg p1) (wg p2)",
            hg=self.grid_shape[0],
            wg=self.grid_shape[1],
            p1=self.cfg.variable_patch_size,
            p2=self.cfg.variable_patch_size,
        )

    def mask_unstructured_tensor_to_variables(
        self,
        mask: (
            Float[Tensor, "batch 1 height width"] | Bool[Tensor, "batch 1 height width"]
        ),
    ) -> Float[Tensor, "batch num_variables"] | Bool[Tensor, "batch num_variables"]:
        mask_rearranged = rearrange(
            mask,
            "b 1 (w p1) (h p2) -> b (w h) (p1 p2)",
            w=self.grid_shape[0],
            h=self.grid_shape[1],
            p1=self.cfg.variable_patch_size,
            p2=self.cfg.variable_patch_size,
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
    ) -> Float[Tensor, "batch 1 height width"] | Bool[Tensor, "batch height width"]:
        # Reshape to b 1 w h
        reshaped = mask.reshape(
            mask.shape[0], 1, self.grid_shape[0], self.grid_shape[1]
        )
        # Use nearest neighbor interpolation to effectively broadcast
        expanded = F.interpolate(
            reshaped,
            scale_factor=(self.cfg.variable_patch_size, self.cfg.variable_patch_size),
            mode="nearest",
        )

        if self.autoencoder is not None:
            expanded = F.interpolate(expanded, scale_factor=self.autoencoder.downscale_factor, mode="nearest")

        return expanded

    @property
    def num_variables(self) -> int:
        return self.grid_shape[0] * self.grid_shape[1]

    @property
    def num_features(self) -> int:
        features_per_patch = (
            self.autoencoder.latent_shape[0]
            if self.autoencoder
            else self.unstructured_sample_shape[0]
        )
        return features_per_patch * self.cfg.variable_patch_size**2
    
    @property
    def num_image_channels(self) -> int:
        return self.unstructured_sample_shape[0]
        

    def _calculate_dependency_matrix(
        self,
    ) -> Float[Tensor, "num_variables num_variables"]:
        "The Default dependency matrix is based on the locality assumption"
        kernel = self._get_dependency_2d_gaussian_kernel()
        dep_matrix = torch.eye(
            self.num_variables, device=self.device
        )  # Initialy all patches are independent

        # (x1 y1) 1 x2 y2 as we want to blur over the last two dimensions,
        # we have to add a dummy dimension for channels
        dep_tensor = rearrange(
            dep_matrix,
            "... (x y) -> ... 1 x y",
            x=self.grid_shape[0],
            y=self.grid_shape[1],
        )
        blurred = F.conv2d(dep_tensor, kernel[None, None], padding="same")
        return rearrange(blurred, "... 1 x y -> ... (x y)")

    def _get_dependency_2d_gaussian_kernel(
        self,
    ) -> Float[Tensor, "max_grid_size max_grid_size"]:
        kernel_size = max(self.grid_shape)

        # make sure kernel size is odd
        kernel_size += 1 - kernel_size % 2

        kernel_1d = torch.tensor(
            [
                torch.exp(
                    -((x - kernel_size // 2) ** 2)
                    / (2 * self.cfg.dependency_matrix_sigma**2)
                )
                for x in range(kernel_size)
            ],
            device=self.device,
        )
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d /= kernel_2d.sum()  # Normalize

        assert kernel_2d.shape == (kernel_size, kernel_size)
        return kernel_2d
