from abc import ABC, abstractmethod
from colorsys import hsv_to_rgb
from dataclasses import dataclass
from functools import lru_cache
from typing import TypeVar

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.v2 import CenterCrop, Compose, Lambda

from spatialreasoners.type_extensions import ConditioningCfg, Stage

from ..dataset_image import DatasetImage, DatasetImageCfg
from .dataset_counting_polygons_base import (
    ConditioningCfg,
    DatasetCountingPolygonsBase,
    DatasetCountingPolygonsCfg,
)


@dataclass(frozen=True, kw_only=True)
class DatasetCountingPolygonsSubdatasetCfg(DatasetCountingPolygonsCfg):
    hsv_saturation: float = 1.0
    hsv_value: float = 0.9

    @property
    @abstractmethod
    def subdataset_cfg(self) -> DatasetImageCfg:
        pass


T = TypeVar("T", bound=DatasetCountingPolygonsSubdatasetCfg)


class DatasetCountingPolygonsSubdataset(DatasetCountingPolygonsBase[T], ABC):
    color_histogram_blur_sigma: float = 26.0
    color_histogram_blur_kernel_size: int = 255  # 256 possible values

    @property
    @abstractmethod
    def dataset_class(self) -> type[DatasetImage]:
        pass

    @property
    @lru_cache(maxsize=None)
    def blur_kernel(self) -> torch.Tensor:
        kernel_size = 255  # 256 possible values
        sigma = self.color_histogram_blur_sigma
        blur_kernel = torch.exp(
            -((torch.arange(kernel_size) - kernel_size // 2) ** 2) / (2 * sigma**2)
        )
        blur_kernel /= blur_kernel.sum()

        return blur_kernel

    def _get_color(
        self, rng: np.random.Generator | None, base_image: Image.Image
    ) -> str | tuple[int, int, int]:
        h, _, _ = base_image.convert("HSV").split()

        hue_histogram = torch.tensor(h.histogram(), dtype=torch.float32)
        padding = self.color_histogram_blur_kernel_size // 2

        padded_histogram = F.pad(
            hue_histogram[None, None, :], (padding, padding), mode="circular"
        )
        histogram = F.conv1d(padded_histogram, self.blur_kernel[None, None, :])

        hue = torch.argmin(histogram[0, 0]).item() / 255

        value = self.cfg.hsv_value
        saturation = self.cfg.hsv_saturation

        rgb = hsv_to_rgb(hue, saturation, value)
        rgb = (
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255),
        ) # Scale to 0-255

        return rgb

    def __init__(
        self,
        cfg: DatasetCountingPolygonsSubdatasetCfg,
        conditioning_cfg: ConditioningCfg,
        stage: Stage,
    ) -> None:
        super().__init__(cfg, conditioning_cfg, stage)

        assert cfg.subdataset_cfg is not None, "subdataset_cfg not defined"
        self.subdataset = self._load_subdataset()
        self.subdataset_image_resize = Compose(
            [
                Lambda(
                    lambda pil_image: self.relative_resize(
                        pil_image, self.cfg.image_resolution
                    )
                ),
                CenterCrop(self.cfg.image_resolution),
            ]
        )

    def _load_subdataset(self):
        return self.dataset_class(
            cfg=self.cfg.subdataset_cfg,
            conditioning_cfg=self.conditioning_cfg,
            stage=self.stage,
        )

    def _split_idx(self, idx) -> tuple[int, int, int]:
        """Loop over both datasets, and loops the smaller one"""
        subdataset_idx = idx % self.subdataset._num_available
        num_circles_idx, circles_image_idx = self.split_circles_idx(
            idx % self._num_overlay_images
        )

        return num_circles_idx, circles_image_idx, subdataset_idx

    def _get_base_image(self, subdataset_idx):
        image = self.subdataset._load(subdataset_idx)["image"]

        return self.subdataset_image_resize(image).convert("RGBA")

    @property
    def _num_available(self) -> int:
        return max(self._num_overlay_images, self.subdataset._num_available)
