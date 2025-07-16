from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from math import prod
from typing import TypeVar

import numpy as np
from jaxtyping import Float
from PIL import Image

from .dataset_image import DatasetImage, DatasetImageCfg
from .type_extensions import ImageExample


@dataclass(frozen=True, kw_only=True)
class DatasetGridCfg(DatasetImageCfg):
    mask_self_dependency: bool = True
    given_cells_range: Sequence[int]  # [min_inclusive, max_exclusive]


T = TypeVar("T", bound=DatasetGridCfg)


class DatasetGrid(DatasetImage[T], ABC):

    @abstractmethod
    def load_full_image(self, idx: int) -> Image.Image:
        pass

    @property
    def num_crops_per_image(self) -> int:
        return prod(self.num_crops_per_axis)

    @property
    @abstractmethod
    def _num_available(self) -> int:
        pass

    @property
    @abstractmethod
    def cell_size(self) -> tuple[int, int]:
        pass

    @property
    @abstractmethod
    def grid_size(self) -> tuple[int, int]:
        pass

    @property
    def crop_size(self) -> tuple[int, int]:
        return tuple(map(prod, zip(self.grid_size, self.cell_size)))

    @classmethod
    def _from_full_idx(cls, full_idx) -> tuple[int, int]:
        return full_idx // cls.grid_size[0], full_idx % cls.grid_size[1]

    def _get_random_masks(
        self, num_given_cells: int | None = None, rng: np.random.Generator | None = None
    ) -> Float[np.ndarray, "height width"]:
        num_cells = prod(self.grid_size)
        if num_given_cells is None:
            num_given_cells = (
                np.random.randint(0, num_cells)
                if rng is None
                else rng.integers(0, num_cells).item()
            )
        mask = np.ones(self.grid_size, dtype=bool)

        grid_idx = (np.random if rng is None else rng).choice(
            num_cells, num_given_cells, replace=False
        )
        row, col = grid_idx // self.grid_size[1], grid_idx % self.grid_size[1]
        mask[row, col] = False
        mask = np.kron(mask, np.ones(self.cell_size, dtype=bool))
        return mask.astype(float)

    def _get_mask(
        self,
        num_given_cells: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> Image.Image | Float[np.ndarray, "height width"]:
        return self._get_random_masks(
            num_given_cells,
            rng,
        )

    def _get_crop_params(self, crop_idx: int):
        row_idx, col_idx = (
            crop_idx // self.num_crops_per_axis[1],
            crop_idx % self.num_crops_per_axis[1],
        )
        top, left = row_idx * self.crop_size[0], col_idx * self.crop_size[1]
        bottom, right = top + self.crop_size[0], left + self.crop_size[1]
        return (left, top, right, bottom)

    def _load(self, idx: int) -> ImageExample:
        image_idx, crop_idx = (
            idx // self.num_crops_per_image,
            idx % self.num_crops_per_image,
        )
        image = self.load_full_image(image_idx)
        crop_params = self._get_crop_params(crop_idx)

        rng = np.random.default_rng(image_idx) if self.stage == "test" else None
        if rng:
            num_given_cells = rng.integers(
                self.cfg.given_cells_range[0], self.cfg.given_cells_range[1]
            ).item()
        else:
            num_given_cells = None

        image = image.crop(crop_params)
        if self.conditioning_cfg.mask or num_given_cells is not None:
            mask = self._get_mask(num_given_cells, rng)
            image = self._concat_mask(image, mask)
        return {"image": image}
