from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool
from PIL import Image

from .. import register_dataset
from . import DatasetImage, DatasetImageCfg
from .type_extensions import ImageExample


@dataclass(frozen=True, kw_only=True)
class DatasetEvenPixelsCfg(DatasetImageCfg):
    saturation: float = 1.0
    value: float = 0.7
    dataset_size: int = 1000_000
    root: None = None # We don't need a root for this dataset


@register_dataset("even_pixels", DatasetEvenPixelsCfg)
class DatasetEvenPixels(DatasetImage[DatasetEvenPixelsCfg]):
    @staticmethod
    def _get_even_binary_mask(
        w: int, h: int, rng: np.random.Generator | None
    ) -> Bool[np.ndarray, "h w"]:
        num_ones = int(w * h / 2)
        flat_mask = np.zeros(w * h)
        flat_mask[:num_ones] = 1

        if rng is not None:
            rng.shuffle(flat_mask)
        else:
            np.random.shuffle(flat_mask)

        return flat_mask.astype(bool).reshape(w, h)

    def _get_image(self, rng: np.random.Generator | None) -> Image.Image:
        w, h = self.cfg.image_resolution

        if rng is not None:
            hue_offset = rng.uniform(0, 0.5)

        else:
            hue_offset = np.random.uniform(0, 0.5)

        data = np.zeros((w, h, 3))
        data[:, :, 0] = (self._get_even_binary_mask(w, h, rng) * 0.5) + hue_offset
        data[:, :, 1] = self.cfg.saturation
        data[:, :, 2] = self.cfg.value

        return Image.fromarray(np.uint8(data * 255), mode="HSV").convert("RGB")

    def _load(self, idx: int) -> ImageExample:
        if self.stage == "train":
            rng = None
        else:
            rng = np.random.default_rng(idx)

        image = self._get_image(rng)

        return {"image": image}

    @property
    def _num_available(self) -> int:
        return self.cfg.dataset_size
