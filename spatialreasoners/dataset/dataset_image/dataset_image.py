from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from math import exp, prod
from pathlib import Path
from typing import TypeVar

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torch.nn.functional import conv2d
from torchvision.transforms import ToTensor
from torchvision.transforms.v2 import (
    RGB,
    CenterCrop,
    Compose,
    Grayscale,
    Lambda,
    Normalize,
    RandomHorizontalFlip,
    ToTensor,
)

from spatialreasoners.type_extensions import ConditioningCfg, Stage, UnstructuredExample

from ..dataset import DatasetCfg, IndexedDataset
from .type_extensions import ImageExample


@dataclass(kw_only=True, frozen=True)
class DatasetImageCfg(DatasetCfg):
    subset_size: int | None
    augment: bool = False
    root: Path | str
    use_saved_latents: bool = False
    latent_root: Path | None = None
    
    @property
    def image_resolution(self) -> tuple[int, int]:
        return self.data_shape[1], self.data_shape[2] # height, width
    
    @property
    def is_grayscale(self) -> bool:
        return self.data_shape[0] == 1 


T = TypeVar("T", bound=DatasetImageCfg)


class DatasetImage(IndexedDataset[T], ABC):
    includes_download: bool = False
    num_classes: int | None = None
    cfg: T
    conditioning_cfg: ConditioningCfg
    stage: Stage

    def __init__(
        self,
        cfg: T,
        conditioning_cfg: ConditioningCfg,
        stage: Stage,
    ) -> None:
        super().__init__(cfg=cfg, conditioning_cfg=conditioning_cfg, stage=stage)
        # Define transforms
        transforms = [
            Lambda(lambda pil_image: self.relative_resize(pil_image, self.cfg.image_resolution)),
            CenterCrop(self.cfg.image_resolution),
            ToTensor()
        ]
        if self.cfg.augment:
            transforms.insert(2, RandomHorizontalFlip())
        self.transform = Compose(transforms)
        self.rgb_transform = Compose([
            Grayscale() if cfg.is_grayscale else RGB(),
            Normalize(mean=self.num_channels * [0.5], std=self.num_channels * [0.5], inplace=True)
        ])
        
    @property
    def num_channels(self) -> int:
        return self.data_shape[0]

    @staticmethod
    def relative_resize(
        image: Image.Image, 
        target_shape: Sequence[int]
    ) -> Image.Image:
        target_shape = np.asarray(target_shape[::-1])
        while np.all(np.asarray(image.size) >= 2 * target_shape):
            image = image.resize(
                tuple(x // 2 for x in image.size), 
                resample=Image.Resampling.BOX
            )

        scale = np.max(target_shape / np.asarray(image.size))
        image = image.resize(
            tuple(round(x * scale) for x in image.size), 
            resample=Image.Resampling.BICUBIC
        )
        return image

    @staticmethod
    def _concat_mask(
        image: Image.Image,
        mask: Image.Image | Float[np.ndarray, "height width"]
    ) -> Image.Image:
        assert image.mode in ("L", "RGB")
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(np.uint8(255 * mask), mode="L")
        else:
            assert mask.mode == "L"
        if image.mode == "L":
            return Image.merge("LA", (image, mask))
        r, g, b = image.split()
        return Image.merge("RGBA", (r, g, b, mask))

    @abstractmethod
    def _load(self, idx: int, **kwargs) -> ImageExample:
        """
        NOTE image of Example must include a mask as alpha channel 
        (LA or RGBA mode) if conditioning_cfg.mask
        """
        pass

    def __getitem__(self, idx: int) -> UnstructuredExample:
        sample = self._load(idx)
        res = {"in_dataset_index": idx}
        if self.stage == "train" and self.cfg.use_saved_latents:
            res["cached_latent"] = torch.from_numpy(sample["latent"])
        else:
            is_mask_given = sample["image"].mode in ("LA", "RGBA")
            assert not self.conditioning_cfg.mask or is_mask_given, "Mask conditioning but no mask given"
            res["z_t"] = self.transform(sample["image"])
            if is_mask_given:
                res["mask"] = res["z_t"][-1:]
                res["z_t"] = res["z_t"][:-1]
                res["mask"].round_()
            res["z_t"] = self.rgb_transform(res["z_t"])
        if "path" in sample:
                res["path"] = sample["path"]
        if self.conditioning_cfg.label:
            res["label"] = sample["label"]
        return UnstructuredExample(**res)



