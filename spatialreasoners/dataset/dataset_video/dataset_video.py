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

from ..dataset import DatasetCfg, IterableDataset


@dataclass(frozen=True, kw_only=True)
class DatasetVideoCfg(DatasetCfg):
    root: Path | str
    subset_size: int | None
    # video_path: Path
    # num_frames: int
    # frame_rate: float
    # image_size: int
    # num_workers: int
    
    
T = TypeVar("T", bound=DatasetVideoCfg)
    
class DatasetVideo(IterableDataset[T]):
    pass
    