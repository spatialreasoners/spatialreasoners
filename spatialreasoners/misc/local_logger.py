import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
from jaxtyping import UInt8
from PIL import Image
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities import rank_zero_only
from torchvision.io import write_video


class LocalLogger(Logger):
    def __init__(self, output_dir: Path = Path("outputs/local")) -> None:
        super().__init__()
        self.experiment = None
        self.output_dir = output_dir / self.name
        
        os.system(f"rm -r {self.output_dir}")

    @property
    def name(self):
        return "LocalLogger"

    @property
    def version(self):
        return 0

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        pass

    @rank_zero_only
    def log_image(
        self,
        key: str,
        images: list[Any] | UInt8[np.ndarray, "batch 3 height width"],
        step: Optional[int] = None,
        caption: Optional[list[str]] = None,
        **kwargs,
    ):
        # The function signature is the same as the wandb logger's, but the step is
        # actually required.
        assert step is not None
        if caption is not None:
            assert len(images) == len(caption)
        for index, image in enumerate(images):
            c = f"{index:0>2}" if caption is None else caption[index]
            path = self.output_dir / f"{key}/{step:0>6}_{c}.png"
            path.parent.mkdir(exist_ok=True, parents=True)
            if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[0] in (3, 4):
                image = image.transpose(1, 2, 0)
            
            Image.fromarray(image).save(path)

    @rank_zero_only
    def log_video(
        self,
        key: str,
        videos: list[UInt8[np.ndarray, "frame 3 height width"]],
        step: Optional[int] = None,
        caption: Optional[list[str]] = None,
        format: Optional[list[str]] = None,
        **kwargs,
    ):
        # The function signature is the same as the wandb logger's, but the step is
        # actually required.
        n = len(videos)
        for k, v in kwargs.items():
            if len(v) != n:
                raise ValueError(f"Expected {n} items but only found {len(v)} for {k}")
        kwarg_list = [{k: kwargs[k][i] for k in kwargs} for i in range(n)]
        assert step is not None
        if caption is not None:
            assert len(videos) == len(caption)
        if format is None:
            # Uses mp4 as default format
            format = len(videos) * ["mp4"]
        else:
            assert len(videos) == len(format)
        for index, video in enumerate(videos):
            c = f"{index:0>2}" if caption is None else caption[index]
            path = self.output_dir / f"{key}/{step:0>6}_{c}.{format[index]}"
            path.parent.mkdir(exist_ok=True, parents=True)
            write_video(str(path), video.transpose(0, 2, 3, 1), **kwarg_list[index])
