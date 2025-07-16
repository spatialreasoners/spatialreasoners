import os
from dataclasses import dataclass

from torchvision.datasets import ImageFolder

from spatialreasoners.type_extensions import ConditioningCfg, Stage

from .. import register_dataset
from . import DatasetImage, DatasetImageCfg
from .latent_folder import LatentFolder
from .type_extensions import ImageExample


@dataclass(frozen=True, kw_only=True)
class DatasetImageNetCfg(DatasetImageCfg):
    pass


@register_dataset("imagenet", DatasetImageNetCfg)
class DatasetImageNet(DatasetImage[DatasetImageNetCfg]):
    num_classes: int = 1000

    def __init__(
        self, 
        cfg: DatasetImageNetCfg,
        conditioning_cfg: ConditioningCfg,
        stage: Stage
    ) -> None:
        super().__init__(cfg, conditioning_cfg, stage)
        split = "val" if stage == "test" else stage
        if split == "train" and cfg.use_saved_latents:
            assert cfg.latent_root is not None
            self.dataset = LatentFolder(
                cfg.latent_root / split, horizontal_flip=cfg.augment
            )
        else:
            self.dataset = ImageFolder(self.cfg.root / split)

    def _load(self, idx: int) -> ImageExample:
        res: ImageExample = {}
        if self.stage == "train" and self.cfg.use_saved_latents:
            latent, label = self.dataset[idx]
            res["latent"] = latent
        else:
            image, label = self.dataset[idx]
            res["image"] = image
        res["label"] = label
        res["path"] = os.path.relpath(
            os.path.splitext(self.dataset.samples[idx][0])[0], 
            start=self.cfg.root
        )
        return res
    
    @property
    def _num_available(self) -> int:
        return len(self.dataset)
