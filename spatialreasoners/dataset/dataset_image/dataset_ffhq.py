import os
from dataclasses import dataclass

from torchvision.datasets import ImageFolder

from spatialreasoners.type_extensions import ConditioningCfg, Stage

from .. import register_dataset
from . import DatasetImage, DatasetImageCfg
from .type_extensions import ImageExample


@dataclass(frozen=True, kw_only=True)
class DatasetFFHQCfg(DatasetImageCfg):
    pass


@register_dataset("ffhq", DatasetFFHQCfg)
class DatasetFFHQ(DatasetImage[DatasetFFHQCfg]):
    def __init__(
        self, 
        cfg: DatasetFFHQCfg,
        conditioning_cfg: ConditioningCfg,
        stage: Stage
    ) -> None:
        super().__init__(cfg, conditioning_cfg, stage)
        self.dataset = ImageFolder(self.cfg.root)

    def _load(self, idx: int) -> ImageExample:
        path = os.path.relpath(self.dataset.samples[idx][0], start=self.cfg.root)
        return {"image": self.dataset[idx][0], "path": path}
    
    @property
    def _num_available(self) -> int:
        return len(self.dataset)
