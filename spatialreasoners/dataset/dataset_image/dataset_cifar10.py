from dataclasses import dataclass

from torchvision.datasets import CIFAR10

from spatialreasoners.type_extensions import ConditioningCfg, Stage

from .. import register_dataset
from .dataset_image import DatasetImage, DatasetImageCfg
from .type_extensions import ImageExample


@dataclass(frozen=True, kw_only=True)
class DatasetCifar10Cfg(DatasetImageCfg):
    pass


@register_dataset("cifar10", DatasetCifar10Cfg)
class DatasetCifar10(DatasetImage[DatasetCifar10Cfg]):
    includes_download = True
    dataset: CIFAR10
    num_classes: int = 10

    def __init__(
        self, 
        cfg: DatasetCifar10Cfg,
        conditioning_cfg: ConditioningCfg,
        stage: Stage
    ) -> None:
        super().__init__(cfg, conditioning_cfg, stage)
        self.dataset = CIFAR10(
            self.cfg.root, 
            train=self.stage=="train",
            download=True
        )

    def _load(self, idx: int) -> ImageExample:
        image, label = self.dataset[idx]
        return {"image": image, "label": label}
    
    @property
    def _num_available(self) -> int:
        return len(self.dataset)
