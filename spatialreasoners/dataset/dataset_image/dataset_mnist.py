from dataclasses import dataclass

from torchvision.datasets import MNIST

from spatialreasoners.type_extensions import ConditioningCfg, Stage

from .. import register_dataset
from . import DatasetImage, DatasetImageCfg
from .type_extensions import ImageExample


@dataclass(frozen=True, kw_only=True)
class DatasetMnistCfg(DatasetImageCfg):
    pass


@register_dataset("mnist", DatasetMnistCfg)
class DatasetMnist(DatasetImage[DatasetMnistCfg]):
    includes_download = True
    dataset: MNIST

    def __init__(
        self, 
        cfg: DatasetMnistCfg,
        conditioning_cfg: ConditioningCfg,
        stage: Stage
    ) -> None:
        super().__init__(cfg, conditioning_cfg, stage)
        self.dataset = MNIST(
            self.cfg.root, 
            train=self.stage=="train",
            download=True
        )

    def _load(self, idx: int) -> ImageExample:
        return {"image": self.dataset[idx][0]}
    
    @property
    def _num_available(self) -> int:
        return len(self.dataset)
