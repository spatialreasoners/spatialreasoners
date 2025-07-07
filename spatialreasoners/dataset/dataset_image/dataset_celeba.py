from dataclasses import dataclass

from torchvision.datasets import CelebA

from spatialreasoners.type_extensions import ConditioningCfg, Stage

from .. import register_dataset
from .dataset_image import DatasetImage, DatasetImageCfg
from .type_extensions import ImageExample


@dataclass(frozen=True, kw_only=True)
class DatasetCelebACfg(DatasetImageCfg):
    pass


@register_dataset("celeba", DatasetCelebACfg)
class DatasetCelebA(DatasetImage[DatasetCelebACfg]):
    includes_download = True
    dataset: CelebA

    def __init__(
        self, 
        cfg: DatasetCelebACfg,
        conditioning_cfg: ConditioningCfg,
        stage: Stage
    ) -> None:
        super().__init__(cfg, conditioning_cfg, stage)
        split = self.stage
        if split == "val":
            split = "valid"
        self.dataset = CelebA(
            self.cfg.root,
            split=split,
            download=True
        )

    def _load(self, idx: int) -> ImageExample:
        return {"image": self.dataset[idx][0]}
    
    @property
    def _num_available(self) -> int:
        return len(self.dataset)
