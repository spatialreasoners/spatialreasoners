from spatialreasoners.registry import Registry
from spatialreasoners.type_extensions import ConditioningCfg, Stage

from .dataset import Dataset, DatasetCfg, IndexedDataset, IterableDataset

_dataset_registry = Registry(Dataset, DatasetCfg)
get_dataset_class = _dataset_registry.get
register_dataset = _dataset_registry.register

def get_dataset(
    cfg: DatasetCfg,
    conditioning_cfg: ConditioningCfg,
    stage: Stage,
) -> Dataset:
    return _dataset_registry.build(cfg, conditioning_cfg, stage)



from . import dataset_image, dataset_video

__all__ = [
    "get_dataset", "get_dataset_class",
    "register_dataset", "Dataset", "DatasetCfg",
    "IterableDataset", "IndexedDataset",
] + dataset_image.__all__ + dataset_video.__all__
