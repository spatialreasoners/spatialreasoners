from dataclasses import dataclass

from ... import register_dataset
from ..dataset_ffhq import DatasetFFHQ, DatasetFFHQCfg
from .dataset_counting_polygons_subdataset import (
    DatasetCountingPolygonsSubdataset,
    DatasetCountingPolygonsSubdatasetCfg,
)


@dataclass(frozen=True, kw_only=True)
class DatasetCountingPolygonsFFHQCfg(DatasetCountingPolygonsSubdatasetCfg):
    subdataset_cfg: DatasetFFHQCfg | None = None  # should never be None in practice


@register_dataset("counting_polygons_ffhq", DatasetCountingPolygonsFFHQCfg)
class DatasetCountingPolygonsFFHQ(
    DatasetCountingPolygonsSubdataset[DatasetCountingPolygonsFFHQCfg]
):
    dataset_class = DatasetFFHQ
