from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Iterator, Sequence, TypeVar

from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import IterableDataset as TorchIterableDataset

from ..type_extensions import ConditioningCfg, Stage, UnstructuredExample


@dataclass(frozen=True, kw_only=True)
class DatasetCfg:
    subset_size: int | None
    data_shape: Sequence[int] # without batch dimension

T = TypeVar("T", bound=DatasetCfg)


class Dataset(Generic[T], ABC):
    includes_download: bool = False
    cfg: T
    conditioning_cfg: ConditioningCfg
    stage: Stage
    
    def __init__(
        self,
        cfg: T,
        conditioning_cfg: ConditioningCfg,
        stage: Stage
    ) -> None:
        self.cfg = cfg
        self.conditioning_cfg = conditioning_cfg
        self.stage = stage
    
    @property
    @abstractmethod
    def _num_available(self) -> int:
        pass

    def __len__(self) -> int:
        if self.cfg.subset_size is not None:
            return self.cfg.subset_size
        return self._num_available
    
    @property 
    def data_shape(self) -> Sequence[int]:
        return self.cfg.data_shape
    
    
class IndexedDataset(Dataset[T], TorchDataset, ABC):
    @abstractmethod
    def __getitem__(self, idx: int) -> UnstructuredExample:
        pass

    
class IterableDataset(Dataset[T], TorchIterableDataset, ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[UnstructuredExample]:
        pass
    
    
    
    