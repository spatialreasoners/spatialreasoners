from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

import hydra
from pytorch_lightning.loggers import Logger

from spatialreasoners.denoising_model import DenoisingModel
from spatialreasoners.type_extensions import BatchUnstructuredExample, Stage
from spatialreasoners.variable_mapper import VariableMapper

from .type_extensions import MetricsBatch


@dataclass(frozen=True, kw_only=True)
class EvaluationCfg:
    deterministic: bool = True


T = TypeVar("T", bound=EvaluationCfg)


class Evaluation(Generic[T], ABC):
    def __init__(
        self,
        cfg: T,
        variable_mapper: VariableMapper,
        tag: str,
        output_dir: Path,
        stage: Stage = "test",
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.variable_mapper = variable_mapper
        
        self.output_dir = output_dir
        
        assert stage != "train", "Evaluation is not supported in train mode"
        self.stage = stage
        self.tag = tag

    @property
    def output_path(self) -> Path:
        return self.output_dir / self.stage / self.tag
    
    @abstractmethod
    def __call__(
        self,
        denoising_model: DenoisingModel,
        batch: BatchUnstructuredExample,
        batch_idx: int,
        logger: Logger,
    ) -> Iterator[MetricsBatch] | None:
        """ Evaluate the model on the given batch
        """
        pass
