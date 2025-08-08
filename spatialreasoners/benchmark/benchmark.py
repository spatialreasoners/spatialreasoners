
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

from spatialreasoners.dataset import DatasetCfg, get_dataset
from spatialreasoners.type_extensions import ConditioningCfg, Stage
from spatialreasoners.variable_mapper import VariableMapper

from .evaluation import EvaluationCfg, get_evaluation


@dataclass(frozen=True, kw_only=True)
class BenchmarkCfg:
    dataset: DatasetCfg
    evaluation: EvaluationCfg
    
T = TypeVar("T", bound=BenchmarkCfg)


class Benchmark(Generic[T]):
    """Benchmark class for evaluating models.
    This class is a wrapper around the dataset and evaluation classes.
    """
    
    def __init__(
        self,
        cfg: T,
        stage: Stage,
        conditioning_cfg: ConditioningCfg,
        variable_mapper: VariableMapper,
        tag: str,
        output_dir: Path,
    ) -> None:
        self.cfg = cfg
        self.dataset = get_dataset(cfg.dataset, conditioning_cfg=conditioning_cfg, stage=stage)
        self.evaluation = get_evaluation(cfg.evaluation, variable_mapper, stage=stage, tag=tag, output_dir=output_dir)
        