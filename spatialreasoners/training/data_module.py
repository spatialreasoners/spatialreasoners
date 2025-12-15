from dataclasses import dataclass
from pathlib import Path

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset

from spatialreasoners.benchmark import Benchmark, BenchmarkCfg
from spatialreasoners.dataset import DatasetCfg, get_dataset, get_dataset_class
from spatialreasoners.env import DEBUG
from spatialreasoners.misc.step_tracker import StepTracker
from spatialreasoners.type_extensions import ConditioningCfg
from spatialreasoners.variable_mapper import VariableMapper


@dataclass
class DataLoaderStageCfg:
    batch_size: int
    num_workers: int
    persistent_workers: bool


@dataclass
class DataLoaderCfg:
    train: DataLoaderStageCfg | None = None
    test: DataLoaderStageCfg | None = None
    val: DataLoaderStageCfg | None = None


class DataModule(LightningDataModule):
    dataset_cfg: DatasetCfg
    data_loader_cfg: DataLoaderCfg
    conditioning_cfg: ConditioningCfg
    validation_benchmark_cfgs: dict[str, BenchmarkCfg] | None
    test_benchmark_cfgs: dict[str, BenchmarkCfg] | None
    step_tracker: StepTracker | None

    def __init__(
        self,
        dataset_cfg: DatasetCfg,
        data_loader_cfg: DataLoaderCfg,
        conditioning_cfg: ConditioningCfg,
        variable_mapper: VariableMapper,
        output_dir: Path | str,
        validation_benchmark_cfgs: dict[str, BenchmarkCfg] | None = None,
        test_benchmark_cfgs: dict[str, BenchmarkCfg] | None = None,
        step_tracker: StepTracker | None = None,
    ) -> None:
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.data_loader_cfg = data_loader_cfg
        self.conditioning_cfg = conditioning_cfg
        self.variable_mapper = variable_mapper
        self.output_dir = Path(output_dir)

        self.step_tracker = step_tracker

        self.validation_benchmark_cfgs = validation_benchmark_cfgs
        self.test_benchmark_cfgs = test_benchmark_cfgs

        self.validation_benchmarks = None  # uninitialized
        self.test_benchmarks = None  # uninitialized

    def get_persistent(self, loader_cfg: DataLoaderStageCfg) -> bool | None:
        return None if loader_cfg.num_workers == 0 else loader_cfg.persistent_workers

    def prepare_data(self):
        # Download data in single process
        if get_dataset_class(self.dataset_cfg).includes_download:
            get_dataset(self.dataset_cfg, self.conditioning_cfg, "train")

        for benchmark_cfg in self.validation_benchmark_cfgs.values():
            if get_dataset_class(benchmark_cfg.dataset).includes_download:
                get_dataset(benchmark_cfg.dataset, self.conditioning_cfg, "val")

        for benchmark_cfg in self.test_benchmark_cfgs.values():
            if get_dataset_class(benchmark_cfg.dataset).includes_download:
                get_dataset(benchmark_cfg.dataset, self.conditioning_cfg, "test")

    def setup_val_evaluations(self):
        if self.validation_benchmark_cfgs is not None:
            self.validation_benchmarks = [
                Benchmark(
                    cfg,
                    conditioning_cfg=self.conditioning_cfg,
                    stage="val",
                    variable_mapper=self.variable_mapper,
                    tag=tag,
                    output_dir=self.output_dir,
                )
                for tag, cfg in self.validation_benchmark_cfgs.items()
            ]
        else:
            self.validation_benchmarks = []

    def setup_test_evaluations(self):
        if self.test_benchmark_cfgs is not None:
            self.test_benchmarks = [
                Benchmark(
                    cfg,
                    conditioning_cfg=self.conditioning_cfg,
                    stage="test",
                    variable_mapper=self.variable_mapper,
                    tag=tag,
                    output_dir=self.output_dir,
                )
                for tag, cfg in self.test_benchmark_cfgs.items()
            ]
        else:
            self.test_benchmarks = []

    def setup(self, stage: str):
        if stage == "fit":
            self.train_data = get_dataset(
                self.dataset_cfg, self.conditioning_cfg, "train"
            )
            self.setup_val_evaluations()
        elif stage == "validate":
            self.setup_val_evaluations()
        elif stage == "test":
            self.setup_test_evaluations()

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            self.data_loader_cfg.train.batch_size,
            shuffle=True if not isinstance(self.train_data, IterableDataset) else None,
            num_workers=self.data_loader_cfg.train.num_workers,
            drop_last=not DEBUG,  # drop last incomplete batch to avoid recompilation
            persistent_workers=self.get_persistent(self.data_loader_cfg.train),
        )

    def val_dataloader(self):
        if self.validation_benchmarks is not None and self.validation_benchmarks:
            return [DataLoader(
                    val_benchmark.dataset,
                    self.data_loader_cfg.val.batch_size,
                    num_workers=self.data_loader_cfg.val.num_workers,
                    persistent_workers=self.get_persistent(self.data_loader_cfg.val),
                )
                for val_benchmark in self.validation_benchmarks
            ]
        return []

    def test_dataloader(self):
        if self.test_benchmarks is not None and self.test_benchmarks:
            return [
                DataLoader(
                    test_benchmark.dataset,
                    self.data_loader_cfg.test.batch_size,
                    num_workers=self.data_loader_cfg.test.num_workers,
                    persistent_workers=self.get_persistent(self.data_loader_cfg.test),
                )
                for test_benchmark in self.test_benchmarks
            ]
        return []
