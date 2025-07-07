from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, TypeVar

import hydra
from dacite import Config, from_dict
from omegaconf import DictConfig, OmegaConf

from .benchmark import BenchmarkCfg
from .dataset import DatasetCfg
from .registry import get_type_hooks
from .training import DataLoaderCfg, SRMLightningModuleCfg
from .type_extensions import FullPrecision, HalfPrecision
from .variable_mapper import VariableMapperCfg


@dataclass(kw_only=True, frozen=True)
class CheckpointingCfg:
    load: Optional[str]  # Not a path, since it could be something like wandb://...
    every_n_train_steps: int
    every_n_train_steps_persistently: int | None = None
    save_top_k: int | None = None
    resume: bool = False
    save: bool = True


@dataclass(kw_only=True, frozen=True)
class TrainerCfg:
    max_epochs: int | None = None
    max_steps: int = -1
    val_check_interval: int | float | None = None
    log_every_n_steps: int | None = None
    task_steps: int | None = None
    accumulate_grad_batches: int = 1
    precision: FullPrecision | HalfPrecision | None = None
    num_nodes: int = 1
    validate: bool = True
    profile: bool = False
    detect_anomaly: bool = False


@dataclass(kw_only=True, frozen=True)
class TorchCfg:
    float32_matmul_precision: Literal["highest", "high", "medium"] | None = None
    cudnn_benchmark: bool = False


@dataclass(kw_only=True, frozen=True)
class WandbCfg:
    project: str
    entity: str
    activated: bool = True
    mode: Literal["online", "offline", "disabled"] = "online"
    tags: list[str] | None = None


@dataclass(kw_only=True, frozen=True)
class RootCfg(SRMLightningModuleCfg):
    mode: Literal["train", "val", "test"]
    dataset: DatasetCfg
    data_loader: DataLoaderCfg
    validation_benchmarks: dict[str, BenchmarkCfg] | None = None,
    test_benchmarks: dict[str, BenchmarkCfg] | None = None,
    variable_mapper: VariableMapperCfg
    checkpointing: CheckpointingCfg
    trainer: TrainerCfg
    seed: int | None = None
    mnist_classifier: str | None = None
    torch: TorchCfg
    wandb: WandbCfg
    
    manual_output_dir: Path | None = None
    
    @property
    def output_dir(self) -> Path:
        if self.manual_output_dir is not None:
            return self.manual_output_dir
        try: 
            return Path(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"])
        except:
            return Path("outputs") / self.mode


TYPE_HOOKS = {
    Path: Path,
}


T = TypeVar("T")


def load_typed_config(
    cfg: DictConfig,
    data_class: type[T],
    extra_type_hooks: dict = {},
) -> T:
    type_hooks = TYPE_HOOKS | extra_type_hooks
    config = Config(type_hooks=type_hooks, strict=True)
    return from_dict(
        data_class,
        OmegaConf.to_container(cfg),
        config=Config(
            type_hooks=type_hooks | get_type_hooks(config),
            strict=True
        ),
    )

def load_typed_root_config(cfg: DictConfig) -> RootCfg:
    return load_typed_config(
        cfg,
        RootCfg,
        {},
    )
