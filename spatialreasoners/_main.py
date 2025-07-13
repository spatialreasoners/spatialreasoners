import json
import os
from datetime import datetime, timezone
from pathlib import Path

import hydra
import torch
from colorama import Fore
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.profilers import AdvancedProfiler

import wandb
from spatialreasoners.config import load_typed_root_config
from spatialreasoners.dataset import get_dataset
from spatialreasoners.env import DEBUG
from spatialreasoners.global_cfg import set_cfg
from spatialreasoners.misc.local_logger import LocalLogger
from spatialreasoners.misc.wandb_tools import update_checkpoint_path
from spatialreasoners.training import DataModule, SRMLightningModule
from spatialreasoners.variable_mapper import get_variable_mapper


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def main(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    if cfg.seed is not None:
        seed_everything(cfg.seed, workers=True)

    # Set torch variables
    if cfg.torch.float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(cfg.torch.float32_matmul_precision)
    torch.backends.cudnn.benchmark = cfg.torch.cudnn_benchmark
    
    # Set up the output directory.
    output_dir = cfg.output_dir
    print(cyan(f"Saving outputs to {output_dir}."))

    # Set up logging with wandb.
    callbacks = []
    if cfg.wandb.activated:
        logger = WandbLogger(
            project=cfg.wandb.project,
            mode=cfg.wandb.mode,
            name=f"{output_dir.parent.name} ({output_dir.name})",
            id=f"{output_dir.parent.name}_{output_dir.name}",
            tags=cfg.wandb.tags,
            log_model=False,
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
            entity=cfg.wandb.entity
        )
        callbacks.append(LearningRateMonitor("step", True))

        # On rank != 0, wandb.run is None.
        if wandb.run is not None:
            wandb.run.log_code("src")
    else:
        logger = LocalLogger(output_dir=output_dir)

    # Set up checkpointing.
    checkpoint_dir = output_dir / "checkpoints"
    if cfg.mode == "train" and cfg.checkpointing.save:
        # Always checkpoint and continue from last state
        callbacks.append(
            ModelCheckpoint(
                checkpoint_dir,
                save_last=True,
                save_top_k=1,
                every_n_train_steps=cfg.checkpointing.every_n_train_steps,
                save_on_train_epoch_end=False
            )
        )
        if cfg.checkpointing.every_n_train_steps_persistently is not None:
            # For safety checkpoint top-k w.r.t. total loss
            callbacks.append(
                ModelCheckpoint(
                    checkpoint_dir,
                    save_last=False,
                    save_top_k=-1,
                    every_n_train_steps=cfg.checkpointing.every_n_train_steps_persistently,
                    save_on_train_epoch_end=False
                )
            )
        if cfg.checkpointing.save_top_k is not None:
            # For safety checkpoint top-k w.r.t. total loss
            callbacks.append(
                ModelCheckpoint(
                    checkpoint_dir,
                    filename="epoch={epoch}-step={step}-loss={loss/total:.4f}",
                    monitor="loss/total",
                    save_last=False,
                    save_top_k=cfg.checkpointing.save_top_k,
                    auto_insert_metric_name=False,
                    every_n_train_steps=cfg.checkpointing.every_n_train_steps,
                    save_on_train_epoch_end=False
                )
            )

    # Prepare the checkpoint for loading.
    checkpoint_path = checkpoint_dir / "last.ckpt"
    if os.path.exists(checkpoint_path):
        resume = True
    else:
        # Sets checkpoint_path to None if cfg.checkpointing.load is None
        checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb.project)
        resume = cfg.checkpointing.resume
    
    # TODO find better place for this somehow?
    dataset = get_dataset(cfg=cfg.dataset, conditioning_cfg=cfg.denoising_model.conditioning, stage="train")
    num_classes = dataset.num_classes
    unstructured_sample_shape = cfg.dataset.data_shape

    variable_mapper = get_variable_mapper(cfg.variable_mapper, unstructured_sample_shape)

    srm_lightling_module = None
    step = 0
    lightning_checkpoint_available = checkpoint_path is not None and checkpoint_path.suffix == ".ckpt"
    
    if cfg.mode == "train" and lightning_checkpoint_available:
        step = torch.load(checkpoint_path)["global_step"]
        if not resume:
            # Just load model weights but no optimizer state
            srm_lightling_module = SRMLightningModule.load_from_checkpoint(
                checkpoint_path, cfg=cfg, num_classes=num_classes, variable_mapper=variable_mapper, strict=False
            )
    
    if srm_lightling_module is None:
        srm_lightling_module = SRMLightningModule(cfg=cfg,num_classes=num_classes, variable_mapper=variable_mapper)
        if checkpoint_path is not None and checkpoint_path.suffix != ".ckpt":
            assert not resume, "Cannot resume from state_dict only"
            print(f"Loading state_dict from {checkpoint_path}")
            srm_lightling_module.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True), strict=False)

    max_steps = cfg.trainer.max_steps
    if cfg.trainer.task_steps is not None:
        # Compute max steps in case of task arrays
        max_task_steps = step + cfg.trainer.task_steps
        max_steps = max_task_steps if max_steps == -1 else min(max_task_steps, cfg.trainer.max_steps)

    # step_tracker allows the current step to be shared with the data loader processes.
    data_module = DataModule(
        dataset_cfg=cfg.dataset, 
        data_loader_cfg=cfg.data_loader, 
        conditioning_cfg=cfg.denoising_model.conditioning,
        variable_mapper=variable_mapper,
        validation_benchmark_cfgs=cfg.validation_benchmarks, 
        test_benchmark_cfgs=cfg.test_benchmarks, 
        step_tracker=srm_lightling_module.step_tracker,
        output_dir=output_dir,
    )

    trainer = Trainer(
        accelerator="gpu",
        logger=logger,
        devices="auto",
        num_nodes=cfg.trainer.num_nodes,
        precision=cfg.trainer.precision,
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        callbacks=callbacks,
        limit_val_batches=None if cfg.trainer.validate else 0,
        val_check_interval=cfg.trainer.val_check_interval if cfg.trainer.validate else None,
        check_val_every_n_epoch=None,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        enable_checkpointing=cfg.checkpointing.save,
        enable_progress_bar=DEBUG or cfg.mode != "train",
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        gradient_clip_val=cfg.optimizer.gradient_clip_val,
        gradient_clip_algorithm=cfg.optimizer.gradient_clip_algorithm,
        max_epochs=cfg.trainer.max_epochs,
        max_steps=max_steps,
        profiler=AdvancedProfiler(dirpath=output_dir, filename="profile") if cfg.trainer.profile else None,
        detect_anomaly=cfg.trainer.detect_anomaly
    )

    if cfg.mode == "train":
        trainer.fit(srm_lightling_module, datamodule=data_module, ckpt_path=checkpoint_path if resume else None)
    elif cfg.mode == "val":
        trainer.validate(srm_lightling_module, datamodule=data_module, ckpt_path=checkpoint_path if lightning_checkpoint_available else None)
    elif cfg.mode == "test":
        metrics = trainer.test(srm_lightling_module, datamodule=data_module, ckpt_path=checkpoint_path if lightning_checkpoint_available else None)
        metrics = {k: v for d in metrics for k, v in d.items()}     # merge list of dicts
        if metrics:
            metric_fname = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]
            metrics_path = output_dir / "test" / f"{metric_fname}.json"
            metrics_path.parent.mkdir(exist_ok=True, parents=True)
            with metrics_path.open("w") as f:
                json.dump(metrics, f, indent=4, sort_keys=True)
    else:
        raise ValueError(f"Unknown mode {cfg.mode}")
