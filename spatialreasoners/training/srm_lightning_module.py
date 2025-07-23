import inspect
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, get_args

import torch
from jaxtyping import Float
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import Tensor, optim

if TYPE_CHECKING:
    # NOTE to avoid circular import wrapper -> datamodule -> evaluation -> wrapper
    from training.data_module import DataModule

from spatialreasoners.benchmark.evaluation.type_extensions import MetricsBatch
from spatialreasoners.denoising_model import DenoisingModel, DenoisingModelCfg
from spatialreasoners.env import DEBUG
from spatialreasoners.misc.local_logger import LocalLogger
from spatialreasoners.misc.step_tracker import StepTracker
from spatialreasoners.misc.time_tester import TimeTester
from spatialreasoners.type_extensions import BatchUnstructuredExample, FullPrecision
from spatialreasoners.variable_mapper import VariableMapper

from .loss import Loss, LossCfg, get_loss
from .time_sampler import TimeSampler, TimeSamplerCfg, get_time_sampler


@dataclass(frozen=True, kw_only=True)
class LRSchedulerCfg:
    name: str
    interval: Literal["epoch", "step"] = "step"
    frequency: int = 1
    monitor: str | None = None
    kwargs: dict[str, Any] | None = None


@dataclass(frozen=True, kw_only=True)
class OptimizerCfg:
    name: str
    lr: float
    scale_lr: bool = False
    kwargs: dict[str, Any] | None = None
    scheduler: LRSchedulerCfg | None = None
    gradient_clip_val: float | int | None = None
    gradient_clip_algorithm: Literal["value", "norm"] = "norm"


@dataclass(frozen=True, kw_only=True)
class TrainCfg:
    step_offset: int
    log_losses_per_time_split: bool = True
    num_time_logging_splits: int = 10
    num_time_samples: int = 1
    ema_decay_rate: float = 0.9999


@dataclass(frozen=True, kw_only=True)
class SRMLightningModuleCfg:
    denoising_model: DenoisingModelCfg
    time_sampler: TimeSamplerCfg
    loss: dict[str, LossCfg]
    optimizer: OptimizerCfg
    train: TrainCfg


class SRMLightningModule(LightningModule):
    cfg: SRMLightningModuleCfg
    logger: LocalLogger | WandbLogger | None
    time_sampler: TimeSampler
    losses: dict[str, Loss]
    step_tracker: StepTracker
    variable_mapper: VariableMapper
    model: DenoisingModel

    def __init__(
        self,
        cfg: SRMLightningModuleCfg,
        variable_mapper: VariableMapper,
        num_classes: int | None = None,
    ) -> None:
        if cfg.denoising_model.conditioning.label:
            assert num_classes is not None
        super().__init__()
        self.cfg = cfg
        self.step_tracker = StepTracker(cfg.train.step_offset)
        self.variable_mapper = variable_mapper
        self.time_sampler = get_time_sampler(cfg.time_sampler, self.variable_mapper)

        self.denoising_model = DenoisingModel(
            cfg=cfg.denoising_model,
            variable_mapper=self.variable_mapper,
            num_classes=num_classes,
            step_tracker=self.step_tracker,
        )

        self.losses = {
            k: get_loss(v, self.denoising_model.flow) for k, v in self.cfg.loss.items()
        }

        # This is used for testing.
        self.time_tester = TimeTester()

    def log_time_split_loss(
        self,
        key: str,
        loss: Float[Tensor, "*batch num_variables feature"],
        t: Float[Tensor, "*batch num_variables"],
    ) -> None:
        if self.cfg.train.num_time_logging_splits > 1:
            # Log mean loss for every equal-size time interval
            interval_size = 1 / self.cfg.train.num_time_logging_splits
            loss_log = loss.detach().mean(dim=-3).flatten()
            t_split_idx = torch.floor_divide(t.flatten(), interval_size).long()
            t_split_loss = torch.full(
                (self.cfg.train.num_time_logging_splits,),
                fill_value=torch.nan,
                dtype=loss_log.dtype,
                device=loss_log.device,
            )
            t_split_loss.scatter_reduce_(
                0, t_split_idx, loss_log, reduce="mean", include_self=False
            )
            for i in range(self.cfg.train.num_time_logging_splits):
                if not torch.isnan(t_split_loss[i]):
                    start = i * interval_size
                    self.log(
                        f"loss/{key}_{start:.1f}-{start+interval_size:.1f}",
                        t_split_loss[i],
                    )

    def training_step(
        self, batch: BatchUnstructuredExample, batch_idx
    ) -> Float[Tensor, ""]:
        if self.cfg.denoising_model.conditioning.mask:
            assert "mask" in batch

        # Tell the data loader processes about the current step.
        self.step_tracker.set_step(self.global_step)
        step = self.step_tracker.get_step()
        self.log(f"step_tracker/step", step)

        batch_variables = self.variable_mapper.unstructured_to_variables(batch)
        device = batch_variables.device

        clean_x = batch_variables.z_t
        clean_x_repeat = clean_x.unsqueeze(  # Original data from dataset
            1
        ).repeat(  # Add time dimension
            1, self.cfg.train.num_time_samples, 1, 1
        )  # Repeat for each time sample

        # TODO TaskSampler
        t, loss_weight, should_denoise_mask = self.time_sampler(
            batch_variables.batch_size, self.cfg.train.num_time_samples, device
        )

        if self.cfg.denoising_model.conditioning.mask:
            # Image level time with standard conditioning on masked (and mask)
            assert (
                batch_variables.mask is not None
            ), "mask must be provided if conditioning on mask"
            x_masked = clean_x * (1 - batch_variables.mask.unsqueeze(-1))
            mask = batch_variables.mask.unsqueeze(
                1
            )  # for broadcasting along time dimension

            loss_weight = loss_weight * mask
            if should_denoise_mask is not None:
                should_denoise_mask.logical_and_(mask > 0)

            t.mul_(mask)
        else:
            mask = None
            x_masked = None

        if self.cfg.denoising_model.conditioning.label:
            assert (
                batch_variables.label is not None
            ), "label must be provided if conditioning on label"
            label = batch_variables.label
        else:
            label = None

        eps = self.denoising_model.flow.sample_eps(clean_x_repeat)
        t_repeat = t.unsqueeze(-1).repeat(*[1] * (t.ndim), clean_x_repeat.shape[-1])
        z_t = self.denoising_model.flow.get_zt(t_repeat, eps=eps, x=clean_x_repeat)

        model_outputs = self.denoising_model(
            z_t=z_t,
            t=t,
            should_denoise=should_denoise_mask,
            mask=batch_variables.mask,
            x_masked=x_masked,
            label=label,
            label_mask=None,  # Only used for classifier-free guidance
            use_ema=False,
            sample=False,
            fixed_conditioning_fields=batch_variables.fixed_conditioning_fields,
        )

        if self.cfg.denoising_model.parameterization == "eps":
            target = eps
        elif self.cfg.denoising_model.parameterization == "ut":
            target = self.denoising_model.flow.get_ut(
                t_repeat, eps=eps, x=clean_x_repeat
            )
        elif self.cfg.denoising_model.parameterization == "v":
            target = self.denoising_model.flow.get_v(
                t=t_repeat, eps=eps, x0=clean_x_repeat
            )
        elif self.cfg.denoising_model.parameterization == "x0":
            target = clean_x_repeat
        else:
            raise ValueError(
                f"Unknown parameterization {self.cfg.denoising_model.parameterization}"
            )

        sigma_pred_repeat = None
        if self.cfg.denoising_model.learn_uncertainty:
            sigma_pred = model_outputs.sigma_theta
            sigma_pred_repeat = sigma_pred.unsqueeze(-1).repeat(
                *[1] * (sigma_pred.ndim), clean_x_repeat.shape[-1]
            )

        total_loss = 0
        for loss_key, loss_func in self.losses.items():
            unweighted_loss = loss_func(
                t=t_repeat,
                eps=eps,
                x=clean_x_repeat,
                z_t=z_t,
                mean_target=target,
                mean_pred=model_outputs.mean_theta,
                v_pred=model_outputs.variance_theta,
                sigma_pred=sigma_pred_repeat,
                global_step=step,
            )

            weighted_loss = unweighted_loss * loss_weight.unsqueeze(-1)
            if self.cfg.train.log_losses_per_time_split:
                self.log_time_split_loss(f"unweighted/{loss_key}", unweighted_loss, t)
                self.log_time_split_loss(f"weighted/{loss_key}", weighted_loss, t)

            self.log(f"loss/unweighted/{loss_key}", unweighted_loss.detach().mean())
            loss_value = weighted_loss.mean()
            self.log(f"loss/weighted/{loss_key}", loss_value)
            total_loss = total_loss + loss_value

        self.log(f"loss/total", total_loss)

        if (
            self.global_rank == 0
            and step % self.trainer.log_every_n_steps == 0
            and (self.trainer.fit_loop.total_batch_idx + 1)
            % self.trainer.accumulate_grad_batches
            == 0
        ):
            # Print progress
            print(f"train step = {step}; loss = {total_loss:.6f};")

        return total_loss


    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        self.denoising_model.update_ema_denoiser()
        return super().on_train_batch_end(outputs, batch, batch_idx)

    def test_step(
        self, batch: BatchUnstructuredExample, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        datamodule: "DataModule" = self.trainer.datamodule
        benchmark = datamodule.test_benchmarks[dataloader_idx]
        assert benchmark is not None, "Test benchmark is not set up"
        metrics_iter = benchmark.evaluation(
            self.denoising_model,
            batch,
            batch_idx,
            logger=self.logger,
            global_rank=self.global_rank,
        )

        if metrics_iter is not None:
            for metrics_batch in metrics_iter:
                self.log_metrics(metrics_batch)

    def validation_step(
        self,
        batch: BatchUnstructuredExample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        datamodule: "DataModule" = self.trainer.datamodule
        benchmark = datamodule.validation_benchmarks[dataloader_idx]
        assert benchmark is not None, "Validation benchmark is not set up"
        metrics_iter = benchmark.evaluation(
            self.denoising_model,
            batch,
            batch_idx,
            logger=self.logger,
            global_rank=self.global_rank,
        )

        if metrics_iter is not None:
            for metrics_batch in metrics_iter:
                self.log_metrics(metrics_batch)

    def log_metrics(self, metrics_batch: MetricsBatch) -> None:
        self.log_dict(
            {
                f"{metrics_batch.evaluation_key}/{metrics_batch.sampler_key}/{metric_key}": metric_value
                for metric_key, metric_value in metrics_batch.metrics.items()
            },
            add_dataloader_idx=False,
            batch_size=metrics_batch.batch_size,
            sync_dist=True,
        )

    @property
    def effective_batch_size(self) -> int:
        datamodule: "DataModule" = self.trainer.datamodule
        # assumes one fixed batch_size for all train dataloaders!
        return (
            self.trainer.accumulate_grad_batches
            * self.trainer.num_devices
            * self.trainer.num_nodes
            * datamodule.data_loader_cfg.train.batch_size
        )

    @property
    def num_training_steps_per_epoch(self) -> int:
        self.trainer.fit_loop.setup_data()
        dataset_size = len(self.trainer.train_dataloader)
        num_steps = dataset_size // self.trainer.accumulate_grad_batches
        return num_steps

    @property
    def num_training_steps(self) -> int:
        if self.trainer.max_steps > -1:
            return self.trainer.max_steps
        return self.trainer.max_epochs * self.num_training_steps_per_epoch

    def configure_optimizers(self):
        cfg = self.cfg.optimizer
        kwargs = {} if cfg.kwargs is None else cfg.kwargs
        opt_class = getattr(optim, cfg.name)
        opt_signature = inspect.signature(opt_class)
        if "eps" in opt_signature.parameters.keys():
            # Too small epsilon in optimizer can cause training instabilities with half precision
            kwargs |= dict(
                eps=(
                    1.0e-8
                    if self.trainer.precision in get_args(FullPrecision)
                    else 1.0e-7
                )
            )
        if "weight_decay" in opt_signature.parameters.keys():
            wd, no_wd = self.denoising_model.get_weight_decay_parameter_groups()
            params = [{"params": wd}, {"params": no_wd, "weight_decay": 0.0}]
        else:
            params = [p for p in self.denoiser.parameters() if p.requires_grad]
        lr = self.cfg.optimizer.lr
        if self.cfg.optimizer.scale_lr:
            lr *= self.effective_batch_size
        opt = opt_class(params, lr=lr, **kwargs)
        res = {"optimizer": opt}
        # Generator scheduler
        if self.cfg.optimizer.scheduler is not None:
            cfg = self.cfg.optimizer.scheduler
            res["lr_scheduler"] = {
                "scheduler": getattr(optim.lr_scheduler, cfg.name)(
                    opt, **(cfg.kwargs if cfg.kwargs is not None else {})
                ),
                "interval": cfg.interval,
                "frequency": cfg.frequency,
            }
            if cfg.monitor is not None:
                res["lr_scheduler"]["monitor"] = cfg.monitor
        return res
