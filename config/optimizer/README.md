# Optimizer Configuration (`config/optimizer/`)

This directory contains YAML configuration files for optimizers used during model training. Each file (e.g., `adamw_default.yaml`) defines the parameters for a specific optimizer setup, based on `OptimizerCfg` from `src/training/srm_lightning_module.py`.

These optimizer configurations are selected in an experiment's `defaults` list (e.g., `optimizer: adamw_default`).

## `OptimizerCfg` Parameters

An optimizer configuration file specifies the following fields from `OptimizerCfg`:

*   **`name`**: (`str`) The name of the optimizer to use (e.g., "AdamW", "SGD"). This name is used by PyTorch Lightning or a custom factory to instantiate the optimizer.
*   **`lr`**: (`float`) The learning rate.
*   **`scale_lr`**: (`bool`, default: `False`) Whether to scale the learning rate by the global batch size. If true, the actual learning rate will be `lr * global_batch_size`.
*   **`kwargs`**: (`dict[str, Any] | None`, default: `None`) Additional keyword arguments to pass to the optimizer constructor (e.g., `weight_decay`, `betas` for AdamW).
    *   Example: `kwargs: {weight_decay: 0.01, betas: [0.9, 0.999]}`
*   **`scheduler`**: (`LRSchedulerCfg | None`, default: `None`) Configuration for a learning rate scheduler.
    *   If `null` or omitted, no scheduler is used.
    *   **`LRSchedulerCfg` fields** (from `src/training/srm_lightning_module.py`):
        *   **`name`**: (`str`) Name of the PyTorch LR scheduler (e.g., "ReduceLROnPlateau", "CosineAnnealingLR").
        *   **`interval`**: (`Literal["epoch", "step"]`, default: `"step"`) When to step the scheduler.
        *   **`frequency`**: (`int`, default: `1`) Frequency of scheduler steps.
        *   **`monitor`**: (`str | None`, default: `None`) Metric to monitor for schedulers like `ReduceLROnPlateau`.
        *   **`kwargs`**: (`dict[str, Any] | None`, default: `None`) Additional keyword arguments for the scheduler.
    *   Example for a scheduler:
        ```yaml
        scheduler:
          name: ReduceLROnPlateau
          interval: "epoch"
          monitor: "val/loss_total"
          kwargs: {factor: 0.1, patience: 10}
        ```
*   **`gradient_clip_val`**: (`float | int | None`, default: `None`) Value for gradient clipping. If `None`, no clipping is performed.
*   **`gradient_clip_algorithm`**: (`Literal["value", "norm"]`, default: `"norm"`) Algorithm for gradient clipping ("value" or "norm").

## Example Optimizer File

```yaml
# Example: config/optimizer/adamw_cosine_annealing.yaml
# This file would be selected in an experiment's defaults list as:
# defaults:
#   - optimizer: adamw_cosine_annealing

name: AdamW
lr: 0.0002
scale_lr: false
kwargs:
  weight_decay: 0.005
  betas: [0.9, 0.995]

scheduler:
  name: CosineAnnealingLR
  interval: "step"
  frequency: 1
  kwargs:
    T_max: 100000 # Corresponds to total training steps, for example
    eta_min: 0.000001

gradient_clip_val: 1.0
gradient_clip_algorithm: "norm"
```

**Important:**

*   The filename (e.g., `adamw_cosine_annealing`) is used in the experiment's `defaults` list to select this optimizer configuration.
*   The `name` field (e.g., `AdamW`, `CosineAnnealingLR`) inside the YAML is crucial as it's typically used by PyTorch Lightning or custom instantiation logic to select the correct optimizer/scheduler class from PyTorch's `torch.optim` and `torch.optim.lr_scheduler` modules.
*   Always refer to the `OptimizerCfg` and `LRSchedulerCfg` dataclass definitions in `src/training/srm_lightning_module.py` for the canonical list of parameters and their types/defaults. 