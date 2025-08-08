# Experiment Configuration (`config/experiment/`)

This directory contains YAML files that define complete experiments. Each file here typically corresponds to a `RootCfg` object (defined in `src/config.py`), which orchestrates all other configuration components like datasets, models, training parameters, etc.

An experiment configuration file is the main entry point you provide to `src/main.py` (e.g., `python -m src.main +experiment=my_experiment_name`).

## `RootCfg` Parameters

An experiment configuration file (e.g., `my_experiment.yaml`) is based on `RootCfg` and can include the following top-level sections and parameters:

*   **`mode`**: (`Literal["train", "val", "test"]`) Specifies the operational mode for the experiment.
*   **`seed`**: (`int | None`) Optional seed for reproducibility.

*   **`dataset`**: Configuration for the dataset. The filename used in the `defaults` list (e.g., `dataset: cifar10`) determines the specific dataset type via the project's registry.
    *   Contains parameters for the selected `DatasetCfg` subclass (e.g., `CIFAR10Cfg`).
    *   See `config/dataset/README.md` for details on base parameters and how specific dataset configs are structured.
    *   Example (assuming `dataset: cifar10` is in `defaults`):
        ```yaml
        # In config/experiment/my_experiment.yaml, if 'dataset: cifar10' is in defaults:
        # To override parameters for cifar10.yaml:
        dataset:
          subset_size: 10000
          data_shape: [3, 32, 32] # Default for cifar10, but can be overridden
        ```

*   **`variable_mapper`**: Configuration for mapping data variables. Selected by filename via `defaults`.
    *   See `config/variable_mapper/README.md` for details.

*   **`denoising_model`**: Configuration for the main denoising model. Selected by filename via `defaults`.
    *   This is a nested structure including `denoiser`, `flow`, `tokenizer`, and `conditioning` sections, where component types are often specified by a `name` field for registry lookup.
    *   See `config/denoising_model/README.md` for comprehensive details.

*   **`time_sampler`**: Configuration for how time steps are sampled. Selected by filename via `defaults`.
    *   Specific sampler types (e.g., `mean_beta`) are determined by the chosen file/name.
    *   See `config/time_sampler/README.md` for details.

*   **`loss`**: A dictionary of loss configurations. Each key is a an arbitrary loss name (e.g. `reconstruction_loss`), and the value is its configuration which must include a `name` field for the registry (e.g. `name: mse`).
    *   The `name` field (e.g., `mse`, `vlb`) is used by the registry to get the specific `LossCfg` subclass and `Loss` implementation.
    *   Base `LossCfg` fields (from `src/training/loss/loss.py`):
        *   `weight` (float | int, default: 1): Weight for this loss component.
        *   `apply_after_step` (int, default: 0): Step after which this loss is applied.
    *   Example (as seen in `config/main.yaml`):
        ```yaml
        loss:
          mu: # Arbitrary key for this loss instance
            name: mse # Registry key for MSECfg/MSELoss
            weight: 1.0
          vlb:
            name: vlb # Registry key for VLBCfg/VLBLoss
            weight: 0.01
            # apply_after_step: 0 # Default, can be added if needed
        ```
    *   Refer to `src/training/loss/` and `config/loss/README.md` (once created) for specific loss types.

*   **`optimizer`**: Configuration for the optimizer. Usually selected by filename via `defaults` (e.g., `optimizer: default`).
    *   The selected file (e.g. `config/optimizer/default.yaml`) will contain `OptimizerCfg` parameters from `src/training/srm_lightning_module.py`.
    *   See `config/optimizer/README.md` for details.

*   **`train`**: Training-specific parameters (`TrainCfg` from `src/training/srm_lightning_module.py`).
    *   Fields: `step_offset`, `log_losses_per_time_split`, `num_time_logging_splits`, `num_time_samples`, `ema_decay_rate`.

*   **`data_loader`**: Configuration for data loaders (`DataLoaderCfg` from `src/training/data_module.py`).
    *   Contains `train`, `val`, `test` sections, each with `batch_size`, `num_workers`, `persistent_workers`.

*   **`validation_benchmarks`**: (Optional) Dictionary of benchmark configurations for validation.
    *   Keys are arbitrary benchmark names. Values are `BenchmarkCfg`.
    *   `BenchmarkCfg` (from `src/benchmark/benchmark.py`) fields:
        *   `dataset`: A dataset configuration (filename specified here, e.g. `dataset: mnist_val_subset`). This will load `config/dataset/mnist_val_subset.yaml`.
        *   `evaluation`: An evaluation configuration. The type is specified by a `name` field for the registry (e.g., `name: fid_evaluation`).
    *   Example:
        ```yaml
        validation_benchmarks:
          mnist_fid_val:
            dataset: mnist_test # Loads config/dataset/mnist_test.yaml
            evaluation:
              name: image_sampling_fid # Name for registry to get FIDEvaluationCfg or similar
              # ... other FID evaluation params
              inference_sampler: default_sampler # Loads config/inference_sampler/default_sampler.yaml
        ```

*   **`test_benchmarks`**: (Optional) Dictionary of benchmark configurations for testing.
    *   Structure similar to `validation_benchmarks`.

*   **`checkpointing`**: (`CheckpointingCfg` from `src/config.py`).
*   **`trainer`**: (`TrainerCfg` from `src/config.py`).
*   **`torch`**: (`TorchCfg` from `src/config.py`).
*   **`wandb`**: (`WandbCfg` from `src/config.py`).
*   **`mnist_classifier`**: (str | None) Path to MNIST classifier. (This is a custom top-level param).

## Defaults and Overrides

Experiment configurations in Hydra heavily utilize a `defaults` list. This list is crucial for composing your final configuration by layering different specific configuration files from other subdirectories (like `config/dataset/`, `config/denoising_model/`, etc.) and then applying experiment-specific overrides.

**Key Concepts:**

1.  **Selecting Named Configs via `defaults` list:**
    *   Inside your experiment file (e.g., `my_experiment.yaml`), the `defaults` list refers to other YAML files by their name within their respective group (config path relative to `config/`). For instance, `dataset: cifar10` in the `defaults` list instructs Hydra to load `config/dataset/cifar10.yaml` and merge its contents under the `dataset` key in the final configuration.
    *   The `_self_` keyword is often included as the first item in `defaults` to ensure that values directly set in the current experiment file take precedence over those from included default files.

2.  **Direct Overrides in Experiment File:**
    *   Any parameter from `RootCfg` (or its nested Cfgs that are brought in via `defaults`) can be directly set or overridden in the main body of your experiment file. These values take precedence.

## Example Experiment Configuration Structure

This example reflects Hydra usage patterns consistent with `config/main.yaml`.

```yaml
# Example: config/experiment/my_training_run.yaml
# @package _global_  # Important: Makes this config part of the global namespace

defaults:
  - _self_ # Ensures values in this file override defaults
  # Select specific configuration files for each component:
  - dataset: cifar10_augmented  # Loads config/dataset/cifar10_augmented.yaml
  - denoising_model: unet_small_eps # Loads config/denoising_model/unet_small_eps.yaml
  - time_sampler: uniform_with_weighting # Loads config/time_sampler/uniform_with_weighting.yaml
  # Variable mapper could also be a default, e.g.:
  # - variable_mapper: image_default

# Direct overrides for RootCfg parameters or nested parameters
mode: train
seed: 1234

# Override parameters within a component loaded from defaults
dataset:
  subset_size: 50000 # Override from cifar10_augmented.yaml if needed

denising_model:
  learn_uncertainty: true # Override a field in the loaded unet_small_eps.yaml

variable_mapper: # Example of defining a component directly if not in defaults
  name: image_passthrough # Assuming 'image_passthrough' is a registered VariableMapper type
  # autoencoder: null # If this mapper type has an autoencoder option

loss:
  reconstruction:
    name: mse
    weight: 1.0
  vlb_component:
    name: vlb
    weight: 0.01
    apply_after_step: 10000

trainer:
  max_steps: 200000
  val_check_interval: 1000
  precision: "16-mixed"

data_loader:
  train:
    batch_size: 64
    num_workers: 8
  val:
    batch_size: 128
    num_workers: 4

checkpointing:
  load: null
  save: true
  every_n_train_steps: 5000

wandb:
  project: "Spatial-Reasoners-CIFAR10-Augmented"
  activated: true
  tags: ["unet_small", "augmented_data"]

validation_benchmarks:
  fid_cifar10_val:
    dataset: cifar10_val_std # Loads config/dataset/cifar10_val_std.yaml
    evaluation:
      name: image_sampling_fid # For registry to find FIDEvaluationCfg
      inference_sampler: default_50_steps # Loads config/inference_sampler/default_50_steps.yaml
      # ... other evaluation params
```

**Explanation of the Example:**

*   The `defaults` list is primary for selecting pre-defined YAML files for major components like `dataset`, `denoising_model`, etc. The filename (e.g., `cifar10_augmented`) acts as the key for your project's registry to load the correct type and its specific parameters from that file.
*   The `# @package _global_` directive is essential for this resolution mechanism to work correctly with such default lists.
*   Direct overrides in the experiment file (e.g., `mode: train`, `dataset.subset_size`) refine the configuration.
*   For dictionaries of components like `loss` or `validation_benchmarks.some_benchmark.evaluation`, a `name` field within that component's YAML structure is used by the registry to pick the specific class (e.g., `loss.reconstruction.name: mse` tells the registry to use the MSE loss).
*   This example avoids using Hydra's `_target_` keyword, assuming your registry system handles class instantiation based on these names or filenames.
