# Time Sampler Configuration (`config/time_sampler/`)

This directory contains YAML configuration files for different time samplers used during the training of denoising models. Time samplers determine how time steps (often denoted as `t`) are sampled from the overall time interval (e.g., [0, 1]) for each training iteration.

Each file in this directory (e.g., `uniform_sampler.yaml`, `loss_aware_sampler.yaml`) corresponds to a specific `TimeSamplerCfg` subclass found in `src/training/time_sampler/time_sampler.py` and its subdirectories (like `src/training/time_sampler/weight_time_sampler.py`).

These configurations are selected in an experiment's `defaults` list (e.g., `time_sampler: my_preferred_sampler`).

## `TimeSamplerCfg` Base Parameters

The base `TimeSamplerCfg` (from `src/training/time_sampler/time_sampler.py`) provides fundamental parameters:

*   **`num_steps`**: (`int | None`, default: `None`) The number of discrete steps to consider for schedules or specific samplers. If `None`, it might imply a continuous sampler or that the number of steps is derived from elsewhere (e.g., the flow model).

## Sampler-Specific Configurations

Each YAML file defines the parameters for a particular time sampler type. The filename (e.g., `uniform.yaml`, `mean_beta_weighted.yaml`) is significant as it's used by the experiment configuration's `defaults` list and the project's registry (via `get_time_sampler` from `src/training/time_sampler/__init__.py`) to identify and instantiate the correct time sampler class.

### Common Sub-Configurations

Many time samplers, especially those in `src/training/time_sampler/weight_time_sampler.py` (like `MeanBetaWeightedTimeSamplerCfg`, `LossAwareTimeSamplerCfg`), might use nested configurations for weighting schemes or PDF estimation:

*   **`time_weighting`**: Defines how weights are applied to different time steps. Based on `TimeWeightingCfg` and its subclasses (e.g., `ConstantTimeWeightingCfg`, `SnrTimeWeightingCfg`) from `src/training/time_sampler/time_weighting.py`.
    *   You typically specify a `name` for the weighting type (e.g., `constant`, `snr`) and its specific parameters.
    *   Example:
        ```yaml
        time_weighting:
          name: snr # Registry key for SnrTimeWeightingCfg
          # ... other SnrTimeWeightingCfg parameters like snr_power
        ```
*   **`histogram_pdf_estimator`**: Used by samplers that adapt based on a probability distribution (e.g., `LossAwareTimeSamplerCfg`). Based on `HistogramPdfEstimatorCfg` from `src/training/time_sampler/histogram_pdf_estimator.py`.
    *   Fields: `num_bins`, `min_count`, `max_history`, `update_every_n_steps`, `smoothing_std`.

### Example: `config/time_sampler/mean_beta_weighted.yaml`

Assuming `mean_beta_weighted.yaml` configures a `MeanBetaWeightedTimeSamplerCfg`:

```yaml
# This file might configure a MeanBetaWeightedTimeSampler.
# Selected in experiment defaults: `time_sampler: mean_beta_weighted`

# Parameters for MeanBetaWeightedTimeSamplerCfg (from spatialreasoners.training.time_sampler.weight_time_sampler)
num_steps: 1000 # From base TimeSamplerCfg, might be relevant here

time_weighting:
  name: snr # Example: use SNR-based weighting
  snr_power: 0.5

# Other MeanBetaWeightedTimeSamplerCfg specific fields (if any)
# For example, if it had a beta_schedule_name parameter:
# beta_schedule_name: "linear"
```

**Important:**

*   When an experiment configuration includes `time_sampler: my_sampler_name` in its `defaults` list, Hydra resolves this to `config/time_sampler/my_sampler_name.yaml`.
*   The project's registry (using `get_time_sampler`) then instantiates the Python class associated with `my_sampler_name` (e.g., `MeanBetaWeightedTimeSampler`) and its configuration (e.g., `MeanBetaWeightedTimeSamplerCfg`).
*   Nested components like `time_weighting` also use a `name` field for registry lookup (e.g., `get_time_weighting`).
*   Always refer to the specific Python dataclass definitions (e.g., `MeanBetaWeightedTimeSamplerCfg` in `src/training/time_sampler/weight_time_sampler.py`, or `SnrTimeWeightingCfg` in `src/training/time_sampler/time_weighting.py`) for a complete list of available parameters. 