# Denoising Model Configuration (`config/denoising_model/`)

This directory stores YAML configuration files for Denoising Models. Each file (e.g., `my_unet_model.yaml`) defines the architecture and behavior of a complete generative model, including its constituent parts: the denoiser network, the noise/flow schedule, the tokenizer, and conditioning parameters.

These configurations correspond to `DenoisingModelCfg` (from `src/denoising_model/denoising_model.py`) and are selected in an experiment's `defaults` list (e.g., `denoising_model: my_unet_model`).

## `DenoisingModelCfg` Parameters

A denoising model configuration file specifies the following main fields:

*   **`denoiser`**: Configuration for the core network. This is a nested structure where you specify the type of denoiser (e.g., UNet, DiT) and its parameters.
    *   See [Denoiser Configuration Detail](#denoiser-configuration-detail) below.
*   **`flow`**: Configuration for the noise schedule or flow dynamics. This is a nested structure for a specific flow type.
    *   See [Flow Configuration Detail](#flow-configuration-detail) below.
*   **`tokenizer`**: Configuration for the data tokenizer. This is a nested structure for a specific tokenizer type.
    *   See [Tokenizer Configuration Detail](#tokenizer-configuration-detail) below.
*   **`conditioning`**: Configuration for how conditional information is used.
    *   Based on `ConditioningCfg` from `src/type_extensions.py`.
    *   Fields:
        *   `label` (bool, default: `False`): Whether to use label conditioning.
        *   `mask` (bool, default: `False`): Whether to use mask-based conditioning.
    *   Example: `conditioning: {label: true, mask: false}`
*   **`learn_sigma`**: (bool) Whether the model should learn the variance (sigma).
*   **`learn_v`**: (bool) Whether the model should predict the 'v' parameterization.
*   **`time_interval`**: (list[float], default: `[0, 1]`) The time interval for the process.
*   **`denoiser_parameterization`**: (`Parameterization`, default: `"ut"`) Parameterization expected by the denoiser (`"eps"`, `"ut"`, `"x0"`, `"v"`).
*   **`parameterization`**: (`Parameterization`, default: `"ut"`) Overall parameterization for the flow model (`"eps"`, `"ut"`, `"x0"`, `"v"`).
*   **`has_ema`**: (bool, default: `False`) Whether to use EMA for denoiser weights.

---

### Denoiser Configuration Detail (`denoiser` section)

This section defines the neural network. You will specify a `name` (or use a structure recognized by the registry) that maps to a specific denoiser type (e.g., "unet", "dit") and then provide its parameters.

**Common wrapped parameters (from base `DenoiserCfg`):**

*   **`class_embedding`**: Configuration for class label embedding.
    *   Based on `ClassEmbeddingCfg` and its subclasses like `ClassEmbeddingParametersCfg`.
    *   Typically specified by a `name` for the embedding type (e.g., "parameters") and its fields:
        *   `dropout_prob` (float, default: `0.0`)
        *   `init_std` (float, default: `0.02` for "parameters" type)
    *   Example:
        ```yaml
        class_embedding:
          name: parameters # Assuming 'parameters' is a registered ClassEmbedding type
          dropout_prob: 0.1
          init_std: 0.02
        ```
*   **`freeze`**: Configuration to freeze parts of the denoiser (`DenoiserFreezeCfg`).
    *   Fields: `time_embedding` (bool, default: `False`), `class_embedding` (bool, default: `False`).

**Specific Denoiser Example (`name: unet`):**
If you specify `name: unet` (assuming "unet" is registered to `UNetCfg`):
```yaml
# Inside the 'denoiser' section of your denoising_model config:
denoiser:
  name: unet # This name is used by the registry to pick UNetCfg and UNet class
  # UNetCfg specific fields:
  d_hidden: 128
  num_resnet_blocks: [2, 2, 2]
  # ... other UNet parameters
  # Inherited/wrapped DenoiserCfg fields:
  class_embedding:
    name: parameters
    dropout_prob: 0.1
  freeze: {time_embedding: false}
```
Refer to specific denoiser classes in `src/denoising_model/denoiser/` (e.g., `unet/model.py` for `UNetCfg`) for their unique parameters.

---

### Flow Configuration Detail (`flow` section)

Defines the noise schedule/dynamics. Specify a `name` for the flow type.

**Base `FlowCfg` fields (often implicitly part of specific flow type configs):**
*   **`variance`**: (`Literal["fixed_small", "fixed_large", "learned_range"]`, default: `"fixed_small"`)

**Specific Flow Example (`name: diffusion`):**
If `name: diffusion` is registered to `DiffusionCfg`:
```yaml
# Inside the 'flow' section:
flow:
  name: diffusion # Registry maps this to DiffusionCfg and Diffusion class
  # DiffusionCfg specific fields:
  beta_schedule:
    name: linear # Assuming 'linear' is a registered BetaSchedule type for LinearScheduleCfg
    beta_start: 0.0001
    beta_end: 0.02
    num_steps: 1000
  variance: "fixed_large"
```
Refer to `src/denoising_model/flow/` for flow types (e.g., `diffusion.py` for `DiffusionCfg`) and their parameters.

---

### Tokenizer Configuration Detail (`tokenizer` section)

Defines data-to-model-input transformation. Specify a `name` for the tokenizer type.
Base `TokenizerCfg` is empty; all parameters are in specific types.

**Specific Tokenizer Example (`name: unet_tokenizer`):**
If `name: unet_tokenizer` is registered to `UNetTokenizerCfg`:
```yaml
# Inside the 'tokenizer' section:
tokenizer:
  name: unet_tokenizer # Registry maps this to UNetTokenizerCfg and UNetTokenizer class
  # UNetTokenizerCfg specific fields might be added here (if any)
  # e.g., patch_size: 16
```
Refer to `src/denoising_model/tokenizer/` for tokenizer types (e.g., `unet_tokenizer/unet_tokenizer.py` for `UNetTokenizerCfg`).

---

## Example Denoising Model File Structure

```yaml
# Example: config/denoising_model/my_unet_diffusion_model.yaml
# This file would be selected in an experiment via 'denoising_model: my_unet_diffusion_model'

denoiser:
  name: unet
  d_hidden: 256
  # ... other UNet params
  class_embedding:
    name: parameters
    dropout_prob: 0.1
  freeze: {}

flow:
  name: diffusion
  beta_schedule:
    name: cosine # Assuming 'cosine' is registered for CosineScheduleCfg
    # ... CosineScheduleCfg params
  variance: "learned_range"

tokenizer:
  name: unet_tokenizer # Or whatever name your UNet-compatible tokenizer is registered with
  # ... any UNetTokenizer params

conditioning:
  label: true
  mask: false

learn_uncertainty: true
learn_variance: false
time_interval: [0.0, 1.0]
denoiser_parameterization: "eps"
parameterization: "eps"
has_ema: true
```

To determine the specific parameters available for each `name` (unet, diffusion, linear beta_schedule, etc.), consult the corresponding Python dataclass definition in the `src/denoising_model/` subdirectories. The project's registry system (e.g., `get_denoiser`, `get_flow`, `get_tokenizer`) uses these names to find and instantiate the correct components with their configurations. 