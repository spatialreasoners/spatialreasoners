# Variable Mapper Configuration (`config/variable_mapper/`)

This directory contains YAML configuration files for variable mappers. Variable mappers are crucial components responsible for transforming data from its original domain (e.g., raw images, text) into a structured representation suitable for the denoising model, and then mapping it back. This transformation can involve:

*   Normalization
*   Reshaping (e.g., splitting images into patches)
*   Tokenization (if not handled by the denoising model's dedicated tokenizer)
*   Passing data through an autoencoder to work within a compressed latent space.

Each YAML file in this directory defines the configuration for a specific type of variable mapper. The filename itself (e.g., `image_mapper.yaml`, `latent_space_mapper.yaml`) is often used in an experiment's `defaults` list to select that mapper. Alternatively, if defined directly within another configuration (like an experiment), a `name` field within the variable mapper's YAML structure will be used by the project's registry to instantiate the correct `VariableMapper` class and its corresponding configuration dataclass.

These configurations correspond to subclasses of `VariableMapperCfg` (from `src/variable_mapper/variable_mapper.py`), such as `ImageVariableMapperCfg` (from `src/variable_mapper/image_variable_mapper.py`).

## General Structure and Key Principles

1.  **Selection by Name/Filename**: An experiment configuration (`config/experiment/*.yaml`) will typically include a section for `variable_mapper`. This can either point to a file in this directory (e.g., `variable_mapper: image_default` which loads `config/variable_mapper/image_default.yaml`) or define the mapper inline with a `name` field.
    ```yaml
    # Option 1: In experiment defaults list
    # defaults:
    #   - variable_mapper: image_processing_v1

    # Option 2: Defined inline in an experiment (or other config)
    # variable_mapper:
    #   name: image_processing_v1 # This 'name' is key for the registry
    #   variable_patch_size: 8
    #   autoencoder: null 
    ```
    The project's registry (using `get_variable_mapper` from `src/variable_mapper/__init__.py`) uses this filename or `name` to find and instantiate the appropriate Python `VariableMapper` class (e.g., `ImageVariableMapper`) and its configuration dataclass (e.g., `ImageVariableMapperCfg`).

2.  **Parameters Map to Dataclass Fields**: The fields in the YAML configuration directly map to the attributes of the corresponding `...Cfg` dataclass for that mapper type.

## Example: Image Variable Mapper (`ImageVariableMapperCfg`)

Let's consider an example for an image variable mapper, which might be named `image_rgb_patches.yaml` in this directory, or referenced with `name: image` if its class `ImageVariableMapper` is registered with that name.

The `ImageVariableMapperCfg` (from `src/variable_mapper/image_variable_mapper.py`) has the following key parameters:

*   **`variable_patch_size`**: (`int`, default: `4`)
    *   Defines the size of the square patches the image will be divided into. For example, a value of `8` means the image is split into 8x8 patches. Each patch becomes a "variable" that the denoising model reasons over.
*   **`dependency_matrix_sigma`**: (`float`, default: `2.0`)
    *   Sigma for a Gaussian kernel used to calculate a dependency matrix between patches. This can inform the model about spatial relationships.
*   **`autoencoder`**: (`ImageAutoencoderCfg | None`, default: `None`)
    *   Configuration for an image autoencoder to map patches to/from a latent space. If `None`, the mapper works directly with raw patch data.
    *   If an autoencoder is used, this section will contain parameters for `ImageAutoencoderCfg` (from `src/variable_mapper/autoencoder/image_autoencoder.py`). The `ImageAutoencoderCfg` itself might then have a `name` field for the specific autoencoder architecture (e.g., "kl_vae", "sd_vae") and its parameters (e.g., `model_path`, `scale_factor`).

### Example `config/variable_mapper/image_rgb_patches.yaml`

```yaml
# config/variable_mapper/image_rgb_patches.yaml
# This mapper could be selected via 'variable_mapper: image_rgb_patches' in defaults,
# or if ImageVariableMapper is registered as "image", then by:
# variable_mapper:
#   name: image
#   variable_patch_size: 8 # Override default
#   ...

# Parameters for ImageVariableMapperCfg
variable_patch_size: 8
dependency_matrix_sigma: 2.5

autoencoder: null # Using raw pixel patches (8x8xC)
# OR, to use an autoencoder:
# autoencoder:
#   name: sd_vae_fp16 # Name for the autoencoder registry (e.g., get_autoencoder)
#   # Parameters for the specific autoencoder Cfg (e.g. ImageAutoencoderCfg)
#   model_path: "/path/to/sd_vae_fp16.ckpt"
#   scale_factor: 0.18215
#   latent_channels: 4 # Example, if the VAE outputs Bx4x(H/f)x(W/f)
```

**Explanation:**

*   If an experiment uses `variable_mapper: image_rgb_patches` in its `defaults` list, Hydra loads `config/variable_mapper/image_rgb_patches.yaml`.
*   The parameters like `variable_patch_size: 8` and `dependency_matrix_sigma: 2.5` directly set the corresponding fields in the `ImageVariableMapperCfg` object that will be instantiated.
*   If an `autoencoder` is specified, its `name` (e.g., `sd_vae_fp16`) would be used by your `get_autoencoder` registry function to instantiate the correct autoencoder model and its configuration. The fields within `autoencoder` (like `model_path`, `scale_factor`) would populate the autoencoder's specific Cfg.

**Important:**

*   Always refer to the specific Python dataclass definition for the `VariableMapper` you are configuring (e.g., `ImageVariableMapperCfg` in `src/variable_mapper/image_variable_mapper.py`) for a complete and authoritative list of available parameters and their default values.
*   Similarly, for nested configurations like `autoencoder`, refer to the relevant autoencoder configuration dataclass (e.g., `ImageAutoencoderCfg` and the Cfg for the specific autoencoder `name` chosen).
*   The project's registry system is key to linking the `name` or filename to the correct Python classes.

## `VariableMapperCfg` Base Parameters

The base `VariableMapperCfg` (from `src/variable_mapper/variable_mapper.py`) provides a fundamental parameter:

*   **`autoencoder`**: (`AutoEncoderCfg | None`, default: `None`) Configuration for an autoencoder to be used for mapping to/from a latent space. If `None`, no autoencoder is used, and the mapping might be an identity or simple normalization.
    *   If an autoencoder is used, this section will contain parameters for `AutoEncoderCfg` (from `src/variable_mapper/autoencoder/autoencoder.py`).
    *   The `AutoEncoderCfg` itself requires a `name` for the specific autoencoder type (e.g., "vae", "vqvae") which is used by the registry (`get_autoencoder` from `src/variable_mapper/autoencoder/__init__.py`) to instantiate the correct model, along with its specific parameters.
    *   Example of an `autoencoder` section:
        ```yaml
        autoencoder:
          name: my_custom_vae # Registry key for your VAE model/config
          # VAE-specific parameters:
          latent_dim: 256
          encoder_layers: [64, 128, 256]
          decoder_layers: [256, 128, 64]
          # pretrained_path: "/path/to/weights.pth" # Optional
        ```

## Mapper-Specific Configurations

Each YAML file in this directory defines the parameters for a particular variable mapper type. The filename (e.g., `image_identity.yaml`, `audio_mel_spectrogram.yaml`) is significant as it's used by the experiment configuration's `defaults` list and the project's registry (via `get_variable_mapper` from `src/variable_mapper/__init__.py`) to identify and instantiate the correct variable mapper class.

### Example: `config/variable_mapper/latent_autoencoder_mapper.yaml`

Assuming `latent_autoencoder_mapper.yaml` configures a `LatentAutoencoderVariableMapperCfg` (a hypothetical subclass of `VariableMapperCfg` that always requires an autoencoder):

```yaml
# This file configures a variable mapper that uses an autoencoder.
# Selected in experiment defaults: `variable_mapper: latent_autoencoder_mapper`

# Parameters for LatentAutoencoderVariableMapperCfg

autoencoder:
  name: standard_image_vae # This name is used by the registry to pick the VAE class & its config
  latent_dim: 128
  # ... other parameters for the 'standard_image_vae' autoencoder type
  # (e.g., path to pretrained weights, architecture details if not fixed by the name)
  # pretrained_checkpoint_path: "/path/to/your/vae_checkpoint.ckpt"

# Other parameters specific to LatentAutoencoderVariableMapperCfg, if any.
# For example, if it had a specific way to handle conditioning variables:
# conditioning_passthrough: true
```

### Example: `config/variable_mapper/image_passthrough_mapper.yaml`

Assuming `image_passthrough_mapper.yaml` configures an `ImageVariableMapperCfg` that might perform simple normalization but no autoencoding by default:

```yaml
# This file configures a simple image variable mapper (e.g., normalization only).
# Selected in experiment defaults: `variable_mapper: image_passthrough_mapper`

# Parameters for ImageVariableMapperCfg (from spatialreasoners.variable_mapper.image_variable_mapper)
autoencoder: null # Explicitly no autoencoder, or omit if 'null' is the default

# ImageVariableMapperCfg might have other fields, e.g.:
# normalization_mean: [0.5, 0.5, 0.5]
# normalization_std: [0.5, 0.5, 0.5]
```

**Important:**

*   When an experiment configuration includes `variable_mapper: my_mapper_name` in its `defaults` list, Hydra resolves this to `config/variable_mapper/my_mapper_name.yaml`.
*   The project's registry (using `get_variable_mapper`) then instantiates the Python class associated with `my_mapper_name` (e.g., `ImageVariableMapper`) and its configuration (e.g., `ImageVariableMapperCfg`).
*   If an `autoencoder` is specified within the mapper's config, its `name` field is used by the autoencoder registry (`get_autoencoder`) to instantiate the correct autoencoder model and its config.
*   Always refer to the specific Python dataclass definitions (e.g., `ImageVariableMapperCfg` in `src/variable_mapper/image_variable_mapper.py`, `AutoEncoderCfg` in `src/variable_mapper/autoencoder/autoencoder.py`, and specific autoencoder configs like `VAECfg`) for a complete list of available parameters. 