# Dataset Configuration (`config/dataset/`)

This directory contains configuration files for datasets used in experiments. These configurations are typically YAML files that specify parameters for dataset loading, preprocessing, and serving.

These files correspond to specific `DatasetCfg` subclasses (e.g., `MNISTCfg`, `CIFAR10Cfg`) found in `src/dataset/` and its subdirectories.

## `DatasetCfg` Base Parameters

The base `DatasetCfg` (from `src/dataset/dataset.py`) provides common parameters that specific dataset configurations might use or inherit:

*   **`subset_size`**: (Optional) An integer specifying the number of samples to use from the dataset. If `null` or not provided, the entire dataset will be used.
    *   Type: `int` | `None`
    *   Example: `subset_size: 1000`

*   **`data_shape`**: A sequence of integers defining the shape of a single data sample, excluding the batch dimension. For example, for a 28x28 grayscale image, this would be `[1, 28, 28]` (channels, height, width) or `[28, 28]` depending on convention.
    *   Type: `Sequence[int]` (e.g., a list in YAML)
    *   Example: `data_shape: [1, 64, 64]`

## Dataset-Specific Configurations

Each YAML file in this directory typically defines the parameters for a specific dataset type. The filename itself (e.g., `mnist.yaml`, `cifar10.yaml`) is significant as it's used by the experiment configuration's `defaults` list and the project's registry to identify and instantiate the correct dataset class.

### Example: `config/dataset/mnist.yaml`

```yaml
# This file configures an MNIST dataset.
# It would be selected in an experiment's defaults list like:
# defaults:
#   - dataset: mnist

# Parameters for an MNIST-specific Cfg (e.g., MNISTCfg from spatialreasoners.dataset.dataset_image.mnist)
subset_size: null
data_shape: [1, 28, 28]
data_root: "/path/to/your/mnist_data" # Example of a parameter specific to this dataset type
# train_split_ratio: 0.9 # Another example of a dataset-specific field
# val_split_ratio: 0.1
```

**Important:**
When an experiment configuration includes `dataset: my_dataset_name` in its `defaults` list, Hydra looks for `config/dataset/my_dataset_name.yaml`. The project's registry (utilizing functions like `get_dataset` from `src/dataset/__init__.py`) then uses this identifier (`my_dataset_name`) to instantiate the corresponding Python `Dataset` class and its configuration (e.g., `MyDatasetNameCfg`).

Always refer to the specific Python dataclass definition for the dataset you are configuring (e.g., `src/dataset/dataset_image/mnist.py` for `MNISTCfg`) for a complete list of available parameters beyond the base `DatasetCfg` fields. 