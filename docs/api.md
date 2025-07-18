# API Reference

This page provides detailed API documentation for Spatial Reasoners.

## High-Level API

### Training Functions

#### `sr.run_training()`

Main function for running training with predefined configurations.

```python
def run_training(
    config_name: str = "api_default",
    config_path: str = None, 
    overrides: List[str] = None,
    enable_beartype: bool = False,
) -> Any
```

**Parameters:**

- `config_name` (str): Name of the configuration file to use. Default: "api_default"
- `config_path` (str): Path to the custom configuration directory. If None, uses embedded configs only
- `overrides` (List[str]): List of configuration overrides in Hydra format
- `enable_beartype` (bool): Enable runtime type checking for better debugging

**Example:**

```python
import spatialreasoners as sr

# Basic usage
sr.run_training()

# With overrides
sr.run_training(overrides=[
    "experiment=mnist_sudoku",
    "trainer.max_epochs=50"
])
```

#### `@sr.config_main`

Decorator for creating training scripts with automatic configuration management.

```python
def config_main(
    config_path: str = "configs",
    config_name: str = "main"
)
```

**Parameters:**

- `config_path` (str): Path to configuration directory
- `config_name` (str): Name of the main configuration file

**Example:**

```python
@sr.config_main(config_path="configs", config_name="main")
def main(cfg):
    lightning_module = sr.create_lightning_module(cfg)
    data_module = sr.create_data_module(cfg)
    trainer = sr.create_trainer(cfg)
    trainer.fit(lightning_module, datamodule=data_module)
```

### Configuration Functions

#### `sr.load_default_config()`

Load the default embedded configuration.

```python
def load_default_config() -> DictConfig
```

**Returns:**
- `DictConfig`: Default configuration object

#### `sr.load_config_from_yaml()`

Load configuration from YAML files with optional overrides.

```python
def load_config_from_yaml(
    config_path: str = None,
    config_name: str = "api_default", 
    overrides: List[str] = None
) -> DictConfig
```

**Parameters:**

- `config_path` (str): Path to configuration directory
- `config_name` (str): Name of configuration file
- `overrides` (List[str]): Configuration overrides

### Component Factory Functions

#### `sr.create_lightning_module()`

Create a PyTorch Lightning module from configuration.

```python
def create_lightning_module(cfg: DictConfig) -> LightningModule
```

#### `sr.create_data_module()`

Create a PyTorch Lightning data module from configuration.

```python
def create_data_module(cfg: DictConfig) -> LightningDataModule
```

#### `sr.create_trainer()`

Create a PyTorch Lightning trainer from configuration.

```python
def create_trainer(cfg: DictConfig) -> Trainer
```

## Registering components

**Example:**

```python
from spatialreasoners.dataset import register_dataset, DatasetCfg

from spatialreasoners.type_extensions import UnstructuredExample

@dataclass
class MyDatasetCfg(DatasetCfg):
    data_path: str = "data/"
    # Other parameters

@register_dataset("my_dataset", MyDatasetCfg)
class MyDataset(sr.Dataset):
    def __init__(self, cfg: MyDatasetCfg):
        super().__init__()
        
    @property
    def _num_available(self) -> int
        pass # get full size of the full dataset

    def __getitem__(self, idx: int) -> UnstructuredExample:
        pass # prepare Unstructured Example
```

## Type Checking

Spatial Reasoners supports optional runtime type checking with beartype:

```python
# Enable type checking
sr.run_training(enable_beartype=True)

# Or via CLI
python training.py --enable-beartype
```

This provides enhanced error messages and catches type mismatches early. All the builtin methods that operate on tensors use `jaxtyping`, with together with `beartype` allows us to check whether there are any `shape` mismatches. We also encourage you to type your methods with `jaxtyping`, as it can really speed up debugging!

## Examples

See the [Quick Tour](getting-started/quick-tour.md) and [example project](https://github.com/spatialreasoners/spatialreasoners/tree/main/example_project) for complete usage examples.

For a minimalistic project template, check out our `example_project` in the the [GitHub repository](https://github.com/spatialreasoners/spatialreasoners). 