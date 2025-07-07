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
- `config_path` (str): Path to the configuration directory. If None, uses embedded configs
- `overrides` (List[str]): List of configuration overrides in Hydra format
- `enable_beartype` (bool): Enable runtime type checking for better error messages

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

### Evaluation Functions

#### `sr.evaluate_model()`

Evaluate a trained model on benchmarks.

```python
def evaluate_model(
    checkpoint_path: str,
    benchmark: str,
    config: DictConfig = None
) -> Dict[str, Any]
```

**Parameters:**

- `checkpoint_path` (str): Path to model checkpoint
- `benchmark` (str): Name of benchmark to evaluate on
- `config` (DictConfig): Optional configuration override

## Core Components

### Dataset Module

#### `register_dataset()`

Register a custom dataset class.

```python
def register_dataset(name: str, config_class: Type[DatasetCfg]):
    def decorator(dataset_class: Type[Dataset]):
        # Registration logic
        return dataset_class
    return decorator
```

**Example:**

```python
from spatialreasoners.dataset import register_dataset, DatasetCfg

@dataclass
class MyDatasetCfg(DatasetCfg):
    name: str = "my_dataset"
    data_path: str = "data/"

@register_dataset("my_dataset", MyDatasetCfg)
class MyDataset(sr.Dataset):
    def __init__(self, cfg: MyDatasetCfg):
        super().__init__()
        # Implementation
```

#### Base Dataset Class

```python
class Dataset(torch.utils.data.Dataset):
    """Base class for all datasets in Spatial Reasoners."""
    
    def __init__(self):
        super().__init__()
    
    def __len__(self) -> int:
        raise NotImplementedError
    
    def __getitem__(self, idx: int) -> Any:
        raise NotImplementedError
```

### Denoising Model Module

#### `register_denoiser()`

Register a custom denoiser model.

```python
def register_denoiser(name: str, config_class: Type[DenoiserCfg]):
    def decorator(denoiser_class: Type[Denoiser]):
        # Registration logic
        return denoiser_class
    return decorator
```

#### Base Denoiser Class

```python
class Denoiser(torch.nn.Module):
    """Base class for denoiser models."""
    
    def __init__(self, tokenizer, num_classes: int = None):
        super().__init__()
        self.tokenizer = tokenizer
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
```

### Variable Mapper Module

#### `register_variable_mapper()`

Register a custom variable mapper.

```python
def register_variable_mapper(name: str, config_class: Type[VariableMapperCfg]):
    def decorator(mapper_class: Type[VariableMapper]):
        # Registration logic
        return mapper_class
    return decorator
```

#### Base VariableMapper Class

```python
class VariableMapper:
    """Base class for variable mappers."""
    
    def __init__(self, cfg: VariableMapperCfg):
        self.cfg = cfg
    
    def map_to_model_space(self, data: Any) -> torch.Tensor:
        raise NotImplementedError
    
    def map_from_model_space(self, tensor: torch.Tensor) -> Any:
        raise NotImplementedError
```

### Tokenizer Module

#### `register_tokenizer()`

Register a custom tokenizer.

```python
def register_tokenizer(name: str, config_class: Type[TokenizerCfg]):
    def decorator(tokenizer_class: Type[Tokenizer]):
        # Registration logic
        return tokenizer_class
    return decorator
```

#### Base Tokenizer Class

```python
class Tokenizer:
    """Base class for tokenizers."""
    
    def __init__(self, cfg: TokenizerCfg):
        self.cfg = cfg
    
    def encode(self, data: Any) -> torch.Tensor:
        raise NotImplementedError
    
    def decode(self, tokens: torch.Tensor) -> Any:
        raise NotImplementedError
    
    @property
    def vocab_size(self) -> int:
        raise NotImplementedError
```

## Configuration Classes

### Core Configuration

#### `DenoisingModelCfg`

Configuration for denoising models.

```python
@dataclass
class DenoisingModelCfg:
    flow: str = "rectified"
    denoiser: DenoiserCfg = field(default_factory=DenoiserCfg)
    time_sampler: str = "uniform"
```

#### `TrainerCfg`

Configuration for PyTorch Lightning trainer.

```python
@dataclass
class TrainerCfg:
    max_epochs: int = 100
    max_steps: int = -1
    val_check_interval: int = 1000
    accelerator: str = "auto"
    devices: Union[int, str] = "auto"
    precision: str = "16-mixed"
```

#### `DataLoaderCfg`

Configuration for data loaders.

```python
@dataclass
class DataLoaderCfg:
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
```

### Component Configurations

All component configurations inherit from base classes:

- `DatasetCfg`: Base configuration for datasets
- `DenoiserCfg`: Base configuration for denoiser models  
- `VariableMapperCfg`: Base configuration for variable mappers
- `TokenizerCfg`: Base configuration for tokenizers

## Built-in Components

### Datasets

- `mnist_sudoku`: MNIST Sudoku puzzle dataset
- `cifar10`: CIFAR-10 image dataset
- `polygon_counting`: Polygon counting dataset

### Denoising Models

#### Flows
- `rectified`: Rectified flow formulation
- `cosine`: Cosine noise schedule
- `linear`: Linear noise schedule

#### Denoisers
- `unet`: U-Net architecture for images
- `dit_s_2`: Small DiT (Diffusion Transformer)
- `dit_b_2`: Base DiT
- `dit_l_2`: Large DiT
- `mar`: Masked Autoregressive model

### Variable Mappers

- `image`: Image variable mapping
- `continuous`: Continuous variable mapping
- `discrete`: Discrete variable mapping

## Error Handling

### Common Exceptions

#### `ComponentNotFoundError`

Raised when a requested component is not registered.

```python
class ComponentNotFoundError(Exception):
    """Raised when a component is not found in the registry."""
    pass
```

#### `ConfigurationError`

Raised when there's an issue with configuration.

```python
class ConfigurationError(Exception):
    """Raised when there's a configuration error."""
    pass
```

## Type Checking

Spatial Reasoners supports optional runtime type checking with beartype:

```python
# Enable type checking
sr.run_training(enable_beartype=True)

# Or via CLI
python training.py --enable-beartype
```

This provides enhanced error messages and catches type mismatches early.

## Utilities

### Logging

```python
import spatialreasoners.utils.logging as sr_logging

# Get logger
logger = sr_logging.get_logger(__name__)
```

### Checkpointing

```python
import spatialreasoners.utils.checkpointing as sr_checkpointing

# Save checkpoint
sr_checkpointing.save_checkpoint(model, path)

# Load checkpoint
model = sr_checkpointing.load_checkpoint(path)
```

## Examples

See the [Quick Tour](getting-started/quick-tour.md) and example projects for complete usage examples.

For more detailed examples and tutorials, visit the [GitHub repository](https://github.com/spatialreasoners/spatialreasoners). 