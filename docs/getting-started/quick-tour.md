# Quick Tour

This quick tour will get you up and running with Spatial Reasoners in just a few minutes.

## Basic Usage with Predefined Experiments

The fastest way to get started is using predefined experiments:

```python
import spatialreasoners as sr

# 🚀 One-line training with sensible defaults
sr.run_training()

# 🔍 With enhanced type checking for better error messages
sr.run_training(enable_beartype=True)

# ⚙️ Customize training parameters
sr.run_training(overrides=[
    "experiment=mnist_sudoku",    # Use specific experiment
    "trainer.max_epochs=50",      # Train for 50 epochs
    "data_loader.train.batch_size=32"  # Adjust batch size
])

# 🔧 Advanced usage with different model architectures
sr.run_training(overrides=[
    "denoising_model.denoiser=dit_l_2",  # Use large DiT model
    "denoising_model.flow=cosine",       # Use cosine flow
    "variable_mapper=image"              # Image variable mapping
])
```

## Custom Projects

Spatial Reasoners provides two clean approaches for creating custom research projects.

### Method 1: @sr.config_main Decorator (Recommended)

The cleanest interface for most use cases - similar to `@hydra.main` but with automatic config merging.

Create your training script (`training.py`):

```python
#!/usr/bin/env python3
import spatialreasoners as sr

# Import your custom components to auto-register them
import src  # This imports and registers all your custom components

@sr.config_main(config_path="configs", config_name="main")
def main(cfg):
    """Main training function with full control over the process."""
    
    # Create components from the loaded config
    lightning_module = sr.create_lightning_module(cfg)
    data_module = sr.create_data_module(cfg)
    trainer = sr.create_trainer(cfg)
    
    # Full control - add custom callbacks, modify trainer, etc.
    trainer.fit(lightning_module, datamodule=data_module)

if __name__ == "__main__":
    main()
```

**CLI Usage:**

```bash
# Basic training with your experiment
python training.py experiment=my_experiment

# Customize any parameter via CLI
python training.py experiment=my_experiment trainer.max_epochs=100

# Multiple overrides
python training.py experiment=my_experiment trainer.max_epochs=50 dataset.subset_size=15000

# Enhanced type checking
python training.py experiment=my_experiment --enable-beartype

# Get help and examples
python training.py --help
```

### Method 2: Programmatic Configuration

For automation, notebooks, or when you need to generate configurations dynamically:

```python
#!/usr/bin/env python3
import spatialreasoners as sr

# Import your custom components to auto-register them
import src

def main():
    """Programmatic training configuration."""
    
    # Define overrides as needed
    overrides = [
        "experiment=my_experiment", 
        "trainer.max_epochs=100",
        "dataset.subset_size=20000"
    ]
    
    # Run training with the overrides
    sr.run_training(
        config_name="main",
        config_path="configs",
        overrides=overrides,
        enable_beartype=True,
    )

if __name__ == "__main__":
    main()
```

## Example Project: Spiral Dataset

The included example project demonstrates a complete working implementation:

### Project Structure

```
your_project/
├── training.py              # Your main training script
├── src/                     # Custom components
│   ├── __init__.py         # Auto-register components
│   ├── dataset.py          # Custom datasets
│   ├── denoiser.py         # Custom models  
│   ├── variable_mapper.py  # Custom variable mappers
│   └── tokenizer.py        # Custom tokenizers
└── configs/                # Configuration files
    ├── main.yaml           # Main config (references experiments)
    ├── experiment/         # Experiment-specific configs
    │   └── my_experiment.yaml
    ├── dataset/            # Custom dataset configs
    └── variable_mapper/    # Custom mapper configs
```

### Running the Example

```bash
cd example_project

# Method 1: @sr.config_main decorator (recommended)
python training_decorator.py experiment=spiral_training

# Method 2: Programmatic configuration  
python training_programmatic.py
```

The spiral example shows a model learning to generate points along a spiral pattern, demonstrating:

- Custom dataset, variable mapper, tokenizer, and denoiser implementations
- Clean configuration management with experiment-specific configs
- Visualization and evaluation during training

## Configuration System

Spatial Reasoners uses Hydra for flexible configuration management:

### Basic Configuration

```yaml
# configs/main.yaml
defaults:
  - experiment: null  # Users specify via CLI
  - time_sampler: mean_beta
  - optimizer: default
  - _self_

# Your project-specific defaults
trainer:
  max_steps: 3000
  val_check_interval: 1000
  
data_loader:
  train:
    batch_size: 128
    num_workers: 16
```

### Experiment Configuration

```yaml
# configs/experiment/my_experiment.yaml
# @package _global_
defaults:
  - /dataset: my_custom_dataset  # Your custom dataset

# Mix local and embedded components
variable_mapper:
  name: my_custom_mapper

denoising_model:
  flow: rectified  # From embedded configs
  denoiser:
    name: my_custom_model
```

## Creating Custom Components

Define custom components and auto-register them:

```python
# src/dataset.py
import spatialreasoners as sr
from spatialreasoners.dataset import register_dataset, DatasetCfg
from dataclasses import dataclass

@dataclass 
class MyDatasetCfg(DatasetCfg):
    name: str = "my_dataset"
    data_path: str = "data/"
    subset_size: int = 10000

@register_dataset("my_dataset", MyDatasetCfg)
class MyDataset(sr.Dataset):
    def __init__(self, cfg: MyDatasetCfg):
        # Your dataset implementation
        pass
```

```python
# src/__init__.py - Auto-register all components
from . import dataset
from . import denoiser
from . import variable_mapper
from . import tokenizer
```

## Running Benchmarks

Evaluate models on standard benchmarks:

```python
import spatialreasoners as sr

# Load a pretrained model and run evaluation
config = sr.load_default_config()
results = sr.evaluate_model(
    checkpoint_path="./checkpoints/mnist_sudoku.ckpt",
    benchmark="mnist_sudoku_hard"
)
```

## Advanced Configuration Loading

```python
# Multiple ways to load and customize configs
config = sr.load_default_config()  # Built-in api_default experiment
config = sr.load_config_from_yaml(overrides=["experiment=mnist_sudoku"])
config = sr.load_config_from_yaml("./configs", "main", ["experiment=custom"])

# Programmatic config modification
config.trainer.max_epochs = 100
config.data_loader.train.batch_size = 32
```

## Next Steps

- **Explore the [API Reference](../api.md)** for detailed documentation
- **Check the example project** for a complete implementation
- **Read about [custom components](../advanced/custom-components.md)** to extend the framework
- **Join the [GitHub Discussions](https://github.com/spatialreasoners/spatialreasoners/discussions)** for community support

## Quick Comparison: Configuration Methods

| Method | Interface | CLI Support | Setup | Best For |
|--------|-----------|-------------|-------|----------|
| `@sr.config_main` | Decorator | ✅ Automatic | Minimal | General use, research, experimentation |
| Programmatic | Function | ❌ None | Minimal | Automation, notebooks, production |

**Recommendation:** Start with Method 1 (`@sr.config_main`) for most use cases. Use Method 2 for automation or when generating configurations dynamically. 