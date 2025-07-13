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

## Data Transformation Flow

<p align="center">
  <img src="../assets/overview.png" alt="Spatial Reasoners Overview"/>
</p>

Due to the datatype-agnostic nature of 🌀Spatial Reasoners the training and inference part of Spatial Reasoners don't operate directly on the data shapes and types provided in the datasets. Datasets in our framework always return an `UnstructuredExample`. This then is batched via a Dataloader into `BatchUnstructuredExample` -- which has the same structure, only the components contain the extra batch dimension. 

Next, a `VariableMapper` is used to map the `BatchUnstructuredExample` into the strcutred `BatchVariables` that store the data samples in a standard shape of `(batch_size, num_variables, num_features)` with some other values (see below). This format is used when calculating the flow, losses and during inference.

The `Denoisers` might require the input data to have some other format -- eg. a `DiT` denoiser requires the input data to be a set of tokens. Depending on the dataset, it might be the case that the Variables and Tokens might not be the same thing (eg. in Video generation where you want Variables to correspond to frames, but you want many more tokens to represent a frame). That's why we need a `Tokenizer` -- a class that the data in the Variables format, and transforms that to model inputs. After passing the data through the `Denoiser`, `Tokenizer` also defines how they should be mapped back to the common format. 

All of this allows you to only define a couple of transformation functions and you'll be ready to start training your own `DiT` (or other model) in any domain!

Soon you can expect the detailed Components documentation, but before it arrives, you can dive into our Spiral project and check the parent classes and ther pararmeters for reference.

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
  - experiment: null  # Users should specify experiment, or API will default to mnist_sudoku
  - time_sampler: mean_beta 
  - optimizer: default
  
validation_benchmarks: {} # override in experiment
test_benchmarks: {} # overrride in experiment

loss:
  mu:
    name: mse
    
train:
  step_offset: 0

hydra:
  run:
    dir: ""   # override!

wandb:
  project: sr #srm
  entity: sr #bartekpog-max-planck-institute-for-informatics
  mode: offline
  activated: false

data_loader:
  # Avoid having to spin up new processes to print out visualizations.
  train:
    num_workers: 16
    persistent_workers: true
    batch_size: 4096
  test:
    num_workers: 4
    persistent_workers: false
    batch_size: 4096
  val:
    num_workers: 16
    persistent_workers: true
    batch_size: 4096

seed: null

trainer:
  max_epochs: -1
  max_steps: 40001
  val_check_interval: 10000
  log_every_n_steps: 5000
  task_steps: null
  accumulate_grad_batches: 1
  precision: bf16-mixed
  num_nodes: 1
  validate: true
  profile: false
  detect_anomaly: false

torch:
  float32_matmul_precision: high  # (null --> default) highest / high / medium
  cudnn_benchmark: false

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
  learn_uncertainty: false
  flow: rectified  # From embedded configs
  denoiser:
    name: my_custom_model
  tokeniezer: 
    name: my_custom_tokenizer
```

## Creating Custom Components

Define custom components and auto-register them:

```python
# src/dataset.py
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

Don't forget to import all the components from your `src` -- for the components to be properly registered and accessible during inference, the scripts with implementations need to be imported (even if you don't reference them directly in your training script).
```python
from src import *
from . import dataset
from . import denoiser
from . import variable_mapper
from . import tokenizer
```

## Next Steps

- **Explore the [API Reference](../api.md)** for detailed documentation
- **Check the example project** for a complete implementation and reference for your projects

## Quick Comparison: Configuration Methods

| Method | Interface | CLI Support | Setup | Best For |
|--------|-----------|-------------|-------|----------|
| `@sr.config_main` | Decorator | ✅ Automatic | Minimal | General use, research, experimentation |
| Programmatic | Function | ❌ None | Minimal | Automation, notebooks, production |

**Recommendation:** Start with Method 1 (`@sr.config_main`) for most use cases. Use Method 2 for automation or when generating configurations dynamically. 