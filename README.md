# 🌀Spatial Reasoners 

**A Python package for spatial reasoning over continuous variables with generative denoising models.**

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.11+-brightgreen.svg)]()
[![PyPI version](https://img.shields.io/pypi/v/spatialreasoners.svg)](https://pypi.org/project/spatialreasoners/)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://spatialreasoners.github.io/spatialreasoners/)

## Overview

<p align="center">
  <img src="https://github.com/spatialreasoners/spatialreasoners/blob/rc/sr-0.1-clean/assets/overview.png?raw=true" alt="Spatial Reasoners Overview"/>
</p>

🌀Spatial Reasoners is a Python package for spatial reasoning over continuous variables with generative denoising models. Denoising generative models have become the de-facto standard for image generation, due to their effectiveness in sampling from complex, high-dimensional distributions. Recently, they have started being explored in the context of reasoning over multiple continuous variables.

Our package provides a comprehensive framework to facilitate research in this area, offering easy-to-use interfaces to control:

*   **Variable Mapping:** Seamlessly map variables from arbitrary data domains.
*   **Generative Model Paradigms:** Flexibly work with a wide range of denoising formulations.
*   **Samplers & Inference Strategies:** Implement and experiment with diverse samplers and inference techniques.

🌀Spatial Reasoners is a generalization of [Spatial Reasoning Models (SRMs)](https://geometric-rl.mpi-inf.mpg.de/srm/) to new domains, packaged as a reusable library for the research community.

## 🛠️ Installation

### Quick Install (Recommended)

Install Spatial Reasoners directly from PyPI:

```bash
pip install spatialreasoners
```

### Development Install

For development or to use the latest features:

```bash
git clone https://github.com/spatialreasoners/spatialreasoners.git
cd spatialreasoners
pip install -e .
```

### Requirements

- Python 3.11+ (Recommended: 3.13)
- PyTorch 1.13+
- PyTorch Lightning 2.0+

## 🚀 Quick Start

### Basic Usage with predefined experiments

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

## 🏗️ Custom Projects & Training

Spatial Reasoners provides two clean approaches for creating custom research projects with your own datasets and models.

### Method 1: @sr.config_main Decorator (Recommended)

The cleanest interface for most use cases - similar to `@hydra.main` but with automatic config merging.

**Create your training script** (`training.py`):
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

**Advantages:**
- ✅ **Cleanest interface** - just like `@hydra.main`
- ✅ **Automatic config merging** (local + embedded configs)
- ✅ **No boilerplate code** - just import, decorate, and run
- ✅ **Enhanced help system** with examples
- ✅ **Full control** - inspect and modify config before training
- ✅ **Beartype integration** via `--enable-beartype` flag

### Method 2: Programmatic Configuration

For automation, notebooks, or when you need to generate configurations dynamically.

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

**Advantages:**
- ✅ **Programmatic control** - generate configs dynamically
- ✅ **Easy integration** into larger Python programs
- ✅ **Good for automation** - scripts, pipelines, notebooks
- ✅ **No CLI complexity** - simple function calls

### Configuration Structure

Organize your project with this recommended structure:

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

**Example main config** (`configs/main.yaml`):
```yaml
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

**Example experiment config** (`configs/experiment/my_experiment.yaml`):
```yaml
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

### Register Custom Components

Define custom components in your `src/` directory and auto-register them:

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
# src/denoiser.py  
import spatialreasoners as sr
from spatialreasoners.denoising_model.denoiser import register_denoiser, DenoiserCfg

@dataclass
class MyModelCfg(DenoiserCfg):
    name: str = "my_model"
    hidden_dim: int = 256

@register_denoiser("my_model", MyModelCfg)
class MyModel(sr.Denoiser):
    def __init__(self, cfg: MyModelCfg, tokenizer, num_classes=None):
        # Your model implementation
        pass
```

```python
# src/__init__.py - Auto-register all components
from . import dataset
from . import denoiser
from . import variable_mapper
from . import tokenizer
# Add other component imports as needed
```

### Config Merging

🌀Spatial Reasoners automatically merges your local configs with embedded configurations:

- **Local configs take precedence** - your custom components override built-in ones
- **Built-in components remain accessible** - use `dataset=cifar10`, `denoising_model.flow=rectified`, etc.
- **Seamless composition** - mix and match local and embedded components freely

### Quick Comparison

| Method | Interface | CLI Support | Setup | Best For |
|--------|-----------|-------------|-------|----------|
| `@sr.config_main` | Decorator | ✅ Automatic | Minimal | General use, research, experimentation |
| Programmatic | Function | ❌ None | Minimal | Automation, notebooks, production |

**Recommendation:** Start with Method 1 (`@sr.config_main`) for most use cases. Use Method 2 for automation or when generating configurations dynamically.

## 📖 Documentation & Examples

### Example Projects

Check out the `example_spiral_project/` directory for a complete working example that demonstrates:

- **Two training approaches**: `@sr.config_main` decorator and programmatic configuration
- **Custom component organization**: Structured `src/` directory with auto-registration
- **Config composition**: Local configs that reference embedded Spatial Reasoners components
- **Professional workflows**: Proper project structure for research

The example implements a spiral dataset where the model learns to generate points along a spiral pattern, showcasing:
- Custom dataset, variable mapper, tokenizer, and denoiser implementations
- Clean configuration management with experiment-specific configs
- Visualization and evaluation during training

**Run the example:**
```bash
cd example_spiral_project

# Method 1: @sr.config_main decorator (recommended)
python training_decorator.py experiment=spiral_training

# Method 2: Programmatic configuration  
python training_programmatic.py
```

### Configuration System

Spatial Reasoners uses Hydra for flexible configuration management with automatic merging between your local configs and embedded components.

**Key Configuration Concepts:**

- **Main Config** (`configs/main.yaml`): Project-wide defaults and structure
- **Experiments** (`configs/experiment/`): Complete task-specific configurations
- **Component Configs**: Modular configs for datasets, models, etc.
- **Embedded Components**: Built-in configs from Spatial Reasoners (datasets, flows, optimizers)

**Advanced Configuration Loading:**
```python
# Multiple ways to load and customize configs
config = sr.load_default_config()                    # Built-in api_default experiment
config = sr.load_config_from_yaml(overrides=["experiment=mnist_sudoku"])
config = sr.load_config_from_yaml("./configs", "main", ["experiment=custom"])

# Programmatic config modification
config.trainer.max_epochs = 100
config.data_loader.train.batch_size = 32
```

**CLI Configuration:**
```bash
# Use embedded experiments
python training.py experiment=mnist_sudoku

# Override any nested parameter
python training.py experiment=mnist_sudoku trainer.max_epochs=100 data_loader.train.batch_size=64

# Mix local and embedded components  
python training.py experiment=my_experiment denoising_model.flow=cosine optimizer=adamw
```

## 💾 Datasets & Checkpoints

### Datasets
We provide datasets from the original SRM project. Download them from the [SRM releases](https://github.com/Chrixtar/SRM/releases):

```bash
# Extract datasets.zip to your data directory
mkdir -p data
cd data
wget https://github.com/Chrixtar/SRM/releases/download/v1.0/datasets.zip
unzip datasets.zip
```

For FFHQ-based datasets, download [FFHQ](https://github.com/NVlabs/ffhq-dataset) and update the path in your dataset config.

### Pretrained Models
Download pretrained checkpoints from the [SRM releases](https://github.com/Chrixtar/SRM/releases):

```bash
mkdir -p checkpoints
cd checkpoints
wget https://github.com/Chrixtar/SRM/releases/download/v1.0/checkpoints.zip
unzip checkpoints.zip
```

## 📊 Research & Benchmarks

### Running Benchmarks

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

## 🏗️ Architecture

Spatial Reasoners is built with modularity and extensibility in mind:

```
spatialreasoners/
├── api/                  # High-level API
├── dataset/              # Data loading and processing
├── denoising_model/      # Model implementations
│   ├── denoiser/         # Denoiser architectures (UNet, DiT, MAR, etc.)
│   ├── flow/             # Flow variants (rectified, cosine, etc.)
│   └── tokenizer/        # Tokenizers of variables for the denoiser
├── training/             # Training infrastructure
├── variable_mapper/      # Variable mapping logic
├── benchmark/            # Evaluation framework
└── configs/              # Embedded default configs
```

### Key Components

- **Variable Mappers**: Transform between data domains and model representations
- **Denoising Models**: Various architectures (UNet, DiT, MAR, etc.)
- **Flow Models**: Different denoising formulations and schedules
- **Training System**: PyTorch Lightning-based training with full configurability
- **Benchmark Suite**: Standardized evaluation protocols

## 🔬 Research Applications

Spatial Reasoners has been used for research in:

- **Spatial reasoning tasks** (MNIST Sudoku, polygon counting)
- **Image generation** where there could be some spatial dependencies between regions of the image
- **Video generation** such as in [Diffusion Forcing](https://www.boyuan.space/diffusion-forcing/)



### Citation

If you use Spatial Reasoners in your research, please cite:

```bibtex
@software{pogodzinski25spatialreasoners,
  title={Spatial Reasoners: A Framework for Spatial Reasoning with Generative Models},
  author={Pogodzinski, Bart and Wewer, Christopher and Lenssen, Jan Eric and Schiele, Bernt},
  year={2025},
  url={https://github.com/spatialreasoners/spatialreasoners}
}

@inproceedings{wewer25srm,
    title     = {Spatial Reasoning with Denoising Models},
    author    = {Wewer, Christopher and Pogodzinski, Bartlomiej and Schiele, Bernt and Lenssen, Jan Eric},
    booktitle = {International Conference on Machine Learning ({ICML})},
    year      = {2025},
}
```

## 🤝 Contributing

We welcome contributions from the research community! Here's how you can help:

### Ways to Contribute

- **New Models**: Implement novel denoising architectures
- **Datasets**: Add support for new spatial reasoning tasks
- **Benchmarks**: Contribute evaluation protocols
- **Documentation**: Improve docs and examples
- **Bug Reports**: Report issues and suggest improvements

### Development Setup

```bash
git clone https://github.com/spatialreasoners/spatialreasoners.git
cd spatialreasoners
pip install -e ".[dev]"
```

<!-- ### Running Tests

```bash
pytest tests/
``` -->

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋 Support & Community

- **Documentation**: [spatialreasoners.github.io/spatialreasoners](https://spatialreasoners.github.io/spatialreasoners/)
- **Issues**: [GitHub Issues](https://github.com/spatialreasoners/spatialreasoners/issues)
- **Discussions**: [GitHub Discussions](https://github.com/spatialreasoners/spatialreasoners/discussions)
- **Email**: bpogodzi@mpi-inf.mpg.de
