#!/usr/bin/env python3
"""
Example showing how to use SpatialReasoners with Hydra CLI patterns.

This demonstrates both custom Hydra usage and the built-in hydra_main function.
"""

import sys
import os
# Add the project root to the Python path for development usage
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import hydra
from omegaconf import DictConfig
import src as spatialreasoners


@hydra.main(version_base=None, config_path="../config", config_name="main")
def custom_hydra_main(cfg: DictConfig) -> None:
    """
    Custom Hydra main function with your own logic.
    
    This allows you to:
    - Add custom preprocessing before training
    - Handle different modes or workflows
    - Integrate with other systems
    - Add custom logging or monitoring
    """
    print("=== Custom Hydra CLI Training ===")
    print(f"Loaded config keys: {list(cfg.keys())}")
    
    # Example: Custom preprocessing
    if hasattr(cfg, 'custom_preprocessing'):
        print("Running custom preprocessing...")
        # Your custom logic here
    
    # Convert to typed config for type safety
    typed_config = spatialreasoners.config.load_typed_root_config(cfg)
    print(f"Dataset: {type(typed_config.dataset).__name__}")
    print(f"Model: {type(typed_config.denoising_model.denoiser).__name__}")
    
    # Create components using the API
    data_module = spatialreasoners.create_data_module(typed_config)
    lightning_module = spatialreasoners.create_lightning_module(typed_config)
    trainer = spatialreasoners.create_trainer(typed_config)
    
    # Run based on mode
    if typed_config.mode == "train":
        print("Starting training...")
        trainer.fit(lightning_module, datamodule=data_module)
    elif typed_config.mode == "val":
        print("Running validation...")
        trainer.validate(lightning_module, datamodule=data_module)
    elif typed_config.mode == "test":
        print("Running test...")
        trainer.test(lightning_module, datamodule=data_module)
    else:
        raise ValueError(f"Unknown mode: {typed_config.mode}")


@hydra.main(version_base=None, config_path="../config", config_name="main")  
def simple_hydra_main(cfg: DictConfig) -> None:
    """
    Simple approach: use SpatialReasoners' built-in Hydra handler.
    
    This is the easiest way to use Hydra with SpatialReasoners.
    """
    print("=== Simple Hydra CLI Training ===")
    spatialreasoners.hydra_main(cfg)


def example_direct_api_usage():
    """
    Example showing direct API usage without Hydra decorators.
    Useful for Jupyter notebooks or custom scripts.
    """
    print("=== Direct API Usage (No Hydra Decorators) ===")
    
    # Load config with overrides programmatically
    config = spatialreasoners.load_config_from_yaml(
        config_name="main",
        overrides=[
            "experiment=api_default",
            "trainer.max_steps=5",
            "wandb.mode=disabled"
        ]
    )
    
    print(f"Loaded config: {type(config).__name__}")
    
    # Use the high-level training function
    spatialreasoners.run_training(
        overrides=[
            "experiment=api_default", 
            "trainer.max_steps=1",
            "wandb.mode=disabled"
        ]
    )


if __name__ == "__main__":
    print("🚀 SpatialReasoners Hydra CLI Examples")
    print("=" * 50)
    
    # For CLI usage, you would typically only call one of these:
    
    # Method 1: Custom Hydra main (uncomment to use)
    # custom_hydra_main()
    
    # Method 2: Simple built-in Hydra main (uncomment to use)  
    # simple_hydra_main()
    
    # Method 3: Direct API usage (works without Hydra decorators)
    print("Running direct API example...")
    try:
        example_direct_api_usage()
    except Exception as e:
        print(f"Example completed with: {e}")


"""
Command line usage examples for this script:

# Basic training with defaults
python hydra_cli_usage.py experiment=api_default

# Override specific parameters
python hydra_cli_usage.py experiment=api_default trainer.max_epochs=50

# Use different experiment
python hydra_cli_usage.py experiment=even_pixels

# Multiple overrides
python hydra_cli_usage.py experiment=api_default \\
    trainer.max_steps=1000 \\
    data_loader.train.batch_size=64 \\
    wandb.mode=offline

# Different modes
python hydra_cli_usage.py experiment=api_default mode=val
python hydra_cli_usage.py experiment=api_default mode=test

# Multirun experiments (sweep parameters)
python hydra_cli_usage.py --multirun \\
    experiment=api_default \\
    trainer.max_epochs=10,20,50

# Complex nested overrides
python hydra_cli_usage.py experiment=api_default \\
    denoising_model.learn_sigma=false \\
    denoising_model.tokenizer.num_tokens_per_spatial_dim=4

# Disable wandb
python hydra_cli_usage.py experiment=api_default wandb.mode=disabled

# Custom batch sizes and learning rates
python hydra_cli_usage.py experiment=api_default \\
    data_loader.train.batch_size=128 \\
    optimizer.lr=0.0001
"""