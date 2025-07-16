#!/usr/bin/env python3
"""
Basic usage examples for the SpatialReasoners package.

This demonstrates how to use the high-level API for common tasks.
"""

import sys
import os
# Add the project root to the Python path for development usage
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import src as spatialreasoners

# Example 1: Simple training with defaults
def example_default_training():
    """Train with default settings using the api_default experiment."""
    print("=== Example 1: Default Training ===")
    
    # This uses embedded configs and sensible defaults
    # Uses the api_default experiment (CIFAR-10, DiT-S/2, cosine flow)
    spatialreasoners.run_training(
        overrides=[
            "trainer.max_steps=100",  # Short training for demo
            "wandb.mode=disabled"     # Disable wandb for example
        ]
    )


# Example 2: Enhanced type checking and different experiment
def example_with_beartype():
    """Train with enhanced type checking and different experiment."""
    print("=== Example 2: Enhanced Type Checking ===")
    
    # Enable beartype for better error messages
    spatialreasoners.enable_beartype_checking()
    
    spatialreasoners.run_training(
        overrides=[
            "experiment=even_pixels",         # Different experiment
            "trainer.max_steps=50",           # Short training
            "data_loader.train.batch_size=32", # Smaller batch
            "wandb.mode=disabled"
        ],
        enable_beartype=True
    )


# Example 3: Load and modify configuration
def example_config_manipulation():
    """Load configuration and modify it programmatically."""
    print("=== Example 3: Config Manipulation ===")
    
    # Load default config
    config = spatialreasoners.load_default_config()
    print(f"Default dataset: {type(config.dataset).__name__}")
    print(f"Default model: {type(config.denoising_model.denoiser).__name__}")
    
    # Load custom config with overrides
    custom_config = spatialreasoners.load_config_from_yaml(
        config_name="main",
        overrides=[
            "experiment=api_default",
            "trainer.max_epochs=5",
            "data_loader.train.batch_size=16"
        ]
    )
    
    print(f"Custom max epochs: {custom_config.trainer.max_epochs}")
    print(f"Custom batch size: {custom_config.data_loader.train.batch_size}")


# Example 4: Create components individually
def example_individual_components():
    """Create individual components for custom training loops."""
    print("=== Example 4: Individual Components ===")
    
    # Load configuration
    config = spatialreasoners.load_config_from_yaml(
        config_name="main",
        overrides=[
            "experiment=api_default",
            "trainer.max_steps=10"
        ]
    )
    
    # Create components individually
    print("Creating data module...")
    data_module = spatialreasoners.create_data_module(config)
    
    print("Creating lightning module...")
    lightning_module = spatialreasoners.create_lightning_module(config)
    
    print("Creating trainer...")
    trainer = spatialreasoners.create_trainer(config)
    
    print("Components created successfully!")
    print(f"Data module: {type(data_module).__name__}")
    print(f"Lightning module: {type(lightning_module).__name__}")
    print(f"Trainer: {type(trainer).__name__}")
    
    # You can now use these components in custom ways
    # trainer.fit(lightning_module, datamodule=data_module)


# Example 5: Load from custom config directory
def example_custom_config_directory():
    """Load configuration from a custom directory."""
    print("=== Example 5: Custom Config Directory ===")
    
    # This would load from a custom config directory
    # Useful for research projects with their own configs
    try:
        config = spatialreasoners.load_config_from_yaml(
            config_path="./example_project/configs",
            config_name="main",
            overrides=["experiment=baseline"]
        )
        print("Successfully loaded from example_project configs")
        print(f"Config type: {type(config).__name__}")
    except Exception as e:
        print(f"Custom config loading failed (expected): {e}")
        print("This example requires the example_project to be set up properly")


# Example 6: Convenience functions
def example_convenience_functions():
    """Use convenience functions for common scenarios."""
    print("=== Example 6: Convenience Functions ===")
    
    print("Quick training with CIFAR-10:")
    try:
        spatialreasoners.quick_train(
            dataset="cifar10", 
            max_epochs=1,  # Very short for demo
            enable_beartype=True
        )
    except Exception as e:
        print(f"Quick train demo completed with: {e}")
    
    print("\nDebug training mode:")
    try:
        spatialreasoners.debug_train(enable_beartype=True)
    except Exception as e:
        print(f"Debug train demo completed with: {e}")


if __name__ == "__main__":
    print("🧪 SpatialReasoners API Usage Examples")
    print("=" * 50)
    
    # Uncomment the examples you want to run:
    
    # Basic examples (config loading and inspection)
    example_config_manipulation()
    example_individual_components()
    example_custom_config_directory()
    
    # Training examples (uncomment to actually run training)
    # example_default_training()
    # example_with_beartype()
    # example_convenience_functions()
    
    print("\n✅ Examples completed!")
    print("Uncomment the training examples to run actual training.")