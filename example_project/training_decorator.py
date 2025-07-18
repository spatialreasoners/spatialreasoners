#!/usr/bin/env python3
"""
Simple demo script for Spiral Training using SpatialReasoners @sr.config_main decorator

This script demonstrates how to use @sr.config_main for flexible training:
1. Import the spiral classes (auto-registers them)
2. Use @sr.config_main decorator to load config from CLI
3. Inspect/modify config as needed
4. Create components manually for full control

Run this script from the project root directory with CLI overrides:
    python training_decorator.py experiment=spiral_training
    python training_decorator.py experiment=spiral_training trainer.max_epochs=100
    python training_decorator.py experiment=spiral_training dataset.n_samples=20000
    python training_decorator.py --help  # Show available options
"""

import spatialreasoners as sr

# Step 1: Import spiral classes to register them
import src  # This imports and registers all spiral components 

@sr.config_main(config_path="configs", config_name="main_spiral")
def demo_spiral_training(cfg):
    """Demonstrate spiral training with @sr.config_main decorator."""
    
    assert cfg.mode == "train", "This script is only for training"
    
    # Create components from the loaded config
    lightning_module = sr.create_lightning_module(cfg)
    data_module = sr.create_data_module(cfg)
    trainer = sr.create_trainer(cfg)
    
    print("✅ Components created successfully!")
    print()
    
    # You have full control here - can add custom callbacks, modify trainer, etc.
    print("🎯 Starting training with custom control...")
    trainer.fit(lightning_module, datamodule=data_module)
    
    print("🎉 Training complete!")


if __name__ == "__main__":
    demo_spiral_training() 