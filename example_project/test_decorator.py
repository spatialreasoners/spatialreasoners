#!/usr/bin/env python3
"""
Simple demo script for Spiral Training using SpatialReasoners @sr.config_main decorator

Run this script from the project root directory with CLI overrides:
    python training_decorator.py experiment=spiral_training
    python training_decorator.py experiment=spiral_training trainer.max_epochs=100
    python training_decorator.py experiment=spiral_training dataset.n_samples=20000
    python training_decorator.py --help  # Show available options
"""

import spatialreasoners as sr

import src  # This imports and registers all spiral components 

@sr.config_main(config_path="configs", config_name="main_spiral")
def demo_spiral_testing(cfg):
    """Demonstrate spiral training with @sr.config_main decorator."""
    
    assert cfg.mode == "test", "This script is only for testing"
    
    # Create components from the loaded config
    lightning_module = sr.create_lightning_module(cfg)
    data_module = sr.create_data_module(cfg)
    trainer = sr.create_trainer(cfg)
    
    checkpoint_path = cfg.checkpointing.load
    assert checkpoint_path is not None, "Checkpoint path is not set"
    
    print("✅ Components created successfully!")
    print()
    
    # You have full control here - can add custom callbacks, modify trainer, etc.
    print("🎯 Starting testing with custom control...")
    trainer.test(lightning_module, datamodule=data_module, ckpt_path=checkpoint_path)
    
    print("🎉 Testing complete!")


if __name__ == "__main__":
    demo_spiral_testing() 