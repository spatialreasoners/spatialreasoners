#!/usr/bin/env python3
"""
Simple demo script for Spiral Training using SpatialReasoners API (Programmatic)

This script demonstrates the programmatic way to run spiral training:
1. Import the spiral classes (auto-registers them)
2. Define overrides as a list of strings
3. Call sr.run_training() with the overrides

This approach is useful when you want to configure training within your code.
For CLI-based configuration, see training_decorator.py instead.

Run this script from the project root directory.
"""

import spatialreasoners as sr

# Step 1: Import spiral classes to register them
import src  # This imports and registers all spiral components 

def demo_spiral_training():
    """Demonstrate spiral training with programmatic configuration."""
    
    print("🌀 Spiral Training Demo (Programmatic)")
    print("=" * 40)
    
    # Step 2: Use spiral_training experiment and just override what we need
    overrides = ["experiment=spiral_training", "mode=train"]
    
    # Step 3: Run training with the overrides
    sr.run_training(
        config_name="main_spiral",
        config_path="configs",
        overrides=overrides,
        enable_beartype=True,
    )
    
    print("🎉 Demo complete!")

if __name__ == "__main__":
    demo_spiral_training() 