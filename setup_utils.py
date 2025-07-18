"""
Custom build utilities for packaging SpatialReasoners with embedded configs.
"""
import os
import shutil
from pathlib import Path
from setuptools.command.build_py import build_py


class BuildPyCommand(build_py):
    """Custom build_py command that embeds config files."""
    
    def run(self):
        # Run the standard build_py command first
        super().run()
        
        # Copy config files to the build directory
        self.copy_configs()
    
    def copy_configs(self):
        """Copy config files from config/ to src/configs/ in the build directory."""
        source_config_dir = Path("config")
        if not source_config_dir.exists():
            print("Warning: config directory not found, skipping config embedding")
            return
        
        # Find the build directory for our package
        build_lib = Path(self.build_lib)
        target_config_dir = build_lib / "spatialreasoners" / "configs"
        
        # Create target directory
        target_config_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy all config files
        print(f"Copying configs from {source_config_dir} to {target_config_dir}")
        
        for config_file in source_config_dir.rglob("*.yaml"):
            # Preserve directory structure
            relative_path = config_file.relative_to(source_config_dir)
            target_file = target_config_dir / relative_path
            
            # Create parent directories if needed
            target_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy the file
            shutil.copy2(config_file, target_file)
            print(f"  Copied: {relative_path}")
        
        # Also copy any .yml files
        for config_file in source_config_dir.rglob("*.yml"):
            relative_path = config_file.relative_to(source_config_dir)
            target_file = target_config_dir / relative_path
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(config_file, target_file)
            print(f"  Copied: {relative_path}")
        
        # Create __init__.py in configs directory
        init_file = target_config_dir / "__init__.py"
        init_file.write_text('"""SpatialReasoners embedded configuration files."""\n')
        
        print(f"✅ Config embedding complete! Embedded configs in {target_config_dir}")


def get_version():
    """Get version from src/__init__.py"""
    init_file = Path("src") / "__init__.py"
    if init_file.exists():
        with open(init_file, 'r') as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('"\'')
    return "0.1.0"


def get_long_description():
    """Get long description from README.md"""
    readme_file = Path("README.md")
    if readme_file.exists():
        return readme_file.read_text(encoding="utf-8")
    return "SpatialReasoners: A framework for training spatial reasoning models"


def get_requirements():
    """Get requirements from requirements.txt if it exists"""
    req_file = Path("requirements.txt")
    if req_file.exists():
        with open(req_file, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return [] 