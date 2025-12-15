# src/__init__.py
from .api import *

# Registry decorators for user extensions
from .denoising_model.denoiser import register_denoiser
# Note: Other registries would be imported here when they exist:
# from .dataset import register_dataset
# from .training.time_sampler import register_time_sampler
# from .training.loss import register_loss
# from .variable_mapper import register_variable_mapper
# from .benchmark.evaluation import register_evaluation

# Direct component access for advanced users
from .training import SRMLightningModule, SRMLightningModuleCfg, DataModule, DataLoaderCfg
from .config import RootCfg, load_typed_root_config
from .registry import Registry

# Base classes for custom implementations
from .denoising_model.denoiser import Denoiser, DenoiserCfg

__version__ = "0.1.0"

__all__ = [
    # High-level API
    "create_lightning_module",
    "create_data_module", 
    "create_trainer",
    "load_config_from_yaml",
    "load_default_config",
    "run_training",
    "enable_beartype_checking",
    
    # Registry decorators for extensions
    "register_denoiser",
    "register_dataset",
    "register_time_sampler", 
    "register_loss",
    "register_variable_mapper",
    "register_evaluation",
    
    # Base classes for custom implementations
    "Denoiser",
    "DenoiserCfg",
    
    # Direct component access
    "SRMLightningModule",
    "SRMLightningModuleCfg",
    "DataModule",
    "DataLoaderCfg",
    "RootCfg",
    "load_typed_root_config",
    "Registry",
    
    # Version
    "__version__",
]

# Package metadata
__title__ = "spatialreasoners"
__description__ = "SpatialReasoners: A framework for training spatial reasoning models"
__author__ = "SpatialReasoners Team"
__license__ = "MIT"
__url__ = "https://github.com/spatialreasoners/spatialreasoners"

# Make commonly used components easily accessible
def get_version():
    """Get the current version of SpatialReasoners."""
    return __version__

def get_registry():
    """Get the global registry instance."""
    return Registry()

# Convenience imports for common patterns
try:
    import torch
    import pytorch_lightning as pl
    _torch_available = True
except ImportError:
    _torch_available = False

if _torch_available:
    # Only expose torch-dependent functionality if torch is available
    __all__.extend([
        "get_version",
        "get_registry",
    ])