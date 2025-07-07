import os
import tempfile
from pathlib import Path
from typing import Optional, Union, Dict, Any, Callable
import importlib.resources
import shutil
import sys
import argparse
from functools import wraps

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.profilers import AdvancedProfiler
import torch
import wandb

from .config import load_typed_root_config, RootCfg
from .dataset import get_dataset
from .env import DEBUG
from .global_cfg import set_cfg
from .misc.local_logger import LocalLogger
from .misc.wandb_tools import update_checkpoint_path
from .training import SRMLightningModule, DataModule
from .variable_mapper import get_variable_mapper


def enable_beartype_checking():
    """
    Enable beartype checking for improved type safety and error messages.
    
    This function should be called before importing other SpatialReasoners modules
    if you want enhanced type checking. It mimics the behavior from main.py.
    
    Example:
        import spatialreasoners as sr
        sr.enable_beartype_checking()  # Enable before using other functions
        
        # Now use SpatialReasoners with enhanced type checking
        config = sr.load_default_config()
        model = sr.create_lightning_module(config)
    """
    try:
        from jaxtyping import install_import_hook
        
        # Configure beartype and jaxtyping for the src package
        with install_import_hook(
            ("spatialreasoners",),
            ("beartype", "beartype"),
        ):
            pass  # The hook is now installed
        
        print("✅ Beartype checking enabled for enhanced type safety")
        
    except ImportError:
        print("⚠️  Beartype/jaxtyping not available - install with: pip install beartype jaxtyping")
        print("   Continuing without enhanced type checking...")


def _get_embedded_config_dir() -> Path:
    """
    Get path to embedded configs, supporting both development and packaged usage.
    
    This function handles the config location in different scenarios:
    1. Development: Uses the config/ directory parallel to src/
    2. Packaged: Uses embedded configs within the package
    3. Custom: Allows users to specify their own config directories
    
    Returns:
        Path to the config directory to use
    """
    try:
        # Try to get embedded configs from package resources (when installed)
        with importlib.resources.path("src.configs", "__init__.py") as config_init:
            embedded_config_dir = config_init.parent
            if embedded_config_dir.exists():
                return embedded_config_dir
    except (ImportError, FileNotFoundError, ModuleNotFoundError):
        pass
    
    # Development fallback: look for config directory parallel to src
    current_file = Path(__file__)
    src_dir = current_file.parent
    project_root = src_dir.parent
    config_dir = project_root / "config"
    
    if config_dir.exists():
        return config_dir
    
    # Last resort: copy configs from source to temp directory for packaging
    temp_dir = Path(tempfile.mkdtemp(prefix="spatialreasoners_configs_"))
    
    if config_dir.exists():
        shutil.copytree(config_dir, temp_dir / "config", dirs_exist_ok=True)
        return temp_dir / "config"
    
    raise FileNotFoundError(
        "Could not locate SpatialReasoners config directory. "
        "This might happen if the package is not properly installed or "
        "if you're running from an unexpected location."
    )


def load_config_from_yaml(
    config_path: Optional[Union[str, Path]] = None,
    config_name: str = "main",
    overrides: Optional[list] = None,
    return_hydra_cfg: bool = False,
) -> Union[RootCfg, DictConfig]:
    """
    Load configuration from YAML files with SpatialReasoners' embedded configs.
    
    This function provides flexible config loading that works in multiple scenarios:
    - Uses SpatialReasoners' embedded configs by default
    - Allows custom config directories for user projects
    - Supports Hydra's full composition and override system
    - When custom config_path is provided, both custom and embedded configs are available
      (custom configs take precedence and can reference embedded components)
    
    Args:
        config_path: Path to config directory. If None, uses embedded configs only.
                    If provided, custom configs take precedence but can still reference
                    embedded configs (e.g., dataset=cifar10, denoising_model.denoiser=dit_l_2)
        config_name: Name of the main config file (without .yaml extension)
        overrides: List of config overrides in Hydra format (e.g., ["dataset=mnist", "trainer.max_epochs=100"])
        return_hydra_cfg: If True, returns raw Hydra DictConfig. If False, returns typed RootCfg.
        
    Returns:
        Loaded configuration (typed or raw depending on return_hydra_cfg)
        
    Examples:
        # Use SpatialReasoners' default configs
        config = load_config_from_yaml()
        
        # Use custom config directory (can still reference embedded components)
        config = load_config_from_yaml("/path/to/my/configs", "my_experiment")
        
        # Override parameters
        config = load_config_from_yaml(overrides=["dataset=mnist", "trainer.max_epochs=50"])
        
        # For projects combining SpatialReasoners + custom configs
        config = load_config_from_yaml(
            config_path="./my_research_configs",
            config_name="experiment", 
            overrides=["model=my_custom_model", "dataset=cifar10"]  # cifar10 from embedded configs
        )
    """
    import tempfile
    
    # Always get embedded config directory for search path
    embedded_config_dir = _get_embedded_config_dir()
    
    # Clean up any existing Hydra instance
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    
    if config_path is None:
        # Use embedded configs only
        with initialize_config_dir(config_dir=str(embedded_config_dir.absolute()), version_base=None):
            cfg = compose(config_name=config_name, overrides=overrides or [])
    else:
        # Create a temporary merged config directory
        # This ensures Hydra can find all configs properly
        temp_dir = Path(tempfile.mkdtemp(prefix=".spatialreasoners_merged_configs_"))
        
        try:
            custom_config_path = Path(config_path).absolute()
            
            # Copy embedded configs to temp directory
            shutil.copytree(embedded_config_dir, temp_dir / "embedded", dirs_exist_ok=True)
            
            # Copy custom configs to temp directory (these will override embedded ones)
            shutil.copytree(custom_config_path, temp_dir / "custom", dirs_exist_ok=True)
            
            # Create a merged structure where custom configs take precedence
            # First copy everything from embedded
            for item in (temp_dir / "embedded").rglob("*.yaml"):
                if item.is_file():
                    rel_path = item.relative_to(temp_dir / "embedded")
                    target = temp_dir / rel_path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, target)
            
            # Then copy custom configs (overwriting where needed)
            for item in (temp_dir / "custom").rglob("*.yaml"):
                if item.is_file():
                    rel_path = item.relative_to(temp_dir / "custom")
                    target = temp_dir / rel_path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, target)
            
            # Now use the merged directory
            with initialize_config_dir(config_dir=str(temp_dir.absolute()), version_base=None):
                cfg = compose(config_name=config_name, overrides=overrides or [])
                
        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    if return_hydra_cfg:
        return cfg
    else:
        # Convert to typed configuration
        return load_typed_root_config(cfg)


def load_default_config() -> RootCfg:
    """
    Load SpatialReasoners' default configuration.
    
    This function loads a working experiment configuration. Since SpatialReasoners
    is designed around experiments, you need to specify which experiment to use.
    This function uses 'api_default' as a simple default.
    
    For custom usage, it's better to call:
    load_config_from_yaml(config_name="main", overrides=["experiment=your_experiment"])
    
    Returns:
        Default typed configuration (api_default experiment)
    """
    # Use main config with api_default experiment
    return load_config_from_yaml(
        config_name="main",
        overrides=["experiment=api_default"]
    )


def create_lightning_module(
    config: Union[RootCfg, DictConfig, str, Path], 
    num_classes: Optional[int] = None,
    variable_mapper = None,
    **kwargs
) -> SRMLightningModule:
    """
    Create a SpatialReasoners Lightning module from configuration.
    
    This function matches the logic from _main.py for proper component creation.
    
    Args:
        config: Configuration (typed config, DictConfig, or path to config file)
        num_classes: Number of classes (inferred from dataset if None)
        variable_mapper: Variable mapper instance (created from config if None)
        **kwargs: Additional arguments to pass to the Lightning module
        
    Returns:
        Configured SRMLightningModule ready for training
        
    Examples:
        # From typed config
        config = load_default_config()
        model = create_lightning_module(config)
        
        # From config file
        model = create_lightning_module("path/to/config.yaml")
        
        # With custom components
        model = create_lightning_module(config, num_classes=10, variable_mapper=my_mapper)
    """
    
    # Handle different config input types
    if isinstance(config, (str, Path)):
        config = load_config_from_yaml(config_path=Path(config).parent, 
                                     config_name=Path(config).stem)
    elif isinstance(config, DictConfig):
        config = load_typed_root_config(config)
    elif not isinstance(config, RootCfg):
        raise TypeError(f"Expected RootCfg, DictConfig, str, or Path, got {type(config)}")
    
    # Create dataset to get num_classes if not provided
    if num_classes is None or variable_mapper is None:
        dataset = get_dataset(cfg=config.dataset, conditioning_cfg=config.denoising_model.conditioning, stage="train")
        if num_classes is None:
            num_classes = dataset.num_classes
        if variable_mapper is None:
            unstructured_sample_shape = config.dataset.data_shape
            variable_mapper = get_variable_mapper(config.variable_mapper, unstructured_sample_shape)
    
    # Create Lightning module with proper parameters (matching _main.py)
    lightning_module = SRMLightningModule(
        cfg=config,  # Full config, not just training section
        num_classes=num_classes,
        variable_mapper=variable_mapper,
        **kwargs
    )
    
    return lightning_module


def create_data_module(
    config: Union[RootCfg, DictConfig, str, Path],
    variable_mapper = None,
    step_tracker = None,
    **kwargs
) -> DataModule:
    """
    Create a SpatialReasoners data module from configuration.
    
    This function matches the logic from _main.py for proper data module creation.
    
    Args:
        config: Configuration (typed config, DictConfig, or path to config file)
        variable_mapper: Variable mapper instance (created from config if None)
        step_tracker: Step tracker for sharing current step with data loader processes
        **kwargs: Additional arguments to pass to the data module
        
    Returns:
        Configured DataModule ready for training
    """
    
    # Handle different config input types  
    if isinstance(config, (str, Path)):
        config = load_config_from_yaml(config_path=Path(config).parent,
                                     config_name=Path(config).stem)
    elif isinstance(config, DictConfig):
        config = load_typed_root_config(config)
    elif not isinstance(config, RootCfg):
        raise TypeError(f"Expected RootCfg, DictConfig, str, or Path, got {type(config)}")
    
    # Create variable mapper if not provided
    if variable_mapper is None:
        unstructured_sample_shape = config.dataset.data_shape
        variable_mapper = get_variable_mapper(config.variable_mapper, unstructured_sample_shape)
    
    # Create data module with proper parameters (matching _main.py)
    data_module = DataModule(
        dataset_cfg=config.dataset, 
        data_loader_cfg=config.data_loader, 
        conditioning_cfg=config.denoising_model.conditioning,
        variable_mapper=variable_mapper,
        validation_benchmark_cfgs=config.validation_benchmarks, 
        test_benchmark_cfgs=config.test_benchmarks, 
        step_tracker=step_tracker,
        output_dir=config.output_dir,
        **kwargs
    )
    
    return data_module


def create_trainer(
    config: Union[RootCfg, DictConfig, str, Path],
    callbacks: Optional[list] = None,
    logger = None,
    **kwargs
) -> Trainer:
    """
    Create a PyTorch Lightning trainer from configuration.
    
    This function matches the logic from _main.py for proper trainer creation.
    
    Args:
        config: Configuration (typed config, DictConfig, or path to config file)
        callbacks: List of callbacks (basic ones created if None)
        logger: Logger instance (LocalLogger created if None)
        **kwargs: Additional arguments to pass to the trainer
        
    Returns:
        Configured PyTorch Lightning Trainer
    """
    
    # Handle different config input types
    if isinstance(config, (str, Path)):
        config = load_config_from_yaml(config_path=Path(config).parent,
                                     config_name=Path(config).stem)
    elif isinstance(config, DictConfig):
        config = load_typed_root_config(config)
        
    elif not isinstance(config, RootCfg):
        raise TypeError(f"Expected RootCfg, DictConfig, str, or Path, got {type(config)}")
    
    # Create default logger if none provided
    if logger is None:
        logger = LocalLogger()
    
    # Create default callbacks if none provided
    if callbacks is None:
        callbacks = []
    
    # Create trainer with full configuration (matching _main.py)
    trainer_kwargs = {
        "accelerator": "gpu",
        "logger": logger,
        "devices": "auto",
        "num_nodes": config.trainer.num_nodes,
        "precision": config.trainer.precision,
        "strategy": "ddp" if torch.cuda.device_count() > 1 else "auto",
        "callbacks": callbacks,
        "limit_val_batches": None if config.trainer.validate else 0,
        "val_check_interval": config.trainer.val_check_interval if config.trainer.validate else None,
        "check_val_every_n_epoch": None,
        "log_every_n_steps": config.trainer.log_every_n_steps,
        "enable_checkpointing": config.checkpointing.save,
        "enable_progress_bar": DEBUG or config.mode != "train",
        "accumulate_grad_batches": config.trainer.accumulate_grad_batches,
        "gradient_clip_val": config.optimizer.gradient_clip_val,
        "gradient_clip_algorithm": config.optimizer.gradient_clip_algorithm,
        "max_epochs": config.trainer.max_epochs,
        "max_steps": config.trainer.max_steps,
        "profiler": AdvancedProfiler(filename="profile") if config.trainer.profile else None,
        "detect_anomaly": config.trainer.detect_anomaly,
        **kwargs
    }
    
    return Trainer(**trainer_kwargs)


def run_training(
    config_path: Optional[Union[str, Path]] = None,
    config_name: str = "main",
    overrides: Optional[list] = None,
    enable_beartype: bool = False,
    **trainer_kwargs
) -> None:
    """
    High-level function to run training with SpatialReasoners.
    
    This is the main entry point for training. It handles:
    - Loading configuration (embedded or custom)
    - Creating all necessary components
    - Setting up the trainer with proper callbacks and logging
    - Running the training loop
    
    Args:
        config_path: Path to config directory (uses embedded configs if None)
        config_name: Name of config file (defaults to main)
        overrides: List of parameter overrides
        enable_beartype: Whether to enable beartype checking for better error messages
        **trainer_kwargs: Additional trainer arguments
        
    Examples:
        # Basic training with defaults (mnist_sudoku experiment)
        run_training()
        
        # With enhanced type checking
        run_training(enable_beartype=True)
        
        # Different experiment
        run_training(config_name="experiment/even_pixels")
        
        # Custom overrides
        run_training(overrides=["trainer.max_epochs=100"])
        
        # Research project with custom configs
        run_training(
            config_path="./my_research_configs",
            config_name="experiment/my_experiment"
        )
        
        # Quick debugging
        run_training(overrides=["trainer.fast_dev_run=True"])
    """
    
    # Enable beartype checking if requested
    if enable_beartype:
        enable_beartype_checking()
    
    print("🚀 Starting SpatialReasoners Training")
    print("=" * 50)
    
    # Load configuration
    print("📋 Loading configuration...")
    
    # If no experiment is specified, use api_default as default
    final_overrides = overrides or []
    if not any(override.startswith("experiment=") or override.startswith("+experiment=") for override in final_overrides):
        final_overrides = ["experiment=api_default"] + final_overrides 
        print("🔧 Using 'api_default' as default experiment")
    
    config = load_config_from_yaml(
        config_path=config_path,
        config_name=config_name, 
        overrides=final_overrides,
        return_hydra_cfg=True  # Keep as DictConfig for global_cfg
    )
    
    # Set global config for compatibility
    set_cfg(config)
    
    # Convert to typed config
    typed_config = load_typed_root_config(config)
    
    # Set random seed
    if typed_config.seed is not None:
        seed_everything(typed_config.seed, workers=True)
    
    # Set torch variables
    if typed_config.torch.float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(typed_config.torch.float32_matmul_precision)
    torch.backends.cudnn.benchmark = typed_config.torch.cudnn_benchmark
    
    # Create components
    print("🔧 Creating components...")
    
    # Create dataset and variable mapper
    dataset = get_dataset(cfg=typed_config.dataset, conditioning_cfg=typed_config.denoising_model.conditioning, stage="train")
    num_classes = dataset.num_classes
    unstructured_sample_shape = typed_config.dataset.data_shape
    variable_mapper = get_variable_mapper(typed_config.variable_mapper, unstructured_sample_shape)
    
    # Create Lightning module
    lightning_module = SRMLightningModule(
        cfg=typed_config,
        num_classes=num_classes,
        variable_mapper=variable_mapper
    )
    
    # Create data module
    data_module = DataModule(
        dataset_cfg=typed_config.dataset, 
        data_loader_cfg=typed_config.data_loader, 
        conditioning_cfg=typed_config.denoising_model.conditioning,
        variable_mapper=variable_mapper,
        validation_benchmark_cfgs=typed_config.validation_benchmarks, 
        test_benchmark_cfgs=typed_config.test_benchmarks, 
        step_tracker=lightning_module.step_tracker,
        output_dir=typed_config.output_dir,
    )
    
    # Create trainer with proper setup
    callbacks = []
    if typed_config.wandb.activated:
        logger = WandbLogger(
            project=typed_config.wandb.project,
            mode=typed_config.wandb.mode,
            tags=typed_config.wandb.tags,
            log_model=False,
            entity=typed_config.wandb.entity
        )
        callbacks.append(LearningRateMonitor("step", True))
        
        if wandb.run is not None:
            wandb.run.log_code("src")
    else:
        logger = LocalLogger()
    
    trainer = create_trainer(typed_config, callbacks=callbacks, logger=logger, **trainer_kwargs)
    
    print("✅ Setup complete!")
    print(f"📊 Dataset: {type(typed_config.dataset).__name__}")
    print(f"🏗️  Model: {type(typed_config.denoising_model.denoiser).__name__}")
    print(f"⚡ Max epochs: {typed_config.trainer.max_epochs}")
    print("=" * 50)
    
    # Start training
    print("🎯 Starting training...")
    trainer.fit(lightning_module, datamodule=data_module)
    
    print("🎉 Training complete!")


# Convenience functions for common workflows
def quick_train(dataset: str = "cifar10", max_epochs: int = 10, enable_beartype: bool = False, **kwargs):
    """Quick training setup for experimentation."""
    return run_training(overrides=[
        f"dataset={dataset}",
        f"trainer.max_epochs={max_epochs}"
    ], enable_beartype=enable_beartype, **kwargs)


def debug_train(enable_beartype: bool = False, **kwargs):
    """Quick debug training setup."""
    return run_training(overrides=[
        "trainer.fast_dev_run=True",
        "trainer.limit_train_batches=2",
        "trainer.limit_val_batches=2"
    ], enable_beartype=enable_beartype, **kwargs)


# Hydra-compatible main function for CLI usage
def hydra_main(cfg: DictConfig) -> None:
    """Main function for Hydra CLI usage."""
    # This matches the logic from _main.py
    typed_config = load_typed_root_config(cfg)
    set_cfg(cfg)
    
    if typed_config.seed is not None:
        seed_everything(typed_config.seed, workers=True)

    # Set torch variables
    if typed_config.torch.float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(typed_config.torch.float32_matmul_precision)
    torch.backends.cudnn.benchmark = typed_config.torch.cudnn_benchmark
    
    # Create components following _main.py logic
    dataset = get_dataset(cfg=typed_config.dataset, conditioning_cfg=typed_config.denoising_model.conditioning, stage="train")
    num_classes = dataset.num_classes
    unstructured_sample_shape = typed_config.dataset.data_shape
    variable_mapper = get_variable_mapper(typed_config.variable_mapper, unstructured_sample_shape)
    
    lightning_module = SRMLightningModule(
        cfg=typed_config,
        num_classes=num_classes,
        variable_mapper=variable_mapper
    )
    
    data_module = DataModule(
        dataset_cfg=typed_config.dataset, 
        data_loader_cfg=typed_config.data_loader, 
        conditioning_cfg=typed_config.denoising_model.conditioning,
        variable_mapper=variable_mapper,
        validation_benchmark_cfgs=typed_config.validation_benchmarks, 
        test_benchmark_cfgs=typed_config.test_benchmarks, 
        step_tracker=lightning_module.step_tracker,
        output_dir=typed_config.output_dir,
    )
    
    # Create trainer and callbacks
    callbacks = []
    if typed_config.wandb.activated:
        logger = WandbLogger(
            project=typed_config.wandb.project,
            mode=typed_config.wandb.mode,
            tags=typed_config.wandb.tags,
            log_model=False,
            entity=typed_config.wandb.entity
        )
        callbacks.append(LearningRateMonitor("step", True))
    else:
        logger = LocalLogger()
    
    trainer = create_trainer(typed_config, callbacks=callbacks, logger=logger)
    
    if typed_config.mode == "train":
        trainer.fit(lightning_module, datamodule=data_module)
    elif typed_config.mode == "val":
        trainer.validate(lightning_module, datamodule=data_module)
    elif typed_config.mode == "test":
        trainer.test(lightning_module, datamodule=data_module)
    else:
        raise ValueError(f"Unknown mode: {typed_config.mode}")


def config_main(
    config_path: Union[str, Path],
    config_name: str = "main",
    version_base: Optional[str] = None
):
    """
    SpatialReasoners configuration decorator with proper config merging.
    
    This decorator handles CLI argument parsing and config loading, then passes
    the loaded configuration to your function. This gives you full control over
    when and how to start training.
    
    The decorator:
    - Parses CLI arguments as Hydra-style overrides
    - Merges local configs with embedded SpatialReasoners configs
    - Loads configuration with proper typing
    - Passes config as first argument to your function
    - Optionally enables beartype checking
    
    Args:
        config_path: Path to your local config directory (relative to script)
        config_name: Name of the main config file (without .yaml extension)
        version_base: Hydra version base (unused but kept for compatibility)
        
    Example:
        ```python
        import spatialreasoners as sr
        from my_custom_classes import *  # Register custom components
        
        @sr.config_main(config_path="configs", config_name="main")
        def my_training_script(cfg):
            print(f"Loaded experiment: {cfg.experiment}")
            print(f"Max epochs: {cfg.trainer.max_epochs}")
            
            # Modify config if needed
            if cfg.trainer.fast_dev_run:
                print("Running in fast dev mode!")
            
            # Start training when ready
            sr.run_training(
                config_path="configs",
                config_name="main", 
                overrides=[],  # Already applied via CLI
            )
            
        if __name__ == "__main__":
            my_training_script()
        ```
        
        Or for more control:
        ```python
        @sr.config_main(config_path="configs", config_name="main") 
        def my_training_script(cfg):
            # Create components manually
            model = sr.create_lightning_module(cfg)
            data = sr.create_data_module(cfg)
            trainer = sr.create_trainer(cfg)
            
            # Custom logic here
            trainer.fit(model, datamodule=data)
        ```
        
        Then run with CLI overrides:
        ```bash
        python my_script.py experiment=my_experiment trainer.max_epochs=100
        python my_script.py experiment=my_experiment --enable-beartype
        python my_script.py --help
        ```
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Parse CLI arguments
            parser = argparse.ArgumentParser(
                description=f"SpatialReasoners Training Script: {func.__name__}",
                formatter_class=argparse.RawDescriptionHelpFormatter,
                epilog=f"""
Examples:
  python {sys.argv[0]} experiment=my_experiment
  python {sys.argv[0]} experiment=my_experiment trainer.max_epochs=100
  python {sys.argv[0]} experiment=my_experiment trainer.fast_dev_run=true
  python {sys.argv[0]} experiment=my_experiment --enable-beartype

Common overrides:
  experiment=NAME                     # Choose experiment configuration
  trainer.max_epochs=N               # Number of training epochs
  wandb.activated=true              # Enable Weights & Biases logging
  
Config files: {config_path}/{config_name}.yaml
                """
            )
            
            parser.add_argument(
                "overrides",
                nargs="*",
                help="Hydra-style configuration overrides (e.g., experiment=my_exp trainer.max_epochs=100)"
            )
            
            parser.add_argument(
                "--enable-beartype",
                action="store_true",
                help="Enable beartype checking for enhanced type safety and error messages"
            )
            
            parsed_args = parser.parse_args()
            
            # Enable beartype if requested
            if parsed_args.enable_beartype:
                enable_beartype_checking()
            
            # Print header
            print(f"🚀 {func.__name__}")
            print("=" * 50)
            
            if parsed_args.overrides:
                print("📋 Configuration overrides:")
                for override in parsed_args.overrides:
                    print(f"   - {override}")
            else:
                print("📋 Using default configuration")
                print(f"   Config: {config_path}/{config_name}.yaml")
                print("   Tip: Add overrides like 'experiment=my_experiment'")
            
            print("=" * 50)
            
            # Load configuration with proper merging
            print("🔧 Loading configuration...")
            cfg = load_config_from_yaml(
                config_path=config_path,
                config_name=config_name,
                overrides=parsed_args.overrides,
                return_hydra_cfg=False  # Return typed config
            )
            
            print("✅ Configuration loaded successfully!")
            print("=" * 50)
            
            # Call the decorated function with config as first argument
            return func(cfg, *args, **kwargs)
            
        return wrapper
    return decorator