"""
Spiral Training Components for SpatialReasoners

This package contains all the custom components needed for spiral training:
- SpiralDataset: Custom dataset for generating spiral data
- SpiralVariablMapper: Maps between structured and unstructured representations
- SpiralTokenizer: Converts variables to model inputs/outputs
- SpiralDenoiser: MLP-based denoiser for spiral data
- SpiralSamplingEvaluation: Visualization and evaluation for spiral samples
- Utility functions for benchmark configuration

Import this package to register all spiral components with SpatialReasoners.
"""

# Import all components to register them with SpatialReasoners
from .dataset import SpiralDataset, SpiralDatasetCfg, generate_spiral_data
from .variable_mapper import SpiralVariablMapper, SpiralVariablMapperCfg
from .tokenizer import SpiralTokenizer, SpiralTokenizerCfg, SpatialDenoiserInputs, SpatialDenoiserOutputs
from .denoiser import SpiralDenoiser, SpiralDenoiserCfg
from .evaluation import SpiralSamplingEvaluation, SpiralSamplingEvaluationCfg

# Export all public components
__all__ = [
    # Dataset
    "SpiralDataset",
    "SpiralDatasetCfg", 
    "generate_spiral_data",
    
    # Variable Mapper
    "SpiralVariablMapper",
    "SpiralVariablMapperCfg",
    
    # Tokenizer
    "SpiralTokenizer",
    "SpiralTokenizerCfg",
    "SpatialDenoiserInputs",
    "SpatialDenoiserOutputs",
    
    # Denoiser
    "SpiralDenoiser",
    "SpiralDenoiserCfg",
    
    # Evaluation
    "SpiralSamplingEvaluation", 
    "SpiralSamplingEvaluationCfg",
]


def register_all_spiral_classes():
    """
    Register all spiral classes. This function ensures all classes are properly
    registered when imported.
    
    Call this function before running training to make sure all components are available.
    Note: Classes are automatically registered when this module is imported.
    """
    print("✅ All spiral classes registered:")
    print("   - SpiralDataset")
    print("   - SpiralVariablMapper") 
    print("   - SpiralTokenizer")
    print("   - SpiralDenoiser")
    print("   - SpiralSamplingEvaluation")


# Auto-register when package is imported
register_all_spiral_classes()
