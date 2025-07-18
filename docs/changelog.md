# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Documentation website with MkDocs
- Comprehensive API reference
- Getting started guide and quick tour

### Changed
- TBD

### Deprecated
- TBD

### Removed
- TBD

### Fixed
- TBD

### Security
- TBD

## [1.0.0] - 2025-07-14

### Added
- **Core Framework**: Complete spatial reasoning framework with generative denoising models
- **High-Level API**: One-line training with `sr.run_training()`
- **Configuration System**: Hydra-based configuration with automatic merging
- **Component Registry**: Pluggable system for datasets, models, and variable mappers
- **Training Infrastructure**: PyTorch Lightning-based training with distributed support

#### Datasets
- MNIST Sudoku puzzle dataset
- CIFAR-10 image dataset  
- Polygon counting dataset
- Example spiral dataset for demonstrations

#### Denoising Models
- **Flows**: Rectified flow, cosine, and linear noise schedules
- **Denoisers**: U-Net, DiT (Diffusion Transformer) variants
- **Tokenizers**: Flexible tokenization for different data types

#### Variable Mappers
- Image variable mapping for visual data
- Continuous variable mapping for real-valued data
- Discrete variable mapping for categorical data

#### Training Features
- **Two Training Approaches**: `@sr.config_main` decorator and programmatic configuration
- **Configuration Merging**: Automatic merging of local and embedded configurations
- **Type Safety**: Optional beartype integration for runtime type checking
- **CLI Support**: Full command-line interface with help

#### Developer Experience
- **Example Projects**: Complete spiral dataset example with multiple training approaches
- **Auto-Registration**: Automatic component registration system
- **Professional Structure**: Clean project organization patterns
- **Comprehensive Documentation**: API reference, guides, and examples

#### Evaluation & Benchmarks
- Built-in evaluation protocols
- Standard benchmark implementations
- Model checkpointing and loading utilities
- Visualization tools for training progress

### Technical Details
- **Python**: 3.11+ support (recommended: 3.13)
- **PyTorch**: 1.13+ compatibility
- **Wandb**: logging directly to your Wandb environment
- **PyTorch Lightning**: 2.0+ for training infrastructure
- **Hydra**: Advanced configuration management
- **Mixed Precision**: FP16 training support
- **Multi-GPU**: Distributed training capabilities

### Research Applications
- Spatial reasoning tasks (MNIST Sudoku, polygon counting)
- Image generation with spatial dependencies
- Video generation (compatible with Diffusion Forcing)
- Custom continuous variable reasoning tasks


# Contributing to Changelog


When contributing:

1. **Add entries to [Unreleased]** for new changes
2. **Follow the format**: Use the categories (Added, Changed, etc.)
3. **Be descriptive**: Explain what changed and why it matters
4. **Link to issues/PRs**: Reference relevant GitHub issues or pull requests
5. **User-focused**: Write for users, not developers

### Example Entry Format

```markdown
### Added
- **New Dataset Support**: Added FFHQ-based spatial reasoning dataset ([#123](https://github.com/spatialreasoners/spatialreasoners/pull/123))
- **Model Architecture**: Implemented Vision Transformer denoiser variant ([#124](https://github.com/spatialreasoners/spatialreasoners/pull/124))

### Fixed
- **Configuration Loading**: Fixed issue with nested config overrides not applying correctly ([#125](https://github.com/spatialreasoners/spatialreasoners/issues/125))
```

## Version History

- **v1.0.0**: Initial release with complete framework
- **v0.x.x**: Pre-release development versions (not publicly released)

---

<!-- ## Release Notes Template

When adding new releases, use this template:

```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- New features and capabilities

### Changed  
- Changes to existing functionality

### Deprecated
- Features marked for removal in future versions

### Removed
- Features removed in this version

### Fixed
- Bug fixes and corrections

### Security
- Security-related improvements
```
 -->