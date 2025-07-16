# Installation

This guide will help you install Spatial Reasoners on your system.

## Requirements

- **Python**: 3.11+ (Recommended: 3.13)
- **PyTorch**: 1.13+
- **PyTorch Lightning**: 2.0+

## Quick Install (Recommended)

Install Spatial Reasoners directly from PyPI:

```bash
pip install spatialreasoners
```

This is the easiest way to get started and includes all required dependencies.

## Development Install

For development or to use the latest features from the repository:

```bash
git clone https://github.com/spatialreasoners/spatialreasoners.git
cd spatialreasoners
pip install -e .
```

The `-e` flag installs the package in "editable" mode, so changes to the source code are immediately reflected.

### Development Dependencies

If you're planning to contribute or need development tools:

```bash
pip install -e ".[dev]"
```

This includes additional dependencies for testing, linting, and documentation.

## Verification

Verify your installation by running:

```python
import spatialreasoners as sr
print(sr.__version__)
```

## Optional: Download Datasets & Checkpoints

### Datasets

We provide datasets from the original SRM project. Download them from the [SRM releases](https://github.com/Chrixtar/SRM/releases):

```bash
# Create data directory
mkdir -p data
cd data

# Download and extract datasets
wget https://github.com/Chrixtar/SRM/releases/download/v1.0/datasets.zip
unzip datasets.zip
```

For FFHQ-based datasets, you'll need to download [FFHQ](https://github.com/NVlabs/ffhq-dataset) separately and update the path in your dataset config.

For non-SRM-specific datasets (such as ImageNet, RealEstate10K, etc.) refer to their respecrive documentations.

### Pretrained Models

Download pretrained checkpoints from the [SRM releases](https://github.com/Chrixtar/SRM/releases):

```bash
# Create checkpoints directory
mkdir -p checkpoints
cd checkpoints

# Download and extract checkpoints
wget https://github.com/Chrixtar/SRM/releases/download/v1.0/checkpoints.zip
unzip checkpoints.zip
```

## Troubleshooting

### Common Issues

**PyTorch Installation**

If you encounter PyTorch-related issues, make sure you have the correct PyTorch version installed. Visit [PyTorch's official installation guide](https://pytorch.org/get-started/locally/) for platform-specific instructions.

**CUDA Support**

For GPU training, ensure you have CUDA-compatible PyTorch:

```bash
# Example for CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Permission Errors**

If you encounter permission errors during installation, try:

```bash
pip install --user spatialreasoners
```

### Getting Help

If you're still having issues:

1. Check the [GitHub Issues](https://github.com/spatialreasoners/spatialreasoners/issues) for known problems
2. Open a new issue with your system details and error messages
<!-- 3. Join the [GitHub Discussions](https://github.com/spatialreasoners/spatialreasoners/discussions) for community support -->

## Next Steps

Now that you have Spatial Reasoners installed, continue with the [Quick Tour](quick-tour.md) to learn the basics, or directly check our [Template Project](https://github.com/spatialreasoners/spatialreasoners/tree/main/example_project).