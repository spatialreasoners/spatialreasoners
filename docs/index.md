# 🌀Spatial Reasoners

**A Python package for spatial reasoning over continuous variables with generative denoising models.**

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.11+-brightgreen.svg)]()
[![PyPI version](https://img.shields.io/pypi/v/spatialreasoners.svg)](https://pypi.org/project/spatialreasoners/)

## Overview

<p align="center">
  <img src="assets/overview.png" alt="Spatial Reasoners Overview"/>
</p>

🌀Spatial Reasoners is a Python package for spatial reasoning over continuous variables with generative denoising models. Denoising generative models have become the de-facto standard for image generation, due to their effectiveness in sampling from complex, high-dimensional distributions. Recently, they have started being explored in the context of reasoning over multiple continuous variables.

Our package provides a comprehensive framework to facilitate research in this area, offering easy-to-use interfaces to control:

*   **Variable Mapping:** Seamlessly map variables from arbitrary data domains.
*   **Generative Model Paradigms:** Flexibly work with a wide range of denoising formulations.
*   **Samplers & Inference Strategies:** Implement and experiment with diverse samplers and inference techniques.

🌀Spatial Reasoners is a generalization of [Spatial Reasoning Models (SRMs)](https://geometric-rl.mpi-inf.mpg.de/srm/) to new domains, packaged as a reusable library for the research community.

## Key Features

- **🚀 One-line Training**: Get started with minimal setup using sensible defaults
- **🔧 Flexible Configuration**: Powerful config system with automatic merging of local and embedded configurations
- **📦 Modular Architecture**: Extensible design with pluggable components for datasets, models, and training strategies
- **🔬 Research-Ready**: Built-in benchmarks, evaluation protocols, and example projects
- **⚡ Production-Ready**: Lightning-based training infrastructure with distributed training support

## Architecture Overview

Spatial Reasoners is built with modularity and extensibility in mind:

```
spatialreasoners/
├── api/                  # High-level API
├── dataset/              # Data loading and processing
├── denoising_model/      # Model implementations
│   ├── denoiser/         # Denoiser architectures (UNet, DiT, MAR, etc.)
│   ├── flow/             # Flow variants (rectified, cosine, etc.)
│   └── tokenizer/        # Tokenizers of variables for the denoiser
├── training/             # Training infrastructure
├── variable_mapper/      # Variable mapping logic
├── benchmark/            # Evaluation framework
└── configs/              # Embedded default configs
```

## Research Applications

Spatial Reasoners has been used for research in:

- **Spatial reasoning tasks** (MNIST Sudoku, polygon counting)
- **Image generation** where there could be some spatial dependencies between regions of the image
- **Video generation** such as in [Diffusion Forcing](https://www.boyuan.space/diffusion-forcing/)

## Next Steps

- [Get started with installation](getting-started/installation.md)
- [Follow the quick tour](getting-started/quick-tour.md)
- [Explore the API reference](api.md)

## Citation

If you use Spatial Reasoners in your research, please cite:

```bibtex
@software{pogodzinski25spatialreasoners,
  title={Spatial Reasoners: A Framework for Spatial Reasoning with Generative Models},
  author={Pogodzinski, Bart and Wewer, Christopher and Lenssen, Jan Eric and Schiele, Bernt},
  year={2025},
  url={https://github.com/spatialreasoners/spatialreasoners}
}

@inproceedings{wewer25srm,
    title     = {Spatial Reasoning with Denoising Models},
    author    = {Wewer, Christopher and Pogodzinski, Bartlomiej and Schiele, Bernt and Lenssen, Jan Eric},
    booktitle = {International Conference on Machine Learning ({ICML})},
    year      = {2025},
}
```

## Support & Community

- **Issues**: [GitHub Issues](https://github.com/spatialreasoners/spatialreasoners/issues)
- **Discussions**: [GitHub Discussions](https://github.com/spatialreasoners/spatialreasoners/discussions)
- **Email**: bpogodzi@mpi-inf.mpg.de 