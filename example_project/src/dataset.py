"""
Spiral Dataset Implementation for SpatialReasoners
"""

from dataclasses import dataclass
import numpy as np
import torch

import spatialreasoners as sr


def generate_spiral_data(n_points=1000, noise=0.5, num_loops=2):
    """Generate spiral dataset points."""
    r = np.sqrt(np.random.rand(n_points))  # More uniform area coverage
    t = r * 2 * np.pi * num_loops
    dx = noise * np.random.randn(n_points)
    dy = noise * np.random.randn(n_points)
    x = r * np.sin(t) + dx
    y = r * np.cos(t) + dy
    
    return np.stack([x, y, r], axis=1)


@dataclass(frozen=True, kw_only=True)
class SpiralDatasetCfg(sr.dataset.DatasetCfg):
    noise: float = 0.001
    num_spiral_loops: int = 4
    subset_size: int = 1_000_000


@sr.dataset.register_dataset("spiral", SpiralDatasetCfg)
class SpiralDataset(sr.dataset.IndexedDataset[SpiralDatasetCfg]):
    def __init__(self, cfg: SpiralDatasetCfg, conditioning_cfg: sr.dataset.ConditioningCfg, stage: sr.dataset.Stage):
        super().__init__(cfg, conditioning_cfg, stage)
        self.num_points = cfg.subset_size
        self.num_loops = cfg.num_spiral_loops
        self.noise = cfg.noise

        spiral_data = generate_spiral_data(n_points=self.num_points, noise=self.noise, num_loops=self.num_loops)
        self.datapoints = torch.from_numpy(spiral_data).float()  # Convert to torch tensor with float32
        
        # Debug: Print statistics of original training data
        if self.num_points > 0:
            colors = self.datapoints[:, 2]  # Third column is the color (radius)
            print(f"Training data color range - min: {colors.min():.3f}, max: {colors.max():.3f}, mean: {colors.mean():.3f}")
            print(f"Sample training points: {self.datapoints[:3].tolist()}")
        
    def __len__(self):
        return self.num_points
    
    def _num_available(self) -> int:
        """Return the number of available samples in the dataset."""
        return self.num_points
    
    @property
    def num_classes(self) -> int:
        """Number of classes in the dataset. Spiral data is unconditional."""
        return 0
    
    def __getitem__(self, idx):
        return sr.type_extensions.UnstructuredExample(
            in_dataset_index=idx,
            z_t=self.datapoints[idx],
        )