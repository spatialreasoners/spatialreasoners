"""
Spiral Evaluation Implementation for SpatialReasoners
"""

from dataclasses import dataclass
from functools import cache

import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float, UInt8
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from torch import Tensor

import spatialreasoners as sr
from spatialreasoners.benchmark.evaluation import register_evaluation
from spatialreasoners.benchmark.evaluation.frame_vis_sampling_evaluation import (
    FrameVisSamplingEvaluation,
    FrameVisSamplingEvaluationCfg,
)
from spatialreasoners.type_extensions import BatchVariables


@dataclass(frozen=True, kw_only=True)
class SpiralSamplingEvaluationCfg(FrameVisSamplingEvaluationCfg):
    calculate_metrics: bool = True
    num_spiral_loops: int = 4  # Should match dataset config


@register_evaluation("spiral_sampling", SpiralSamplingEvaluationCfg)
class SpiralSamplingEvaluation(FrameVisSamplingEvaluation[SpiralSamplingEvaluationCfg]):

    @staticmethod
    def get_rgb_array_from_figure(fig):
        """Convert a Matplotlib figure to a NumPy array in (3, H, W) format."""
        canvas = FigureCanvas(fig)
        canvas.draw()
        w, h = fig.canvas.get_width_height()

        # Use buffer_rgba() for newer matplotlib versions
        try:
            buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
            buf = buf.reshape(h, w, 4)[:, :, :3]  # Remove alpha channel
        except AttributeError:
            # Fallback for older versions
            buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(h, w, 3)

        # Transpose to (3, H, W)
        return buf.transpose(2, 0, 1)

    @cache
    def get_spiral_points(self, num_points: int = 1000) -> Float[Tensor, "num_points 3"]:
        # Mathematical spiral matching training data generation (much faster than random sampling)
        # Use same relationship as generate_spiral_data: t = r * 2 * pi * num_loops
        # Match the sqrt distribution used in training data for proper density
        uniform_r = np.linspace(0, 1, num_points)
        r = np.sqrt(uniform_r)  # Same as np.sqrt(np.random.rand()) but deterministic
        t = r * 2 * np.pi * self.cfg.num_spiral_loops  # Use same formula as training data
        x = r * np.sin(t)
        y = r * np.cos(t)
        return np.stack([x, y, r], axis=-1)

    def get_base_figure(self):
        """Create base figure for spiral visualization."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))  # Smaller figure

        points = self.get_spiral_points(num_points=2000)

        # Color the spiral by radius using the same colormap as data points
        ax.scatter(
            points[:, 0],
            points[:, 1],
            c=points[:, 2],
            cmap="Spectral",
            s=4,
            alpha=0.9,
            linewidths=0.1,
            vmin=0,
            vmax=1,
        )

        ax.set_title("Spiral")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)

        return fig, ax

    def _get_visualization_images(
        self, batch_variables: BatchVariables
    ) -> UInt8[np.ndarray, "*batch 3 height width"]:
        """Generate visualization images for the batch (optimized for speed)."""

        unstructured = self.variable_mapper.variables_to_unstructured(batch_variables)
        batch_size = unstructured["z_t"].shape[0]

        # For batch processing, we'll create separate figures to avoid conflicts
        figures = []

        for batch_index in range(batch_size):
            fig, ax = self.get_base_figure()

            # Add the data points
            data_points = unstructured["z_t"][batch_index].cpu().numpy()

            # Ensure data_points has the right shape: should be [3] for (x, y, color)
            if data_points.ndim == 1:
                if len(data_points) == 3:
                    data_points = data_points.reshape(1, 3)
                else:
                    data_points = data_points.reshape(-1, 3)

            ax.scatter(
                data_points[:, 0],
                data_points[:, 1],
                c=data_points[:, 2],
                cmap="Spectral",
                s=30,
                alpha=0.8,
                vmin=0,
                vmax=1,
            )  # Ensure colormap range

            rgb_array = self.get_rgb_array_from_figure(fig)
            figures.append(rgb_array)

            # Clean up to save memory
            plt.close(fig)

        return np.stack(figures)

    def _get_metrics(
        self,
        batch_variables: BatchVariables,
        labels: Tensor | None,
    ) -> dict[str, Float[Tensor, ""]] | None:
        """Here you can define your own metrics.
        For example, you can calculate the distance of the sampled points to the spiral.
        This implementation is simple and slow, but it's a good example of how to calculate metrics.
        """
        # Calculate the distance of the sampled points to the spiral

        # We can convert to the unstructured format so that we have shape (batch_size, 3)
        unstructured = self.variable_mapper.variables_to_unstructured(batch_variables)

        predicted_points = unstructured["z_t"]

        # Define the spiral
        spiral_points = self.get_spiral_points(num_points=1000)
        spiral_points_tensor = torch.from_numpy(spiral_points).float()
        spiral_points_tensor = spiral_points_tensor.to(predicted_points.device)

        # Calculate the minimum distance of each sampled point to the spiral
        distances = torch.cdist(predicted_points, spiral_points_tensor)
        min_distances = distances.min(dim=-1).values

        # Return the average distance
        return {"distance": min_distances.mean()}
