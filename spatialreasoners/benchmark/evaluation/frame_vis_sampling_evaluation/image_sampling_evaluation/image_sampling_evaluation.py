from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, Union

import numpy as np
from jaxtyping import UInt8

from spatialreasoners.misc.image_io import prep_images
from spatialreasoners.type_extensions import BatchVariables, Stage
from spatialreasoners.variable_mapper import VariableMapper
from spatialreasoners.visualization.color_map import apply_color_map_to_image
from spatialreasoners.visualization.layout import add_border, hcat

from ... import register_evaluation
from ..frame_vis_sampling_evaluation import (
    FrameVisSamplingEvaluation,
    FrameVisSamplingEvaluationCfg,
)


@dataclass(kw_only=True, frozen=True)
class ImageSamplingEvaluationCfg(FrameVisSamplingEvaluationCfg):
    visualize_time: bool = True
    normalize_sigma_per_image: bool = False  # If False, max_sigma_threshold must be set
    max_sigma_threshold: float | None = 2.0  # Max sigma value for normalization


T = TypeVar("T", bound=ImageSamplingEvaluationCfg)


@register_evaluation("image_sampling", ImageSamplingEvaluationCfg)
class ImageSamplingEvaluation(FrameVisSamplingEvaluation[T]):

    def __init__(
        self,
        cfg: T,
        variable_mapper: VariableMapper,
        tag: str,
        output_dir: Path,
        stage: Stage = "test",
    ) -> None:
        super().__init__(cfg, variable_mapper, tag, stage=stage, output_dir=output_dir)

        if self.cfg.save_sampling_video:
            if not self.cfg.normalize_sigma_per_image:
                assert (
                    self.cfg.max_sigma_threshold is not None
                ), "max_sigma_threshold must be set if normalize_sigma_per_image is False"
                assert (
                    self.cfg.max_sigma_threshold > 0
                ), "max_sigma_threshold must be greater than 0"

    def _get_visualization_images(
        self,
        batch_variables: BatchVariables,
    ) -> Union[
        UInt8[np.ndarray, "*batch 3 height width"], UInt8[np.ndarray, "*batch 4 height width"]
    ]:
        """Get the prepared image from the batch variables."""
        unstructured = self.variable_mapper.variables_to_unstructured(batch_variables)

        """NOTE expects images to be in [-1, 1]"""
        image = (unstructured["z_t"] + 1) / 2
        if image.size(1) == 4:
            image = image[:, :3] * image[:, 3:]

        if self.cfg.visualize_time:
            assert "t" in unstructured, "t should always be in inference batch"
            image = hcat(image, (1 - unstructured["t"]).expand_as(image))

        if "sigma" in unstructured:
            sigma = unstructured["sigma"].squeeze(1)
            mask = sigma == 0  # mask out regions with no uncertainty

            sigma_min = sigma.min() if self.cfg.normalize_sigma_per_image else 0
            sigma_max = (
                sigma.max() if self.cfg.normalize_sigma_per_image else self.cfg.max_sigma_threshold
            )

            sigma = (sigma - sigma_min) / (sigma_max - sigma_min)
            sigma_color = apply_color_map_to_image(sigma)
            sigma_color.masked_fill_(mask.unsqueeze(1), 1)
            image = hcat(image.expand(-1, 3, -1, -1), sigma_color)

        if "x_pred" in unstructured:
            x = (unstructured["x_pred"] + 1) / 2
            if x.size(1) == 4:
                x = x[:, :3] * x[:, 3:]
            image = hcat(image, x.expand(-1, image.shape[1], -1, -1))

        if any(k in unstructured for k in ("t", "sigma", "x_pred")):
            image = add_border(image)

        return prep_images(image, channel_last=False)
