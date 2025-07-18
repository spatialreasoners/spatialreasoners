import os
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree
from typing import TypeVar, Union

import numpy as np
import torch
from jaxtyping import Float, Int, UInt8
from PIL import Image
from pytorch_lightning.loggers import Logger
from torch import Tensor

from spatialreasoners.denoising_model import DenoisingModel
from spatialreasoners.denoising_model.denoising_model import DenoisingModel
from spatialreasoners.misc.image_io import load_image, prep_images, save_image
from spatialreasoners.type_extensions import BatchVariables, Stage
from spatialreasoners.variable_mapper import VariableMapper
from spatialreasoners.visualization.color_map import apply_color_map_to_image
from spatialreasoners.visualization.layout import add_border, hcat, vcat

from .. import register_evaluation
from ..frame_vis_sampling_evaluation import FrameVisSamplingEvaluation, FrameVisSamplingEvaluationCfg


@dataclass(frozen=True, kw_only=True)
class VideoPoseSamplingEvaluationCfg(FrameVisSamplingEvaluationCfg):
    visualize_time: bool = True
    normalize_sigma_per_image: bool = False
    max_sigma_threshold: float | None = 2.0 # Max sigma value for normalization


@register_evaluation("video_pose_sampling", VideoPoseSamplingEvaluationCfg)
class VideoPoseSamplingEvaluation(FrameVisSamplingEvaluation[VideoPoseSamplingEvaluationCfg]):
    def __init__(
        self, 
        cfg: VideoPoseSamplingEvaluationCfg, 
        variable_mapper: VariableMapper,
        tag: str,
        output_dir: Path,
        stage: Stage = "test",
    ) -> None:
        super().__init__(cfg, variable_mapper, tag, stage=stage, output_dir=output_dir)
        
        if self.cfg.save_sampling_video or self.cfg.log_sampling_video:
            if not self.cfg.normalize_sigma_per_image:
                assert self.cfg.max_sigma_threshold is not None, "max_sigma_threshold must be set if normalize_sigma_per_image is False"
                assert self.cfg.max_sigma_threshold > 0, "max_sigma_threshold must be greater than 0"
            
    def _get_visualization_images(
        self,
        batch_variables: BatchVariables,
    ) -> Union[
        UInt8[np.ndarray, "*batch 3 merged_height merged_width"],
        UInt8[np.ndarray, "*batch 4 merged_height merged_width"]
    ]:
        """Get the prepared image from the batch variables."""
        unstructured = self.variable_mapper.variables_to_unstructured(batch_variables)
        frames = (unstructured["z_t"] + 1) / 2
        
        if frames.size(2) == 4: # alpha channel
            frames = frames[:, :, :3] * frames[:, :, 3:]
        
        images = hcat(*[frame for frame in frames.unbind(1)])
            
        if self.cfg.visualize_time:
            assert "t" in unstructured, "t should always be in inference batch"
            t = (1-unstructured["t"]).expand_as(frames)
            
            t_cat = hcat(*[t_frame for t_frame in t.unbind(1)])
            
            images = vcat(images, t_cat)
            
        if self.cfg.visualize_x:
            assert batch_variables.x_pred is not None, "x should be in inference batch"
            assert "x_pred" in unstructured, "x should be in inference batch"
            x = (unstructured["x_pred"] + 1) / 2
            x_cat = hcat(*[x_frame for x_frame in x.unbind(1)])
            images = vcat(images, x_cat)
            
        if any(k in unstructured for k in ("t", "x_pred")):
            images = add_border(images)
            
        return prep_images(images, channel_last=False)