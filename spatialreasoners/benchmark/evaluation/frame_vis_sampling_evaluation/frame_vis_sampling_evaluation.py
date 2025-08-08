import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree
from typing import Union
from typing import TypeVar


import numpy as np
from jaxtyping import Int, UInt8
from PIL import Image
from pytorch_lightning.loggers import Logger
from torch import Tensor

from ..sampling_evaluation import (
    SamplingEvaluation,
    SamplingEvaluationCfg,
)
from spatialreasoners.denoising_model import DenoisingModel
from spatialreasoners.misc.image_io import save_image
from spatialreasoners.type_extensions import BatchVariables, Stage
from spatialreasoners.variable_mapper import VariableMapper


# Evaluation Configuration and Implementation
@dataclass(frozen=True, kw_only=True)
class FrameVisSamplingEvaluationCfg(SamplingEvaluationCfg):
    image_format: str = "png"
    fps: int = 6
    video_format: str = "mp4"
    save_sampling_video: bool = True
    log_sampling_video: bool = False
    save_final_images: bool = True
    save_intermediate_images: bool = False
    log_final_images: bool = False

T = TypeVar("T", bound=FrameVisSamplingEvaluationCfg)

class FrameVisSamplingEvaluation(SamplingEvaluation[T], ABC):
    def __init__(
        self,
        cfg: T,
        variable_mapper: VariableMapper,
        tag: str,
        output_dir: Path,
        stage: Stage = "test",
    ):
        super().__init__(cfg, variable_mapper, tag, stage=stage, output_dir=output_dir)

        # Create output directories
        self.image_output_path = self.output_path / "images"
        self.video_output_path = self.output_path / "videos"

        # Create directories if they don't exist
        self.image_output_path.mkdir(parents=True, exist_ok=True)

        if self.cfg.save_sampling_video or self.cfg.log_sampling_video:
            self.video_output_path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def _get_visualization_images(
        self, batch_variables: BatchVariables
    ) -> UInt8[np.ndarray, "*batch 3 height width"]:
        """Here implement how each batch element is visualized into a single image"""
        pass

    @property
    def visualize_intermediate(self) -> bool:
        return (
            self.cfg.save_intermediate_samples
            or self.cfg.save_sampling_video
            or self.cfg.log_sampling_video
        )

    @property
    def visualize_final(self) -> bool:
        return (
            self.cfg.save_final_samples
            or self.cfg.log_final_images
            or self.cfg.save_sampling_video
            or self.cfg.log_sampling_video
        )

    def _get_in_dataset_sampler_name(
        self, in_dataset_index: Union[int, Int[Tensor, ""]], sampler_key: str
    ) -> str:
        if isinstance(in_dataset_index, Tensor):
            in_dataset_index = in_dataset_index.item()
        return f"{in_dataset_index:06d}/{sampler_key}"

    def _persist_intermediate_sample(
        self,
        model: DenoisingModel,
        batch_variables: BatchVariables,
        in_batch_index: Int[Tensor, "batch"],
        sampler_key: str,
        logger: Logger,
    ) -> None:
        images = self._get_visualization_images(batch_variables)

        # Save video frame if sampling_video is True
        if self.cfg.save_sampling_video or self.cfg.log_sampling_video:
            assert len(images) > 0, "No images found"
            for i, image in enumerate(images):
                video_name = self._get_in_dataset_sampler_name(
                    batch_variables.in_dataset_index[i], sampler_key
                )
                frame_index = batch_variables.num_steps[i].item()
                self._save_video_frame(image, frame_index, video_name)

        # Save image if sampling_video is False
        if self.cfg.save_intermediate_images:
            for i, image in enumerate(images):
                image_name = self._get_in_dataset_sampler_name(
                    batch_variables.in_dataset_index[i], sampler_key
                )
                save_image(image, self.image_output_path / f"{image_name}.{self.cfg.image_format}")

    def _persist_final_sample(
        self,
        model: DenoisingModel,
        batch_variables: BatchVariables,
        in_batch_index: Int[Tensor, "batch"],
        sampler_key: str,
        logger: Logger,
    ) -> None:
        assert (
            batch_variables.num_steps is not None
        ), "num_steps should always be in inference batch"
        images = self._get_visualization_images(batch_variables)
        image_captions = [f"img_{i:06d}" for i in batch_variables.in_dataset_index]

        # Save final image
        if self.cfg.save_final_images:
            for i, image in enumerate(images):
                image_name = self._get_in_dataset_sampler_name(
                    batch_variables.in_dataset_index[i], sampler_key
                )
                save_image(image, self.image_output_path / f"{image_name}.{self.cfg.image_format}")

        if self.cfg.log_final_images:
            channel_last_images = images.transpose(0, 2, 3, 1)

            logger.log_image(
                f"sample/sampler_{sampler_key}",
                [image for image in channel_last_images],
                caption=image_captions,
                step=model.step_tracker.get_step(),
            )
            
        if self.cfg.log_sampling_video:
            videos = []
            captions = [
                f"img_{i:06d}"
                for i in batch_variables.in_dataset_index
            ]
            
            for i, image in enumerate(images):
                video_name = self._get_in_dataset_sampler_name(batch_variables.in_dataset_index[i], sampler_key)
                frame_index = batch_variables.num_steps[i].item()
                self._save_video_frame(image, frame_index, video_name)
                
                video_path = self.video_output_path / video_name
                if not video_path.exists():
                    raise FileNotFoundError(f"Video path {video_path} does not exist")
                
                video_frame_paths = [
                    video_path / f"{(i+1):06d}.{self.cfg.image_format}" # Index from 1, as we save based on step
                    for i in range(len(os.listdir(video_path)))
                ]
            
                assert len(video_frame_paths) > 0, "No video frames found"

                video_frames = [
                    np.asarray(Image.open(frame_path).convert("RGB")).transpose(2, 0, 1) # HWC -> CHW
                    for frame_path in video_frame_paths
                ]
                
                videos.append(np.stack(video_frames, axis=0))
                
            logger.log_video(
                f"video/sampler_{sampler_key}", 
                videos, 
                step=model.step_tracker.get_step(),
                fps=[self.cfg.fps] * len(videos),
                format=[self.cfg.video_format] * len(videos),
                caption=captions
            )

        # Save video frame if sampling_video is True
        if self.cfg.save_sampling_video:
            for i, image in enumerate(images):
                video_name = self._get_in_dataset_sampler_name(
                    batch_variables.in_dataset_index[i], sampler_key
                )

                if not self.cfg.log_sampling_video:  # Otherwise already last frame saved
                    frame_index = batch_variables.num_steps[i].item()
                    self._save_video_frame(image, frame_index, video_name)

                self._merge_video_frames(video_name)

    def _save_video_frame(
        self, video_frame: UInt8[np.ndarray, "3 height width"], frame_index: int, video_name: str
    ) -> None:
        # Create directory for video frames
        video_dir = self.video_output_path / video_name
        video_dir.mkdir(parents=True, exist_ok=True)

        # Save the frame
        frame_path = video_dir / f"{frame_index:06d}.{self.cfg.image_format}"
        save_image(
            video_frame.transpose(1, 2, 0),
            frame_path,
        )

    def _merge_video_frames(self, video_name: str) -> None:
        video_dir = self.video_output_path / video_name
        output_video = (
            self.video_output_path / f"{video_name.replace('/', '_')}.{self.cfg.video_format}"
        )

        if not video_dir.exists():
            return

        # Check if there are any frames
        frame_files = list(video_dir.glob(f"*.{self.cfg.image_format}"))
        if not frame_files:
            rmtree(video_dir, ignore_errors=True)
            return

        # Ensure output directory exists
        output_video.parent.mkdir(parents=True, exist_ok=True)

        # Create video using ffmpeg - run from video output directory for cleaner paths
        try:
            # Run from the video output directory, use simple relative paths
            output_filename = f"{video_name.replace('/', '_')}.{self.cfg.video_format}"
            frame_pattern = f"{video_name}/*.{self.cfg.image_format}"

            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                str(self.cfg.fps),
                "-pattern_type",
                "glob",
                "-i",
                frame_pattern,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                output_filename,
            ]

            result = subprocess.run(
                command, cwd=self.video_output_path, capture_output=True, text=True
            )

            if result.returncode != 0:
                print(f"❌ Video creation failed for {video_name}: {result.stderr}")

        except Exception as e:
            print(f"❌ Video creation failed for {video_name}: {e}")
        finally:
            # Clean up frame directory
            rmtree(video_dir, ignore_errors=True)
