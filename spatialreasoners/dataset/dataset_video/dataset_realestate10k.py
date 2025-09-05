import json
import os
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal, Sequence

import torch
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

from spatialreasoners.type_extensions import ConditioningCfg, Stage, UnstructuredExample

from .. import register_dataset
from ..dataset import IterableDataset
from .dataset_video import DatasetVideo, DatasetVideoCfg
from .geometry_utils import CameraPose
from .video_transform import VideoTransform


@dataclass(frozen=True, kw_only=True)
class DatasetRealEstate10kCfg(DatasetVideoCfg):
    frame_skip_range: Sequence[int] = (10, 20)
    force_shuffle: bool = False
    
    @property
    def num_frames(self) -> int:
        return self.data_shape[0]
    
    @property
    def frame_shape(self) -> Sequence[int]:
        return self.data_shape[1:] # [C, H, W]
    

@register_dataset("realestate10k", DatasetRealEstate10kCfg)
class DatasetRealEstate10K(IterableDataset[DatasetRealEstate10kCfg]):
    num_classes = None
    cfg: DatasetRealEstate10kCfg
    stage: Stage
    # view_sampler: ViewSampler
    to_tensor: ToTensor
    chunk_paths: list[Path]
    near: float = 0.1
    far: float = 1000.0
    force_shuffle: bool = False

    def __init__(
        self,
        cfg: DatasetRealEstate10kCfg,
        conditioning_cfg: ConditioningCfg,
        stage: Stage,
        
    ) -> None:
        super().__init__(cfg, conditioning_cfg, stage)
        # self.view_sampler = view_sampler
        self.to_tensor = ToTensor()
        self.force_shuffle = self.cfg.force_shuffle
        assert self.cfg.frame_skip_range[0] > 0, f"Frame skip {self.cfg.frame_skip_range[0]} should be greater than 0"

        
        data_root_path = Path(cfg.root) / self.data_stage
        chunk_paths = sorted(
            [path for path in data_root_path.iterdir() if path.suffix == ".torch"]
        )
        self.chunk_paths = chunk_paths
        
        self.video_transform = VideoTransform(self.cfg.frame_shape[1:])

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]
    
    @property
    def should_shuffle(self) -> bool:
        return self.stage in ("train", "val") or self.force_shuffle

    def __iter__(self):
        yield_count = 0
        
        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.
        chunk_paths = self.chunk_paths
        if self.should_shuffle:
            chunk_paths = self.shuffle(chunk_paths)

        # When testing, the data loaders alternate chunks.
        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            chunk_paths = [
                chunk_path
                for chunk_index, chunk_path in enumerate(chunk_paths)
                if chunk_index % worker_info.num_workers == worker_info.id
            ]

        for chunk_index, chunk_path in enumerate(chunk_paths):
            # Load the chunk.
            chunk = torch.load(chunk_path)

            if self.should_shuffle:
                chunk = self.shuffle(chunk)

            for example_index, example in enumerate(chunk):
                # extrinsics, intrinsics = self.convert_poses(example["cameras"])
                video_name = example["key"]
                # num_views = example["cameras"].shape[0]
                # video_length = example["images"].shape[0]
                video_length = example['cameras'].shape[0]
                
                frame_skip = self.get_frame_skip(video_length)
                
                
                
                frame_indices_unshifted = torch.arange(0, self.cfg.num_frames) * frame_skip
                max_frame_shift = video_length - frame_indices_unshifted[-1] - 1
                
                if max_frame_shift <= 0:
                    continue # Skip if the video is too short
                
                frame_shift = max_frame_shift #torch.randint(0, max_frame_shift + 1, (1,)).item()
                frame_indices = frame_indices_unshifted + frame_shift
                
                if frame_indices[-1] >= video_length:
                    continue # Skip if the video is too short
                
                images_data = [
                    example["images"][frame_index]
                    for frame_index in frame_indices
                ]
                images = self.convert_images(images_data) # [T, 3, H, W]
                cameras = example["cameras"][frame_indices]
                
                poses = CameraPose.from_full_poses(cameras.unsqueeze(0))
                poses.normalize_by_first() 
                
                rays = poses.rays(resolution=self.cfg.frame_shape[1])
                pos_encodings, _ = rays.to_pos_encoding() 
                pos_encodings = rearrange(pos_encodings, "1 t h w c -> t c h w") # [T, 180, H, W]
                
                
                transformed_images = self.video_transform(images)
                
                yield UnstructuredExample(
                    in_dataset_index=chunk_index * len(chunk) + example_index,
                    path=video_name,
                    z_t=transformed_images,
                    fixed_conditioning_fields={
                        "pos_encodings": pos_encodings,  
                    },
                )
                
                yield_count += 1
                if self.cfg.subset_size is not None and yield_count >= self.cfg.subset_size:
                    return
                
                
                
                
                

                # try:
                #     instance_indices = self.view_sampler.sample(video_name, num_views)
                # except ValueError:
                #     # Skip because the example doesn't have enough frames.
                #     continue
                
                # for instance_index in instance_indices:
                #     sample = {"scene": video_name}

                #     # Resize the world to make the baseline 1.
                #     # context_extrinsics = extrinsics[instance_index.context]
                #     # if context_extrinsics.shape[0] == 2 and self.cfg.make_baseline_1:
                #     #     a, b = context_extrinsics[:, :3, 3]
                #     #     scale = (a - b).norm()
                #     #     if scale < self.cfg.baseline_epsilon:
                #     #         print(
                #     #             f"Skipped {video_name} because of insufficient baseline "
                #     #             f"{scale:.6f}"
                #     #         )
                #     #         continue
                #     #     extrinsics[:, :3, 3] /= scale
                #     # else:
                #     #     scale = 1

                #     frame_indices = []
                #     view_to_index = {}
                #     cum_frames = 0
                #     for view_type, view_indices in asdict(instance_index).items():
                #         if view_indices is None:
                #             continue
                #         frame_indices.append(view_indices)
                #         end = cum_frames + len(view_indices)
                #         view_to_index[view_type] = tuple(range(cum_frames, end))
                #         cum_frames = end

                #     frame_indices = torch.cat(frame_indices)

                #     # Load the images.
                #     images = [
                #         example["images"][index.item()] for index in frame_indices
                #     ]
                #     images = self.convert_images(images)

                #     # Skip the example if the images don't have the right shape.
                #     image_invalid = images.shape[1:] != (3, 360, 640)
                #     if image_invalid:
                #         print(
                #             f"Skipped bad example {video_name}. "
                #             f"Shape was {images.shape}."
                #         )
                #         continue
                    
                #     sample["views"] = {
                #         # "extrinsics": extrinsics[frame_indices],
                #         # "intrinsics": intrinsics[frame_indices],
                #         "image": images,
                #         # "near": self.get_bound("near", len(frame_indices)) / scale,
                #         # "far": self.get_bound("far", len(frame_indices)) / scale,
                #         "index": frame_indices
                #     }
                #     sample["index"] = view_to_index

                    # # TODO adapt all shims for new format
                    # if self.stage == "train" and self.cfg.augment:
                    #     sample = apply_augmentation_shim(sample)
                    # yield apply_crop_shim(sample, tuple(self.cfg.image_shape))

    # def convert_poses(
    #     self,
    #     poses: Float[Tensor, "batch 18"],
    # ) -> tuple[
    #     Float[Tensor, "batch 4 4"],  # extrinsics
    #     Float[Tensor, "batch 3 3"],  # intrinsics
    # ]:
    #     b, _ = poses.shape

    #     # Convert the intrinsics to a 3x3 normalized K matrix.
    #     intrinsics = torch.eye(3, dtype=torch.float32)
    #     intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
    #     fx, fy, cx, cy = poses[:, :4].T
    #     intrinsics[:, 0, 0] = fx
    #     intrinsics[:, 1, 1] = fy
    #     intrinsics[:, 0, 2] = cx
    #     intrinsics[:, 1, 2] = cy

    #     # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
    #     w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
    #     w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
    #     return w2c.inverse(), intrinsics
    
    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        # if self.cfg.overfit_to_scene is not None:
        #     return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    @cached_property
    def data_index(self) -> dict[str, Path]:
        merged_index = {}
        data_stages = [self.data_stage]
        if self.cfg.overfit_to_scene is not None:
            data_stages = ("test", "train")
        for data_stage in data_stages:
            for root in self.cfg.roots:
                # Load the root's index.
                with (root / data_stage / "index.json").open("r") as f:
                    index = json.load(f)
                index = {k: Path(root / data_stage / v) for k, v in index.items()}

                # The constituent datasets should have unique keys.
                assert not (set(merged_index.keys()) & set(index.keys()))

                # Merge the root's index into the main index.
                merged_index = {**merged_index, **index}
        return merged_index

    def _num_available(self) -> int:
        return len(self.data_index.keys())
    
    
    def get_frame_skip(self, video_length: int) -> int:
        num_frames = self.cfg.num_frames
        # Check how many frames can we skip to still fit within the video length
        # so we get the max_frame_skip
        
        max_frame_skip = (video_length - 1) // num_frames
        max_frame_skip = max(max_frame_skip, self.cfg.frame_skip_range[0])
        max_frame_skip = min(max_frame_skip, self.cfg.frame_skip_range[1])
        
        return torch.randint(0, max_frame_skip + 1, (1,)).item()
        

