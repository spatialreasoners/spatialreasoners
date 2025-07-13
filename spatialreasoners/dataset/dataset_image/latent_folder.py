import os
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float
from torchvision.datasets import DatasetFolder


class LatentFolder(DatasetFolder):
    # NOTE it will not check for validity of samples during initialization
    def __init__(
        self, 
        root: Path,
        horizontal_flip: bool
    ):
        super().__init__(root, loader=None)
        self.horizontal_flip = horizontal_flip

    @staticmethod
    def make_dataset(
        directory: str | Path,
        class_to_idx: dict[str, int],
        extensions: None = None,
        is_valid_file: None = None,
        allow_empty: bool = False,
    ) -> list[tuple[str, int]]:
        directory = os.path.expanduser(directory)
        if not class_to_idx:
            raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

        instances = []
        available_classes = set()
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            # NOTE considers only direct subdirectories and -files
            for dname in sorted(os.listdir(target_dir)):
                path = os.path.join(target_dir, dname)
                item = path, class_index
                instances.append(item)

                if target_class not in available_classes:
                    available_classes.add(target_class)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes and not allow_empty:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            raise FileNotFoundError(msg)

        return instances

    def __getitem__(self, index: int) -> tuple[
        Float[np.ndarray, "channel height width"],
        int
    ]:
        path, target = self.samples[index]
        filename = "latent"
        if self.horizontal_flip and torch.rand(1) < 0.5:
            filename = "latent_hflip"
        latent = np.load(os.path.join(path, filename + ".npy"))
        return latent, target
