import os
from pathlib import Path

import numpy as np
from pytorch_lightning import LightningModule
from torchvision.transforms.functional import hflip

from spatialreasoners.misc.nn_module_tools import freeze
from spatialreasoners.variable_mapper.autoencoder import Autoencoder, AutoencoderCfg, get_autoencoder
from spatialreasoners.type_extensions import BatchedExample


class LatentSaver(LightningModule):
    autoencoder: Autoencoder

    def __init__(
        self,
        autoencoder: AutoencoderCfg,
        output_path: Path,
        horizontal_flip: bool = True
    ) -> None:
        super(LatentSaver, self).__init__()
        self.autoencoder = get_autoencoder(autoencoder)
        freeze(self.autoencoder)
        self.output_path = output_path
        self.named_transforms = [(None, lambda x: x)]
        if horizontal_flip:
            self.named_transforms.append(("hflip", hflip))

    def test_step(self, batch: BatchedExample, batch_idx):
        # NOTE assumes that elements in batch are all not augmented!
        assert "path" in batch, "LatentSaver requires image paths"

        for name, transform in self.named_transforms:
            encoding = self.autoencoder.encode_deterministic(transform(batch["image"]))
            encoding_tensor = self.autoencoder.encoding_to_tensor(encoding)

            encoding_numpy = encoding_tensor.cpu().numpy()

            for i, path in enumerate(batch["path"]):
                save_path = os.path.join(
                    self.output_path, path, f"latent{'_' + name if name is not None else ''}.npy"
                )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.save(save_path, encoding_numpy[i])
