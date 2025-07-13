from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence, TypeVar

import torch
from torch import nn

from spatialreasoners.denoising_model.tokenizer import Tokenizer
from spatialreasoners.misc.nn_module_tools import freeze

from ..denoiser import Denoiser, DenoiserCfg
from .modules.embeddings import RandomDropoutCondEmbedding, StochasticTimeEmbedding
from .type_extensions import UViT3DInputs, UViT3DOutputs


@dataclass(frozen=True, kw_only=True)
class BaseBackboneConfig(DenoiserCfg):
    channels: List[int]
    emb_channels: int
    patch_size: int
    block_types: List[str]
    block_dropouts: List[float]
    use_fourier_noise_embedding: bool = True
    external_cond_dropout: float = 0.0
    use_causal_mask: bool = True


T = TypeVar("T", bound=BaseBackboneConfig)

class BaseBackbone(Denoiser[T, UViT3DInputs, UViT3DOutputs]):
    def __init__(
        self,
        cfg: T,
        tokenizer: Tokenizer,
        num_classes: None = None
    ):
        super().__init__(cfg, tokenizer, num_classes)
        self.use_causal_mask = cfg.use_causal_mask
        assert num_classes is None, "num_classes is not supported for U-ViT3D"

        self.noise_level_pos_embedding = StochasticTimeEmbedding(
            dim=self.noise_level_dim,
            time_embed_dim=self.noise_level_emb_dim,
            use_fourier=self.cfg.use_fourier_noise_embedding,
        )
        self.external_cond_embedding = self._build_external_cond_embedding()
        
    def freeze_time_embedding(self) -> None:
        freeze(self.noise_level_pos_embedding, eval=False)
        
    @property
    def model_input_shape(self) -> Sequence[int]:
        return self.tokenizer.model_input_shape

    def _build_external_cond_embedding(self) -> Optional[nn.Module]:
        return (
            RandomDropoutCondEmbedding(
                self.external_cond_dim,
                self.external_cond_emb_dim,
                dropout_prob=self.cfg.external_cond_dropout,
            )
            if self.external_cond_dim
            else None
        )
        
    @property 
    def d_conditioning(self) -> int:
        return self.external_cond_dim

    @property
    def noise_level_dim(self):
        return max(self.noise_level_emb_dim // 4, 32)

    @property
    @abstractmethod
    def noise_level_emb_dim(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def external_cond_emb_dim(self):
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        model_inputs: UViT3DInputs,
        sample: bool = False,
    ):
        raise NotImplementedError
