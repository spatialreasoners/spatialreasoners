from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import torch
from jaxtyping import Bool, Float, Int64
from torch import Tensor
from torch.nn import Module, Parameter

from spatialreasoners.env import DEBUG
from spatialreasoners.misc.nn_module_tools import freeze

from ..tokenizer import Tokenizer
from .class_embedding import (
    ClassEmbeddingCfg,
    ClassEmbeddingParametersCfg,
    get_class_embedding,
)


@dataclass(frozen=True, kw_only=True)
class DenoiserFreezeCfg:
    time_embedding: bool = False
    class_embedding: bool = False


F = TypeVar("F", bound=DenoiserFreezeCfg)


@dataclass(frozen=True, kw_only=True)
class DenoiserCfg:
    class_embedding: ClassEmbeddingCfg = field(default_factory=ClassEmbeddingParametersCfg)
    freeze: F = field(default_factory=DenoiserFreezeCfg)


T_CFG = TypeVar("T", bound=DenoiserCfg)


T_INPUT = TypeVar("T_INPUT")
T_OUTPUT = TypeVar("T_OUTPUT")


class Denoiser(Module, ABC, Generic[T_CFG, T_INPUT, T_OUTPUT]):
    """ Base class for denoising models
    """
    cfg: T_CFG

    def __init__(
        self,
        cfg: T_CFG,
        tokenizer: Tokenizer,
        num_classes: int | None = None
    ) -> None:
        super(Denoiser, self).__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        
        self.labels = num_classes is not None
        if self.labels:
            assert cfg.class_embedding is not None
            self.class_embedding = get_class_embedding(cfg.class_embedding, self.d_conditioning, num_classes)

    @property
    def d_conditioning(self) -> int:
        raise NotImplementedError(f"d_conditioning must be implemented in the {self.__class__.__name__}")
    
    @abstractmethod
    def freeze_time_embedding(self) -> None:
        pass
    
    @property
    def d_time_embedding(self) -> int:
        """
        Dimension of the time embedding
        """
        return self.d_conditioning

    def freeze(self) -> None:
        if self.cfg.freeze.time_embedding:
            self.freeze_time_embedding()
            # freeze(self.t_emb, eval=False)
        if self.cfg.freeze.class_embedding:
            freeze(self.class_embedding, eval=False)

    def on_sampling_start(self) -> None:
        """ Hook for start of sampling
        """
        return

    def on_sampling_end(self) -> None:
        """ Hook for end of sampling
        """
        return

    def embed_conditioning(
        self,
        label: Int64[Tensor, "batch"] | None = None,
        label_mask: Bool[Tensor, "batch"] | None = None
    ) -> Float[Tensor, "#batch d_c"] | None:
        emb = None
        if self.labels:
            assert label is not None
            emb = self.class_embedding.forward(label, label_mask)
        return emb

    @abstractmethod
    def forward(
        self, 
        model_inputs: T_INPUT,
        sample: bool = False
    ) -> T_OUTPUT:
        pass
    
    @torch.compile(disable=DEBUG)
    def forward_compiled(
        self, 
        model_inputs: T_INPUT,
        sample: bool = False,
    ) -> T_OUTPUT:
        return self.forward(model_inputs=model_inputs, sample=sample)


    def init_weights(self) -> None:
        if self.labels:
            self.class_embedding.init_weights()

    def get_weight_decay_parameter_groups(self) -> tuple[list[Parameter], list[Parameter]]:
        return list(self.parameters()), []
