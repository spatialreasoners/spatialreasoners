from spatialreasoners.registry import Registry
from spatialreasoners.variable_mapper import VariableMapper

from .tokenizer import Tokenizer, TokenizerCfg

_tokenizer_registry = Registry(Tokenizer, TokenizerCfg)


def get_tokenizer(
    cfg: TokenizerCfg,
    variable_mapper: VariableMapper,
    predict_uncertainty: bool = False,
    predict_variance: bool = False,
) -> Tokenizer:
    return _tokenizer_registry.build(
        cfg, variable_mapper,
        predict_uncertainty=predict_uncertainty,
        predict_variance=predict_variance
    )


register_tokenizer = _tokenizer_registry.register

from .dit_tokenizer import (
    DiTTokenizer,
    DiTTokenizerCfg,
    ImageDiTTokenizer,
    ImageDiTTokenizerCfg
)
from .u_vit_tokenizer.pose_video_u_vit_tokenizer import (
    PoseVideoUViTTokenizer,
    PoseVideoUViTTokenizerCfg,
)
from .unet_tokenizer.unet_tokenizer import UNetTokenizer, UNetTokenizerCfg

__all__ = [
    "Tokenizer", "TokenizerCfg",
    "DiTTokenizer", "DiTTokenizerCfg",
    "ImageDiTTokenizer", "ImageDiTTokenizerCfg",
    "get_tokenizer",
    "register_tokenizer",
    "PoseVideoUViTTokenizer",
    "PoseVideoUViTTokenizerCfg",
    "UNetTokenizer",
    "UNetTokenizerCfg",
]
