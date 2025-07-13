from collections.abc import Sequence

from spatialreasoners.registry import Registry
from .pos_embedding import PosEmbedding, PosEmbeddingCfg


_pos_embedding_registry = Registry(PosEmbedding, PosEmbeddingCfg)


def get_pos_embedding(
    pos_embedding_cfg: PosEmbeddingCfg,
    dim: int,
    grid_size: Sequence[int]
) -> PosEmbedding:
    return _pos_embedding_registry.build(pos_embedding_cfg, dim, grid_size)


register_pos_embedding = _pos_embedding_registry.register


from .frequency import FrequencyPosEmbedding, FrequencyPosEmbeddingCfg
from .rotary import RotaryPosEmbedding, RotaryPosEmbeddingCfg


__all__ = [
    "PosEmbedding", "PosEmbeddingCfg",
    "FrequencyPosEmbedding", "FrequencyPosEmbeddingCfg",
    "RotaryPosEmbedding", "RotaryPosEmbeddingCfg",
    "get_pos_embedding",
    "register_pos_embedding"
]
