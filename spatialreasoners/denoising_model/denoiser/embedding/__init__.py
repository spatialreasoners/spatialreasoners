from spatialreasoners.registry import Registry
from .embedding import Embedding, EmbeddingCfg


_embedding_registry = Registry(Embedding, EmbeddingCfg)


def get_embedding(
    embedding_cfg: EmbeddingCfg,
    d_out: int
) -> Embedding:
    return _embedding_registry.build(embedding_cfg, d_out)


register_embedding = _embedding_registry.register


from .sinusodial import EmbeddingSinusodial, EmbeddingSinusodialCfg


__all__ = [
    "Embedding", "EmbeddingCfg",
    "EmbeddingSinusodial", "EmbeddingSinusodialCfg",
    "get_embedding",
    "register_embedding"
]
