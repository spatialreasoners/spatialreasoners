from spatialreasoners.registry import Registry
from .class_embedding import ClassEmbedding, ClassEmbeddingCfg


_class_embedding_registry = Registry(ClassEmbedding, ClassEmbeddingCfg)


def get_class_embedding(
    embedding_cfg: ClassEmbeddingCfg,
    d_out: int,
    num_classes: int
) -> ClassEmbedding:
    return _class_embedding_registry.build(embedding_cfg, d_out, num_classes)


register_class_embedding = _class_embedding_registry.register


from .parameters import ClassEmbeddingParameters, ClassEmbeddingParametersCfg


__all__ = [
    "ClassEmbedding", "ClassEmbeddingCfg",
    "ClassEmbeddingParameters", "ClassEmbeddingParametersCfg",
    "get_class_embedding",
    "register_class_embedding"
]
