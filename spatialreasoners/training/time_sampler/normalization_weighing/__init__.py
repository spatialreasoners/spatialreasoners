from typing import Callable

from spatialreasoners.registry import Registry
from jaxtyping import Bool, Float
from torch import Tensor, device

from .normalization_weighing import NormalizationWeighing, NormalizationWeighingCfg

_normalization_weighing_registry = Registry(NormalizationWeighing, NormalizationWeighingCfg)


def get_normalization_weighing(
    cfg: NormalizationWeighingCfg, 
    get_time_func: Callable[
            [int, int, str], 
            tuple[Float[Tensor, "batch sample num_variables"], Bool[Tensor, "batch #sample num_variables"] | None]
        ],
        device: device | str = "cpu"
) -> NormalizationWeighing:
    return _normalization_weighing_registry.build(cfg, get_time_func, device)


register_normalization_weighing = _normalization_weighing_registry.register


from .histogram_pdf_weighing import InverseHistogramPdfWeighing, InverseHistogramPdfWeighingCfg
from .trainable import TrainableWeighting, TrainableWeightingCfg

__all__ = [
    "InverseHistogramPdfWeighing", "InverseHistogramPdfWeighingCfg",
    "TrainableWeighting", "TrainableWeightingCfg",
    "register_normalization_weighing"
]