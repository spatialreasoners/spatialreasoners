from jaxtyping import Float
from torch import Tensor

from spatialreasoners.registry import Registry
from spatialreasoners.variable_mapper import VariableMapper

from .sampling_schedule import SamplingSchedule, SamplingScheduleCfg

_sampling_schedule_registry = Registry(SamplingSchedule, SamplingScheduleCfg)


def get_sampling_schedule(
    cfg: SamplingScheduleCfg,
    max_steps: int,
    batch_size: int,
    device: int | str,
    variable_mapper: VariableMapper,
    mask: Float[Tensor, "batch num_variables"] | None = None,
) -> SamplingSchedule:
    return _sampling_schedule_registry.build(
        cfg,
        max_steps,
        batch_size,
        device,
        variable_mapper,
        mask,
    )


get_sampling_schedule_class = _sampling_schedule_registry.get
register_sampling_schedule = _sampling_schedule_registry.register


from .adaptive import (
    AdaptiveInverse,
    AdaptiveInverseCfg,
    AdaptiveSoftmax,
    AdaptiveSoftmaxCfg,
)
from .certainty_sequential import (
    AdaptiveSequential,
    AdaptiveSequentialCfg,
    GraphSequential,
    GraphSequentialCfg,
)
from .fixed import Fixed, FixedCfg
from .mask_git import MaskGIT, MaskGITCfg
from .raster import Raster, RasterCfg

__all__ = [
    "AdaptiveInverse", "AdaptiveInverseCfg",
    "AdaptiveSoftmax", "AdaptiveSoftmaxCfg",
    "AdaptiveSequential", "AdaptiveSequentialCfg",
    "GraphSequential", "GraphSequentialCfg",
    "Fixed", "FixedCfg",
    "MaskGIT", "MaskGITCfg",
    "Raster", "RasterCfg",
    "get_sampling_schedule", "get_sampling_schedule_class",
    "register_sampling_schedule"
]
