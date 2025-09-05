from spatialreasoners.registry import Registry
from spatialreasoners.variable_mapper import VariableMapper

from .inference_sampler import InferenceSampler, InferenceSamplerCfg

_sampler_registry = Registry(InferenceSampler, InferenceSamplerCfg)


def get_inference_sampler(
    cfg: InferenceSamplerCfg,
    variable_mapper: VariableMapper,
) -> InferenceSampler:
    return _sampler_registry.build(cfg, variable_mapper)


register_inference_sampler = _sampler_registry.register


from .scheduled_inference_sampler import (
    ScheduledInferenceSampler,
    ScheduledInferenceSamplerCfg,
)
from .type_extensions import (
    FinalInferenceBatchSample,
    InferenceBatchSample,
    IntermediateInferenceBatchSample,
)

from .sampling_schedule import __all__ as sampling_schedule_all

__all__ = sampling_schedule_all + [
    "InferenceSampler", "InferenceSamplerCfg",
    "ScheduledInferenceSampler", "ScheduledInferenceSamplerCfg",
    "get_inference_sampler",
    "register_inference_sampler",
    "IntermediateInferenceBatchSample", "FinalInferenceBatchSample", "InferenceBatchSample",
]
