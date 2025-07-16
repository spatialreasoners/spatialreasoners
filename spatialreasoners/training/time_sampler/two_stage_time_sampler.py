from abc import ABC
from dataclasses import dataclass, field
from typing import TypeVar

from spatialreasoners.variable_mapper import VariableMapper

from .scalar_time_sampler import (
    ScalarTimeSampler,
    ScalarTimeSamplerCfg,
    UniformCfg,
    get_scalar_time_sampler,
)
from .time_sampler import TimeSampler, TimeSamplerCfg


@dataclass
class TwoStageTimeSamplerCfg(TimeSamplerCfg):
    scalar_time_sampler: ScalarTimeSamplerCfg = field(default_factory=UniformCfg)


T = TypeVar("T", bound=TwoStageTimeSamplerCfg)


class TwoStageTimeSampler(TimeSampler[T], ABC):
    scalar_time_sampler: ScalarTimeSampler

    def __init__(
        self,
        cfg: T,
        variable_mapper: VariableMapper,
    ) -> None:
        super(TwoStageTimeSampler, self).__init__(cfg, variable_mapper)
        self.scalar_time_sampler = get_scalar_time_sampler(cfg.scalar_time_sampler)    
