from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor


@dataclass(slots=True, frozen=True)
class MetricsBatch:
    metrics: dict[str, Float[Tensor, ""]]
    batch_size: int
    evaluation_key: str
    sampler_key: str

