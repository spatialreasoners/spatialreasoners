from abc import ABC
from dataclasses import dataclass

from jaxtyping import Float, Int64
from torch import Tensor


@dataclass(slots=True, frozen=True, kw_only=True)
class InferenceBatchSample(ABC):
    z_t: Float[Tensor, "batch num_variables dim"] | None = None
    t: Float[Tensor, "batch num_variables"] | None = None
    sigma: Float[Tensor, "batch num_variables"] | None = None
    x: Float[Tensor, "batch num_variables dim"] | None = None
    step: Int64[Tensor, "batch"]
     
     
@dataclass(slots=True, frozen=True, kw_only=True)
class IntermediateInferenceBatchSample(InferenceBatchSample):
    """ 
    Intermediate inference batch sample. This is used to store the intermediate results of the inference process.
    It only contains the batch elements that were updated during the inference process. Their index is stored in the `in_batch_index` field.
    """
    in_batch_index: Int64[Tensor, "batch"] # Index of elements in the batch that were updated
    
    
@dataclass(slots=True, frozen=True, kw_only=True)
class FinalInferenceBatchSample(InferenceBatchSample):
    """
    Final inference batch sample. This is used to store the final results of the inference process. It always contains all the elements of the batch.
    """
    pass