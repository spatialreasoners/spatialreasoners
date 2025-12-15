from dataclasses import dataclass

import numpy as np
from jaxtyping import Float


@dataclass(kw_only=True, frozen=True)
class VideoExample: 
    video: Float[np.ndarray, "frames height width channels"]
    camera_params: Float[np.ndarray, "frames 6"]
    path: str