from typing import NotRequired, TypedDict

from jaxtyping import Float
from numpy import ndarray
from PIL.Image import Image


class ImageExample(TypedDict, total=True):
    # NOTE this Image can include a mask as alpha channel (LA or RGBA mode)
    image: NotRequired[Image]
    latent: NotRequired[Float[ndarray, "channel height width"]]
    label: NotRequired[int | dict[str, int]]
    path: NotRequired[str]
