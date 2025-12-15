import io
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor
from torchvision.io import write_video


def fig_to_image(
    fig: Figure,
    dpi: int = 100,
    device: torch.device = torch.device("cpu"),
) -> Float[Tensor, "3 height width"]:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="raw", dpi=dpi)
    buffer.seek(0)
    data = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    h = int(fig.bbox.bounds[3])
    w = int(fig.bbox.bounds[2])
    data = rearrange(data, "(h w c) -> c h w", h=h, w=w, c=4)
    buffer.close()
    return (torch.tensor(data, device=device, dtype=torch.float32) / 255)[:3]


def prep_images(
    tensor: Float[Tensor, "*batch channel height width"],
    normalize: bool = False,
    value_range: tuple[int, int] | None = (-1, 1),
    scale_each: bool = False,
    channel_last: bool = True
) -> Union[
    UInt8[np.ndarray, "*batch height width 3"],
    UInt8[np.ndarray, "*batch height width 4"],
    UInt8[np.ndarray, "*batch 3 height width"],
    UInt8[np.ndarray, "*batch 4 height width"]
]:
    # Ensure that there are 3 or 4 channels.
    *_, channel, _, _ = tensor.shape
    if channel == 1:
        tensor = repeat(tensor, "... () h w -> ... c h w", c=3)
    assert tensor.shape[-3] in (3, 4)

    if normalize:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if value_range is not None and not isinstance(value_range, tuple):
            raise TypeError("value_range has to be a tuple (min, max) if specified. min and max are numbers")

        def norm_ip(img: Tensor, low, high):
            img.clamp_(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        if value_range is not None:
            norm_ip(tensor, value_range[0], value_range[1])
        elif scale_each:
            norm_ip(tensor, tensor.amin(dim=(-1, -2, -3), keepdim=True), tensor.amax(dim=(-1, -2, -3), keepdim=True))
        else:
            norm_ip(tensor, tensor.min(), tensor.max())

    tensor = (tensor.detach().clip(min=0, max=1) * 255)\
        .to(dtype=torch.uint8, device="cpu")
    if channel_last:
        tensor = rearrange(tensor, "... c h w -> ... h w c")
    return tensor.numpy()


def save_image(
    image: UInt8[np.ndarray, "height width channel"],
    path: Union[Path, str],
) -> None:
    """Save an image. Assumed to be in range 0-1."""
    # Create the parent directory if it doesn't already exist.
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
        
    # Handle both channel-first and channel-last formats
    if image.shape[0] in (3, 4):  # Channel-first format
        image = np.transpose(image, (1, 2, 0))
    
    # Save the image.
    Image.fromarray(image).save(path)


def load_image(
    path: Union[Path, str],
) -> Float[Tensor, "3 height width"]:
    return tf.ToTensor()(Image.open(path))[:3]


def save_video(
    video: UInt8[np.ndarray, "frame 3 height width"],
    path: Union[Path, str],
    **kwargs
) -> None:
    """Save an image. Assumed to be in range 0-1."""
    # Create the parent directory if it doesn't already exist.
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    # Save the video.
    write_video(str(path), video.transpose(0, 2, 3, 1), **kwargs)
