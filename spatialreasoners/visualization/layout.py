"""This file contains useful layout utilities for (batched) images. They are:

- add_border: Add a border to an image.
- cat/hcat/vcat: Join images by arranging them in a line. If the images have different
  sizes, they are aligned as specified (start, end, center). Allows you to specify a gap
  between images.

Images are assumed to be float32 tensors with shape (*batch channel, height, width).
"""
from collections.abc import Generator, Iterable
from typing import Any, Literal, Optional, Union

from einops import rearrange
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

Alignment = Literal["start", "center", "end"]
Axis = Literal["horizontal", "vertical"]
Color = Union[
    int,
    float,
    Iterable[int],
    Iterable[float],
    Float[Tensor, "#channel"],
    Float[Tensor, ""],
]


def _sanitize_color(color: Color) -> Float[Tensor, "#channel"]:
    # Convert tensor to list (or individual item).
    if isinstance(color, torch.Tensor):
        color = color.tolist()

    # Turn iterators and individual items into lists.
    if isinstance(color, Iterable):
        color = list(color)
    else:
        color = [color]

    return torch.tensor(color, dtype=torch.float32)


def _intersperse(iterable: Iterable, delimiter: Any) -> Generator[Any, None, None]:
    it = iter(iterable)
    yield next(it)
    for item in it:
        yield delimiter
        yield item


def _get_main_dim(main_axis: Axis) -> int:
    return {
        "horizontal": -1,
        "vertical": -2,
    }[main_axis]


def _get_cross_dim(main_axis: Axis) -> int:
    return {
        "horizontal": -2,
        "vertical": -1,
    }[main_axis]


def _compute_offset(base: int, overlay: int, align: Alignment) -> slice:
    assert base >= overlay
    offset = {
        "start": 0,
        "center": (base - overlay) // 2,
        "end": base - overlay,
    }[align]
    return slice(offset, offset + overlay)


def overlay(
    base: Float[Tensor, "*batch channel base_height base_width"],
    overlay: Float[Tensor, "*batch channel overlay_height overlay_width"],
    main_axis: Axis,
    main_axis_alignment: Alignment,
    cross_axis_alignment: Alignment,
) -> Float[Tensor, "*batch channel base_height base_width"]:
    # The overlay must be smaller than the base.
    *_, base_height, base_width = base.shape
    *_, overlay_height, overlay_width = overlay.shape
    assert base_height >= overlay_height and base_width >= overlay_width

    # Compute spacing on the main dimension.
    main_dim = _get_main_dim(main_axis)
    main_slice = _compute_offset(
        base.shape[main_dim], overlay.shape[main_dim], main_axis_alignment
    )

    # Compute spacing on the cross dimension.
    cross_dim = _get_cross_dim(main_axis)
    cross_slice = _compute_offset(
        base.shape[cross_dim], overlay.shape[cross_dim], cross_axis_alignment
    )

    # Combine the slices and paste the overlay onto the base accordingly.
    selector = [..., None, None]
    selector[main_dim] = main_slice
    selector[cross_dim] = cross_slice
    result = base.clone()
    result[selector] = overlay
    return result


def cat(
    main_axis: Axis,
    *images: Float[Tensor, "*batch channel _ _"],
    align: Alignment = "center",
    gap: int = 8,
    gap_color: Color = 1,
) -> Float[Tensor, "*batch channel height width"]:
    """Arrange images in a line. The interface resembles a CSS div with flexbox."""
    device = images[0].device
    gap_color = _sanitize_color(gap_color).to(device)

    # Find the maximum image side length in the cross axis dimension.
    cross_dim = _get_cross_dim(main_axis)
    cross_axis_length = max(image.shape[cross_dim] for image in images)

    # Pad the images.
    padded_images = []
    for image in images:
        # Create an empty image with the correct size.
        padded_shape = list(image.shape)
        padded_shape[cross_dim] = cross_axis_length
        base = torch.ones(padded_shape, dtype=torch.float32, device=device)
        base = base * gap_color[:, None, None]
        padded_images.append(overlay(base, image, main_axis, "start", align))

    # Intersperse separators if necessary.
    if gap > 0:
        # Generate a separator.
        *batch, _, _ = images[0].shape
        separator_size = [gap, gap]
        separator_size[cross_dim] = cross_axis_length
        separator = torch.ones((*batch, *separator_size), dtype=torch.float32, device=device)
        separator = separator * gap_color[:, None, None]

        # Intersperse the separator between the images.
        padded_images = list(_intersperse(padded_images, separator))

    return torch.cat(padded_images, dim=_get_main_dim(main_axis))


def hcat(
    *images: Float[Tensor, "*batch channel _ _"],
    align: Literal["start", "center", "end", "top", "bottom"] = "start",
    gap: int = 8,
    gap_color: Color = 1,
):
    """Shorthand for a horizontal linear concatenation."""
    return cat(
        "horizontal",
        *images,
        align={
            "start": "start",
            "center": "center",
            "end": "end",
            "top": "start",
            "bottom": "end",
        }[align],
        gap=gap,
        gap_color=gap_color,
    )


def vcat(
    *images: Float[Tensor, "*batch channel _ _"],
    align: Literal["start", "center", "end", "left", "right"] = "start",
    gap: int = 8,
    gap_color: Color = 1,
):
    """Shorthand for a horizontal linear concatenation."""
    return cat(
        "vertical",
        *images,
        align={
            "start": "start",
            "center": "center",
            "end": "end",
            "left": "start",
            "right": "end",
        }[align],
        gap=gap,
        gap_color=gap_color,
    )


def add_border(
    image: Float[Tensor, "*batch channel height width"],
    border: int = 8,
    color: Color = 1,
) -> Float[Tensor, "*batch channel new_height new_width"]:
    color = _sanitize_color(color).to(image)
    *batch, h, w = image.shape
    result = color.expand(*batch, h + 2 * border, w + 2 * border).contiguous()
    result[..., border : h + border, border : w + border] = image
    return result


def resize(
    image: Float[Tensor, "*batch channel height width"],
    shape: Optional[tuple[int, int]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Float[Tensor, "*batch channel new_height new_width"]:
    assert (shape is not None) + (width is not None) + (height is not None) == 1
    *batch, h, w = image.shape

    if width is not None:
        shape = (int(h * width / w), width)
    elif height is not None:
        shape = (height, int(w * height / h))

    res: Tensor = F.interpolate(
        rearrange(image, "... c h w -> (...) c h w"),
        shape,
        mode="bilinear",
        align_corners=False,
        antialias="bilinear",
    )
    return res.reshape(*batch, *shape)
