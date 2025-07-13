from dataclasses import dataclass, fields
from typing import Literal

from torch import nn, Tensor
import torch.nn.functional as F


@dataclass
class UpsampleCfg:
    with_conv: bool = True
    mode: Literal["nearest"] = "nearest"


class Upsample(nn.Module):
    """Upsampling operation used in the denoising network. Allows users to
    apply an additional convolution layer after the nearest interpolation
    operation.

    Args:
        in_channels (int): Number of channels of the input feature map to be
            downsampled.
        with_conv (bool, optional): Whether apply an additional convolution
            layer after upsampling.  Defaults to `True`.
    """

    def __init__(
        self,
        in_channels: int,
        with_conv: bool = True,
        mode: Literal["nearest"] = "nearest"
    ):
        super(Upsample, self).__init__()
        self.with_conv = with_conv
        self.mode = mode
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

    @classmethod
    def from_config(
        cls: type["Upsample"], 
        config: UpsampleCfg, 
        in_channels: int
    ) -> "Upsample":
        return cls(in_channels, **{f.name: getattr(config, f.name) for f in fields(config)})

    def forward(self, x: Tensor, shape: tuple[int, int] | None = None):
        """Forward function for upsampling operation.
        Args:
            x (torch.Tensor): Feature map to upsample.
            shape (tuple[int, int]): Output size.
        Returns:
            torch.Tensor: Feature map after upsampling.
        """
        x = F.interpolate(x, size=shape, mode=self.mode)
        if self.with_conv:
            x = self.conv(x)
        return x
