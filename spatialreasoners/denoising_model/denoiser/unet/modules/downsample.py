from dataclasses import dataclass, fields

from torch import nn, Tensor


@dataclass
class DownsampleCfg:
    with_conv: bool = True


class Downsample(nn.Module):
    """Downsampling operation used in the denoising network. Support average
    pooling and convolution for downsample operation.

    Args:
        in_channels (int): Number of channels of the input feature map to be
            downsampled.
        with_conv (bool, optional): Whether use convolution operation for
            downsampling.  Defaults to `True`.
    """

    def __init__(
        self, 
        in_channels: int, 
        with_conv: bool = True
    ):
        super(Downsample, self).__init__()
        if with_conv:
            self.downsample = nn.Conv2d(in_channels, in_channels, 3, 2, 1)
        else:
            self.downsample = nn.AvgPool2d(stride=2)

    @classmethod
    def from_config(
        cls: type["Downsample"], 
        config: DownsampleCfg, 
        in_channels: int
    ) -> "Downsample":
        return cls(in_channels, **{f.name: getattr(config, f.name) for f in fields(config)})

    def forward(self, x: Tensor):
        """Forward function for downsampling operation.
        Args:
            x (torch.Tensor): Feature map to downsample.

        Returns:
            torch.Tensor: Feature map after downsampling.
        """
        return self.downsample(x)
