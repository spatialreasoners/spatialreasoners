from dataclasses import dataclass, fields

from einops import rearrange
import torch
from torch import nn, Tensor
from torch.nn.functional import interpolate

from ...activations import get_activation, Activation
from ...norm_layers import get_norm, Norm
from ...embedding import Embedding
from spatialreasoners.misc.nn_module_tools import constant_init
from spatialreasoners.misc.tensor import unsqueeze_as


class NormWithEmbedding(nn.Module):
    """Normalization with embedding layer. If `use_scale_shift == True`,
    embedding results will be chunked and used to re-shift and re-scale
    normalization results. Otherwise, embedding results will directly add to
    input of normalization layer.

    Args:
        in_channels (int): Number of channels of the input feature map.
        emb_channels (int) Number of channels of the input embedding.
        norm (dict, optional): The config for the normalization layers.
            Defaults to GroupNorm with 32 groups.
        act (dict, optional): The config for the activation layers.
            Defaults to non-inplace SiLU.
        use_scale_shift (bool): If True, the output of Embedding layer will be
            split to 'scale' and 'shift' and map the output of normalization
            layer to ``out * (1 + scale) + shift``. Otherwise, the output of
            Embedding layer will be added with the input before normalization
            operation. Defaults to True.
    """

    def __init__(
        self,
        in_channels: int,
        t_emb: Embedding,
        norm: Norm = "group",
        act: Activation = "silu",
        use_scale_shift: bool = True
    ):
        super(NormWithEmbedding, self).__init__()
        self.t_emb = t_emb
        self.use_scale_shift = use_scale_shift
        self.norm = get_norm(norm, in_channels)

        emb_output = in_channels * 2 if use_scale_shift else in_channels
        self.embedding_layer = nn.Sequential(
            get_activation(act),
            nn.Linear(t_emb.d_out, emb_output)
        )

    def forward(self, x: Tensor, t: Tensor, c_emb: Tensor | None = None):
        """Forward function.

        Args:
            x (torch.Tensor) [B, C, ...]: Input feature map tensor.
            # TODO adapt this for other kinds of spatial conditionings
            t_emb (torch.Tensor) [B, 1, ...]: Shared time embedding or shared label embedding.
            c_emb (torch.Tensor) [B, d_c]

        Returns:
            torch.Tensor : Output feature map tensor.
        """
        t = interpolate(t, size=x.shape[2:], mode="bilinear")
        emb = self.t_emb.forward(t)
        if c_emb is not None:
            emb = emb + unsqueeze_as(c_emb, emb, 1)
        emb = rearrange(
            self.embedding_layer(emb), 
            "b () ... c -> b c ..."
        )
        if self.use_scale_shift:
            scale, shift = torch.chunk(emb, 2, dim=1)
            x = self.norm(x)
            x = x * (1 + scale) + shift
        else:
            x = self.norm(x + emb)
        return x


@dataclass
class ResBlockCfg:
    """
    norm (str, optional): The name of the normalization layers.
        Defaults to GroupNorm with 32 groups.
    act (str, optional): The name of the activation layers.
        Defaults to SiLU.
    shortcut_kernel_size (int, optional): The kernel size for the shortcut
        conv. Defaults to ``1``.
    use_scale_shift_norm (bool): Whether use scale-shift-norm in
        `NormWithEmbedding` layer. Defaults to ``False``
    dropout (float): Probability of the dropout layers. Defaults to ``0``
    """
    norm: Norm = "group"
    act: Activation = "silu"
    shortcut_kernel_size: int = 1
    use_scale_shift_norm: bool = False
    dropout: float | int = 0


class ResBlock(nn.Module):
    """Resblock for the denoising network. If `in_channels` not equals to
    `out_channels`, a learnable shortcut with conv layers will be added.

    Args:
        in_channels (int): Number of channels of the input feature map.
        embedding_channels (int): Number of channels of the input embedding.
        out_channels (int, optional): Number of output channels of the
            ResBlock. If not defined, the output channels will equal to the
            `in_channels`. Defaults to `None`.
    """
    cfg: ResBlockCfg

    def __init__(
        self,
        in_channels: int,
        t_emb: Embedding,
        out_channels: int | None = None,
        norm: Norm = "group",
        act: Activation = "silu",
        shortcut_kernel_size: int = 1,
        use_scale_shift_norm: bool = False,
        dropout: float = 0
    ):
        super(ResBlock, self).__init__()
        out_channels = in_channels if out_channels is None else out_channels

        self.conv_1 = nn.Sequential(
            get_norm(norm, in_channels),
            get_activation(act),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        self.norm_with_embedding = NormWithEmbedding(
            in_channels=out_channels,
            t_emb=t_emb,
            norm=norm,
            use_scale_shift=use_scale_shift_norm,
        )
        self.conv_2 = nn.Sequential(
            get_activation(act),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

        assert shortcut_kernel_size in [
            1, 3
        ], ('Only support `1` and `3` for `shortcut_kernel_size`, but '
            f'receive {shortcut_kernel_size}.')

        self.learnable_shortcut = out_channels != in_channels

        if self.learnable_shortcut:
            shortcut_padding = 1 if shortcut_kernel_size == 3 else 0
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                shortcut_kernel_size,
                padding=shortcut_padding
            )
        self.init_weights()

    @classmethod
    def from_config(
        cls: type["ResBlock"], 
        config: ResBlockCfg, 
        in_channels: int,
        t_emb: Embedding,
        out_channels: int | None = None,
    ) -> "ResBlock":
        return cls(in_channels, t_emb, out_channels, **{f.name: getattr(config, f.name) for f in fields(config)})

    def forward_shortcut(self, x):
        if self.learnable_shortcut:
            return self.shortcut(x)
        return x

    def forward(self, x: Tensor, t: Tensor, c_emb: Tensor | None = None):
        """Forward function.

        Args:
            x (torch.Tensor) [B, C, H, W]: Input feature map tensor.
            t (torch.Tensor) [B, 1, orig_height, orig_width]: Original resolution timestep map
            c_emb (torch.Tensor) [B, d_hid]: Shared embeddings

        Returns:
            torch.Tensor : Output feature map tensor.
        """
        shortcut = self.forward_shortcut(x)
        x = self.conv_1(x)
        x = self.norm_with_embedding(x, t, c_emb)
        x = self.conv_2(x)
        return x + shortcut

    def init_weights(self):
        # apply zero init to last conv layer
        constant_init(self.conv_2[-1], 0)
