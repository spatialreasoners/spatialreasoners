from dataclasses import dataclass, fields

import torch
from torch import nn

from ...norm_layers import get_norm, Norm
from spatialreasoners.misc.nn_module_tools import constant_init


@dataclass
class MultiHeadAttentionCfg:
    """
    num_heads (int, optional): Number of heads in the attention.
    norm (str, optional): Name of normalization layer. Defaults to GroupNorm with 32 groups.
    """
    num_heads: int = 1  # will be ignored if num_head_channels is set
    num_head_channels: int | None = None
    norm: Norm = "group"


class MultiHeadAttention(nn.Module):
    """An attention block allows spatial position to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.  # noqa

    Args:
        in_channels (int): Channels of the input feature map.
        num_heads (int, optional): Number of heads in the attention.
        norm (dict, optional): Config for normalization layer. Defaults to GroupNorm with 32 groups.
    """

    def __init__(
        self,
        in_channels: int,
        num_heads: int = 1,  # will be ignored if num_head_channels is set
        num_head_channels: int | None = None,
        norm: Norm = "group"
    ):
        super(MultiHeadAttention, self).__init__()
        if num_head_channels is None:
            self.num_heads = num_heads
        else:
            assert in_channels % num_head_channels == 0
            self.num_heads = in_channels // num_head_channels
        self.norm = get_norm(norm, in_channels)
        self.qkv = nn.Conv1d(in_channels, in_channels * 3, 1)
        self.proj = nn.Conv1d(in_channels, in_channels, 1)
        self.init_weights()

    @classmethod
    def from_config(
        cls: type["MultiHeadAttention"], 
        config: MultiHeadAttentionCfg, 
        in_channels: int
    ) -> "MultiHeadAttention":
        return cls(in_channels, **{f.name: getattr(config, f.name) for f in fields(config)})

    @staticmethod
    def QKVAttention(qkv):
        channel = qkv.shape[1] // 3
        q, k, v = torch.chunk(qkv, 3, dim=1)
        scale = 1 / channel ** 0.25
        weight = torch.einsum('bct,bcs->bts', q * scale, k * scale)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        weight = torch.einsum('bts,bcs->bct', weight, v)
        return weight

    def forward(self, x):
        """Forward function for multi head attention.
        Args:
            x (torch.Tensor): Input feature map.

        Returns:
            torch.Tensor: Feature map after attention.
        """
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.QKVAttention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj(h)
        return (h + x).reshape(b, c, *spatial)

    def init_weights(self):
        constant_init(self.proj, 0)
