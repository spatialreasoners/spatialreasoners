from torch import Tensor
from torch.nn import Sequential

from .res_block import ResBlock
from .upsample import Upsample


class EmbedSequential(Sequential):
    """A sequential module that passes timestep embeddings to the children that
    support it as an extra input.

    Modified from
    https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/unet.py#L35
    """

    def forward(
        self, 
        x: Tensor, 
        t: Tensor,
        c_emb: Tensor | None = None,
        shape: tuple[int, int] | None = None
    ) -> Tensor:
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, t, c_emb)
            elif isinstance(layer, Upsample):
                x = layer(x, shape)
            else:
                x = layer(x)
        return x
