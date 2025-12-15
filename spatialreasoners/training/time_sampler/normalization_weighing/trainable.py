from dataclasses import dataclass
from typing import Callable
import torch

from torch import nn
from jaxtyping import Float, Bool
from torch import Tensor, device as torch_device

from . import register_normalization_weighing
from .normalization_weighing import NormalizationWeighing, NormalizationWeighingCfg


class WeighingModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, max_weight=15, num_residual_blocks=6):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.residual_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_residual_blocks)
            ]
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

        self.max_weight = max_weight

    def _model(self, x: Float[Tensor, "batch 2"]) -> Float[Tensor, "batch"]:
        in_processed = self.input_layer(x)
        for block in self.residual_blocks:
            in_processed = in_processed + block(in_processed)
        return self.output_layer(in_processed)

    def forward(self, x: Float[Tensor, "batch 2"]) -> Float[Tensor, "batch"]:
        """
        x: [batch, 2] tensor with t and t_mean
        return: [batch, 1] tensor with the weight
        """
        assert x.shape[1] == 2
        assert x.shape[0] > 0
        x = x.clone()

        shifted_x = x - 0.5  # center around 0
        shifted_x = shifted_x

        out = self._model(shifted_x).flatten()  # [batch]
        out = torch.sigmoid(out)

        return out * self.max_weight  # scale back to [0, max_weight]


@dataclass(frozen=True, kw_only=True)
class TrainableWeightingCfg(NormalizationWeighingCfg):
    weighing_model_path: str = "weighing_model.pth"
    hidden_dim: int = 64
    num_residual_blocks: int = 6
    max_weight: float = 15.0


@register_normalization_weighing("trainable", TrainableWeightingCfg)
class TrainableWeighting(NormalizationWeighing[TrainableWeightingCfg]):

    def __init__(
        self,
        cfg: TrainableWeightingCfg,
        get_time_func: Callable[
            [int, int, str],
            tuple[
                Float[Tensor, "batch sample num_variables"],
                Bool[Tensor, "batch #sample num_variables"] | None,
            ],
        ],
        device: torch_device | str = "cpu",
    ):
        super().__init__(cfg, get_time_func, device)
        self.model = WeighingModel(
            input_dim=2,
            hidden_dim=cfg.hidden_dim,
            num_residual_blocks=cfg.num_residual_blocks,
            max_weight=cfg.max_weight,
        )
        self.model.load_state_dict(torch.load(cfg.weighing_model_path))
        self.model.eval()
        self.model.to(device)

    def __call__(
        self,
        t: Float[Tensor, "batch num_samples num_variables"],
        mask: Bool[Tensor, "batch #sample num_variables"] | None = None,
    ) -> Float[Tensor, "batch num_samples num_variables"]:  # batch, num_samples, num_variables
        final_shape = t.shape
        t_means = t.mean(dim=-1)

        t_means = t_means.unsqueeze(-1)
        t_means = t_means.repeat_interleave(t.shape[-1], dim=-1)  # Add back the variables dimension

        t_means = t_means.flatten()
        t_flat = t.flatten()

        x = torch.stack([t_flat, t_means], dim=-1)  # [batch * num_samples * num_variables, 2]

        weights = self.model(x)  # [batch * num_samples * num_variables]
        return weights.reshape(final_shape)
