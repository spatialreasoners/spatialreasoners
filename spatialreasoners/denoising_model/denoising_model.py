from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from spatialreasoners.misc.nn_module_tools import requires_grad
from spatialreasoners.misc.step_tracker import StepTracker
from spatialreasoners.type_extensions import ConditioningCfg, Parameterization
from spatialreasoners.variable_mapper import VariableMapper

from .denoiser import DenoiserCfg, get_denoiser
from .flow import FlowCfg, get_flow
from .tokenizer import TokenizerCfg, get_tokenizer
from .type_extensions import ModelOutputs


@dataclass(frozen=True, kw_only=True)
class DenoisingModelCfg:
    denoiser: DenoiserCfg
    flow: FlowCfg
    tokenizer: TokenizerCfg
    conditioning: ConditioningCfg
    learn_uncertainty: bool
    learn_variance: bool
    time_interval: list[float] = field(default_factory=lambda: [0.0, 1.0])
    denoiser_parameterization: Parameterization = "ut"
    parameterization: Parameterization = "ut"
    has_ema: bool = False  # Exponential moving average of denoiser
    ema_decay_rate: float = 0.9999
    use_scaled_logsnr: bool = False
    logsnr_scale_factor: float = 0.125


C = TypeVar("C", bound=DenoisingModelCfg)


class DenoisingModel(nn.Module, Generic[C]):
    """
    Base class for tokenizers.
    """

    def __init__(
        self,
        cfg: DenoisingModelCfg,
        variable_mapper: VariableMapper,
        num_classes: int | None,
        step_tracker: StepTracker,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.variable_mapper = variable_mapper
        self.tokenizer = get_tokenizer(
            cfg.tokenizer,
            variable_mapper,
            predict_uncertainty=self.cfg.learn_uncertainty,
            predict_variance=self.cfg.learn_variance,
        )
        self.denoiser = get_denoiser(
            cfg.denoiser,
            self.tokenizer,
            num_classes=num_classes if cfg.conditioning.label else None,
        )

        if self.cfg.has_ema:
            self.ema_denoiser = AveragedModel(
                self.denoiser, multi_avg_fn=get_ema_multi_avg_fn(self.cfg.ema_decay_rate)
            )
            requires_grad(self.ema_denoiser, False)
        else:
            self.ema_denoiser = None

        self.flow = get_flow(cfg.flow, cfg.parameterization)
        self.step_tracker = step_tracker

    def forward(
        self,
        z_t: Float[Tensor, "batch time num_variables dim"],
        t: Float[Tensor, "batch time num_variables"],
        should_denoise: Bool[Tensor, "batch time num_variables"] | None = None,
        mask: Float[Tensor, "batch num_variables"] | None = None,
        x_masked: Float[Tensor, "batch num_variables dim"] | None = None,
        label: Int[Tensor, "batch"] | None = None,
        label_mask: Bool[Tensor, "batch"] | None = None,
        fixed_conditioning_fields: Any | None = None,
        sample: bool = False,
        use_ema: bool = True,  # NOTE ignored if sample == False or model has no EMA
    ) -> ModelOutputs:
        batch_size = z_t.shape[0]
        num_times = z_t.shape[1]  # Should be 1 for typical inference step

        t_flow_adjusted = self.flow.adjust_time(t)

        if self.cfg.use_scaled_logsnr:
            logsnr_val = self.flow.logsnr_t(t_flow_adjusted)
            scaled_logsnr_for_network = self.cfg.logsnr_scale_factor * logsnr_val
            t_input_for_network = scaled_logsnr_for_network  # Shape: [B, 1, N_vars]
        else:
            t_input_for_network = t_flow_adjusted  # Shape: [B, 1, N_vars]

        model_inputs = self.tokenizer.variables_to_model_inputs(
            z_t=z_t,
            t=t_input_for_network,  # Use scaled logSNR for the network, shape [B, 1, N_vars]
            should_denoise=should_denoise,
            mask=mask,
            x_masked=x_masked,
            label=label,
            label_mask=label_mask,
            fixed_conditioning_fields=fixed_conditioning_fields,
        )

        if sample:
            # NOTE Do not use compile for sampling because of varying batch sizes
            predicting_denoiser = (
                self.ema_denoiser if self.cfg.has_ema and use_ema else self.denoiser
            )
            unnormalized_outputs = predicting_denoiser.forward(model_inputs, sample=True)
        else:
            unnormalized_outputs = self.denoiser.forward_compiled(model_inputs)

        tokenizer_outputs = self.tokenizer.model_outputs_to_variable_predictions(
            model_outputs=unnormalized_outputs, batch_size=batch_size, num_times=num_times
        )

        t_flow_get_param_expanded = t_flow_adjusted.unsqueeze(-1).repeat(1, 1, 1, z_t.shape[-1])

        mean_theta = getattr(self.flow, f"get_{self.cfg.parameterization}")(
            t=t_flow_get_param_expanded,  # Use adjusted (physics) time, shape [B, 1, N_vars, Dim]
            zt=z_t,
            **{self.cfg.denoiser_parameterization: tokenizer_outputs.mean_theta},
        )

        # NOTE sigma parameterization is always the same as the model parameterization!
        if self.cfg.learn_uncertainty:
            sigma_theta = torch.exp(0.5 * tokenizer_outputs.logvar_theta)
        else:
            sigma_theta = None

        return ModelOutputs(
            mean_theta=mean_theta,
            variance_theta=tokenizer_outputs.variance_theta,
            sigma_theta=sigma_theta,
        )

    def on_sampling_start(self, num_noise_level_values: int | None = None) -> None:
        self.denoiser.on_sampling_start()
        self.flow.on_sampling_start(num_noise_level_values)

    def on_sampling_end(self) -> None:
        self.flow.on_sampling_end()
        self.denoiser.on_sampling_end()

    def get_weight_decay_parameter_groups(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        return self.denoiser.get_weight_decay_parameter_groups()

    def update_ema_denoiser(self) -> None:
        if self.ema_denoiser is not None:
            self.ema_denoiser.update_parameters(self.denoiser)
