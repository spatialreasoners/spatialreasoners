from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import torch
from jaxtyping import Bool, Float, Int64
from torch import Tensor

from spatialreasoners.denoising_model import DenoisingModel
from spatialreasoners.misc.tensor import unsqueeze_as
from spatialreasoners.variable_mapper import VariableMapper

from . import InferenceSampler, InferenceSamplerCfg, register_inference_sampler
from .sampling_schedule import (
    FixedCfg,
    SamplingSchedule,
    SamplingScheduleCfg,
    get_sampling_schedule_class,
)
from .type_extensions import FinalInferenceBatchSample, IntermediateInferenceBatchSample


@dataclass(slots=True, frozen=True)
class IntermediateSamplingState:
    z_t: Float[Tensor, "batch num_variables dim"]
    t: Float[Tensor, "batch num_variables"]
    should_denoise: Bool[Tensor, "batch num_variables"] | None
    sigma_theta: Float[Tensor, "batch num_variables"] | None
    x_pred: Float[Tensor, "batch num_variables dim"] | None


@dataclass(kw_only=True, frozen=True, slots=True)
class ScheduledInferenceSamplerCfg(InferenceSamplerCfg):
    sampling_schedule: SamplingScheduleCfg = field(
        default_factory= lambda: FixedCfg(num_steps=10)
    ) # Default to 10 steps
    

@register_inference_sampler("scheduled", ScheduledInferenceSamplerCfg)
class ScheduledInferenceSampler(InferenceSampler[ScheduledInferenceSamplerCfg]):
    def __init__(
        self,
        cfg: ScheduledInferenceSamplerCfg,
        variable_mapper: VariableMapper,
    ) -> None:
        super().__init__(cfg, variable_mapper)
        self.sampling_schedule_class = get_sampling_schedule_class(
            cfg.sampling_schedule
        )

    @property
    def num_noise_level_values(self) -> int | None:
        return self.cfg.sampling_schedule.num_noise_level_values

    def _sampling_step(
        self,
        denoising_model: DenoisingModel,
        sampling_schedule: SamplingSchedule,
        z_t: Float[Tensor, "batch num_variables dim"],
        t: Float[Tensor, "batch num_variables"],
        should_denoise: Bool[Tensor, "batch num_variables"] | None = None,
        label: Int64[Tensor, "batch"] | None = None,
        mask: Float[Tensor, "batch num_variables"] | None = None,
        x_masked: Float[Tensor, "batch num_variables dim"] | None = None,
        fixed_conditioning_fields: Any | None = None,
        return_sigma: bool = True,
        return_x_pred: bool = True,
    ) -> IntermediateSamplingState:
        batch_size = z_t.shape[0]
        device = z_t.device
        feature_dim = z_t.shape[-1]

        # DenoisingModel.forward now expects the raw schedule time, unsqueezed to [B, 1, N_vars]
        t_input_for_model = t.unsqueeze(1) # Shape: [B, 1, N_vars]
        t_flow_current_adjusted = denoising_model.flow.adjust_time(t) # Shape: [B, N_vars]
        t_flow_current_for_ops = t_flow_current_adjusted.unsqueeze(1) # Shape: [B, 1, N_vars]

        # Handle Classifier-Free Guidance
        z_t_model_input = z_t.unsqueeze(1) # Shape: [B, 1, N_vars, Dim]
        should_denoise_model_input = should_denoise.unsqueeze(1) if should_denoise is not None else None # Shape: [B, 1, N_vars] or None

        if denoising_model.cfg.conditioning.label:
            label_cfg_scale = unsqueeze_as(self.label_cfg_scheduler(t), t) 
            apply_label_cfg = (label_cfg_scale != 1).any()
        else:
            apply_label_cfg = False

        if apply_label_cfg:
            label_mask, (z_t_cfg, t_cfg, should_denoise_cfg, label_cfg) = (
                self._prepare_classifier_free_guidance_split(
                    batch_size, device, z_t_model_input, t_input_for_model, 
                    should_denoise_model_input, label
                )
            )
        else:
            z_t_cfg = z_t_model_input
            t_cfg = t_input_for_model # Pass raw schedule time to DenoisingModel
            should_denoise_cfg = should_denoise_model_input
            label_cfg = label
            label_mask = None

        model_outputs = denoising_model(
            z_t=z_t_cfg, 
            t=t_cfg,     # Pass raw schedule time (unsqueezed) here
            should_denoise=should_denoise_cfg,
            label=label_cfg,
            label_mask=label_mask,
            mask=mask, # Original mask [B, N_vars], tokenizer handles it
            x_masked=x_masked, # Original x_masked [B, N_vars, Dim]
            sample=True,
            use_ema=self.cfg.use_ema,
            fixed_conditioning_fields=fixed_conditioning_fields,
        )
        mean_theta = model_outputs.mean_theta # Shape [B_cfg, 1, N_vars, Dim]
        v_theta = model_outputs.variance_theta       # Shape [B_cfg, 1, N_vars, Dim] or None
        sigma_theta = model_outputs.sigma_theta # Shape [B_cfg, 1, N_vars] or None
        
        if apply_label_cfg:
            # t_cfg is the scaled_logsnr time. It's used by merge for consistency if needed, but its output t is not used further.
            # The important outputs are mean_theta, v_theta, sigma_theta, now guided. (Shape [B, 1, ...])
            # z_t_cfg is not returned by merge. The z_t for conditional_p is the original z_t_model_input.
            _, _, mean_theta, v_theta, sigma_theta = (
                self._merge_classifier_free_guidance(
                    label_cfg_scale, z_t_cfg, t_cfg, mean_theta, v_theta, sigma_theta
                )
            )

        # Squeeze the time=1 dimension from model outputs
        mean_theta = mean_theta.squeeze(1) # Shape [B, N_vars, Dim]
        if v_theta is not None:
            v_theta = v_theta.squeeze(1)   # Shape [B, N_vars, Dim]
        
        # sigma_theta for output needs to be [B, N_vars]
        sigma_theta_for_schedule = None
        if sigma_theta is not None:
            sigma_theta_for_schedule = sigma_theta.squeeze(1) # Shape [B, N_vars]
            sigma_theta_for_schedule = (
                sigma_theta_for_schedule.masked_fill_(t == 0, 0) if return_sigma else None # Use t_schedule for masking
            )
        
        # Note: sampling_schedule expects t without the 'time=1' dimension.
        t_next, should_denoise = sampling_schedule(t, sigma_theta_for_schedule) # t_schedule is [B, N_vars]
        
        # Adjust t_next for flow operations
        t_flow_next_adjusted = denoising_model.flow.adjust_time(t_next) # Shape [B, N_vars]
        t_flow_next_for_ops = t_flow_next_adjusted.unsqueeze(1)

        t_repeat_for_cp = t_flow_current_for_ops.unsqueeze(-1).repeat(1, 1, 1, feature_dim) # [B,1,N_vars,Dim]
        t_next_repeat_for_cp = t_flow_next_for_ops.unsqueeze(-1).repeat(1, 1, 1, feature_dim)# [B,1,N_vars,Dim]
        
        conditional_p = denoising_model.flow.conditional_p(
            mean_theta.unsqueeze(1), # [B, 1, N_vars, Dim]
            z_t_model_input,         # [B, 1, N_vars, Dim]
            t_repeat_for_cp,         # Current flow time, expanded
            t_next_repeat_for_cp,    # Next flow time, expanded
            self.cfg.alpha,
            self.cfg.temperature,
            v_theta=v_theta.unsqueeze(1) if v_theta is not None else None, # [B,1,N_vars,Dim]
        )
        
        x_pred_output = None
        if return_x_pred:
            t_flow_current_for_get_x = t_flow_current_for_ops.unsqueeze(-1).repeat(1, 1, 1, feature_dim)
            
            x_pred_output = denoising_model.flow.get_x(
                t_flow_current_for_get_x, 
                zt=z_t_model_input,     
                **{denoising_model.cfg.parameterization: mean_theta.unsqueeze(1)}
            ).squeeze(1) # Squeeze the time=1 dim -> [B, N_vars, Dim]
            
        where_condition = t_flow_next_adjusted.unsqueeze(1).unsqueeze(-1) > 0 # [B,1,N_vars,1]
        z_t_next_intermediate = torch.where(
            where_condition, conditional_p.sample(), conditional_p.mean
        )
        z_t_next = z_t_next_intermediate.squeeze(1) # [batch, num_variables, dim]

        # sigma_theta for output needs to be [B, N_vars]
        output_sigma_theta = sigma_theta_for_schedule if return_sigma else None

        return IntermediateSamplingState(
            z_t=z_t_next, t=t_next, sigma_theta=output_sigma_theta, x_pred=x_pred_output, should_denoise=should_denoise
        )

    @torch.no_grad()
    def _sample(
        self,
        denoising_model: DenoisingModel,
        z_t: Float[Tensor, "batch num_variables dim"],
        t: Float[Tensor, "batch num_variables"],
        mask: Float[Tensor, "batch num_variables"] | None = None,
        x_masked: Float[Tensor, "batch num_variables dim"] | None = None,
        label: Int64[Tensor, "batch"] | None = None,
        fixed_conditioning_fields: Any | None = None,
        return_intermediate: bool = False,
        return_sigma: bool = False,
        return_x_pred: bool = False,
    ) -> Iterator[IntermediateInferenceBatchSample | FinalInferenceBatchSample]:
        return_sigma &= denoising_model.cfg.learn_uncertainty
        
        batch_size = z_t.shape[0]
        full_z_t = z_t.clone()
        full_t = t.clone()
        full_batch_steps = torch.zeros((batch_size, ), dtype=torch.long)
        
        if return_sigma:
            full_sigma = torch.ones_like(t) 

        sampling_schedule = self.sampling_schedule_class(
            cfg=self.cfg.sampling_schedule,
            batch_size=batch_size,
            device=z_t.device,
            variable_mapper=self.variable_mapper,
            model_flow=denoising_model.flow,
            mask=mask,
        )

        should_denoise = mask > 0 if mask is not None else None
        is_unfinished_mask = sampling_schedule.is_unfinished_mask
        last_unfinished_mask = torch.ones(batch_size, device=z_t.device, dtype=torch.bool)

        while is_unfinished_mask.any():
            # We can keep the same mask for all the samples if it didn't change
            if not last_unfinished_mask.equal(is_unfinished_mask):
                # Get mask, that corresponds to the current unfinished samples
                sub_unfinished_mask = is_unfinished_mask[last_unfinished_mask]

                z_t = z_t[sub_unfinished_mask]
                t = t[sub_unfinished_mask]
                label = label[sub_unfinished_mask] if label is not None else None
                mask = mask[sub_unfinished_mask] if mask is not None else None
                x_masked = x_masked[sub_unfinished_mask] if x_masked is not None else None
                last_unfinished_mask = is_unfinished_mask

            next_state = self._sampling_step(
                denoising_model=denoising_model,
                sampling_schedule=sampling_schedule,
                z_t=z_t,
                t=t,
                mask=mask,
                x_masked=x_masked,
                should_denoise=should_denoise,
                label=label,
                fixed_conditioning_fields=fixed_conditioning_fields,
                return_sigma=return_sigma,
                return_x_pred=return_x_pred,
            )

            if mask is not None:
                z_t_next = x_masked + mask.unsqueeze(-1) * next_state.z_t

            else:
                z_t_next = next_state.z_t

            # Update state for next iteration
            last_unfinished_mask = is_unfinished_mask   
            is_unfinished_mask = sampling_schedule.is_unfinished_mask  
            just_finished_mask = last_unfinished_mask & (~is_unfinished_mask)  # was unfinished, but now is finished
            
            # Update the full state -- but we can do that only if some elements just finished
            if (just_finished_mask).any():
                full_batch_steps[last_unfinished_mask] = sampling_schedule.current_step
                full_z_t[last_unfinished_mask] = z_t_next
                full_t[last_unfinished_mask] = next_state.t
                
                if return_sigma:
                    full_sigma[last_unfinished_mask] = next_state.sigma_theta
            
            if return_intermediate and is_unfinished_mask.any():
                updated_unfinished_idx = is_unfinished_mask.nonzero(as_tuple=True)[0] 
                
                # current unfinished batch elements are a subset of the last unfinished batch elements
                in_last_step = last_unfinished_mask[is_unfinished_mask]
                in_last_step_unfinished_idx = in_last_step.nonzero(as_tuple=True)[0]
                
                yield IntermediateInferenceBatchSample(
                    step=torch.full_like(
                        updated_unfinished_idx,
                        fill_value=sampling_schedule.current_step,
                        dtype=torch.long,
                        device=z_t.device,
                    ),
                    z_t=z_t_next[in_last_step_unfinished_idx],
                    t=next_state.t[in_last_step_unfinished_idx],
                    sigma=next_state.sigma_theta[in_last_step_unfinished_idx] if return_sigma else None,
                    x=next_state.x_pred[in_last_step_unfinished_idx] if return_x_pred else None,
                    in_batch_index=updated_unfinished_idx,
                )
                
            # Update the full state
            t = next_state.t
            z_t = z_t_next
            should_denoise = next_state.should_denoise
 
        yield FinalInferenceBatchSample(
            step=full_batch_steps,
            z_t=full_z_t,
            t=full_t,
            sigma=full_sigma if return_sigma else None,
            x=full_z_t if return_x_pred else None,
        )
