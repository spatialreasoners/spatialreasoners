from abc import ABC
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

import torch
from jaxtyping import Float, Int
from pytorch_lightning.loggers import Logger
from torch import Tensor

from spatialreasoners.denoising_model import DenoisingModel
from spatialreasoners.type_extensions import BatchUnstructuredExample, BatchVariables, Stage
from spatialreasoners.variable_mapper import VariableMapper

from . import register_evaluation
from .evaluation import Evaluation, EvaluationCfg
from .inference_sampler import (
    FinalInferenceBatchSample,
    InferenceBatchSample,
    InferenceSamplerCfg,
    IntermediateInferenceBatchSample,
    ScheduledInferenceSamplerCfg,
    get_inference_sampler,
)
from .type_extensions import MetricsBatch


@dataclass(frozen=True, kw_only=True)
class SamplingEvaluationCfg(EvaluationCfg):
    samplers: dict[str, InferenceSamplerCfg] = field(
        default_factory=lambda: {"default_fixed": ScheduledInferenceSamplerCfg()}
    )
    num_log_samples: int | None = None         # saves all if None
    save_final_samples: bool = True           # whether to save test samples
    save_intermediate_samples: bool = True
    calculate_metrics: bool = True     
    visualize_sigma: bool = True
    visualize_x: bool = True


T = TypeVar("T", bound=SamplingEvaluationCfg)


@register_evaluation("sampling", SamplingEvaluationCfg)
class SamplingEvaluation(Evaluation[T], ABC):
    def __init__(
        self,
        cfg: T,
        variable_mapper: VariableMapper,
        tag: str,
        output_dir: Path,
        stage: Stage = "test",
    ) -> None:
        super().__init__(cfg, variable_mapper, tag, stage=stage, output_dir=output_dir)
        
        self.samplers = {
            sampler_key: get_inference_sampler(config, variable_mapper) 
            for sampler_key, config in self.cfg.samplers.items()
        }
            
    @property
    def visualize_intermediate(self) -> bool:
        return self.cfg.save_intermediate_samples 
    
    @property
    def visualize_final(self) -> bool:
        return self.cfg.save_final_samples


    def _log_intermediate_sample(
        self,
        model: DenoisingModel,
        batch_variables: BatchVariables,
        in_batch_index: Int[Tensor, "batch"],
        sampler_key: str,
        logger: Logger,
    ) -> None:
        # raise exception instead of abstract method because it might not be required
        raise NotImplementedError("Intermediate sample visualization not implemented.")
        
    def _log_final_sample(
        self,
        model: DenoisingModel,
        batch_variables: BatchVariables,
        in_batch_index: Int[Tensor, "batch"],
        sampler_key: str,
        logger: Logger,
    ) -> None:
        """ Visualize and log the final sample
        For the implementation you could use the self.variable_mapper to get the unstructured example. 
        Keep in mind that for the mapping to succeed you need to either implement how to map sigma 
        and t to the unstructured example or set them to None in the batch_variables.
        
        You can access the logger using model.logger.
        """
        # raise exception instead of abstract method because it might not be required
        raise NotImplementedError("Sample visualization not implemented.")
    
    def _get_metrics(
        self,
        batch_variables: BatchVariables,
        labels: Tensor | None,
    ) -> dict[str, Float[Tensor, ""]] | None:
        """ Get the metrics for the given batch variables
        """
        # raise exception instead of abstract method because it might not be required
        raise NotImplementedError("Metrics calculation not implemented.")
    
    def _get_noise_z_and_t(
        self,
        denoising_model: DenoisingModel,
        batch: BatchVariables
    ) -> tuple[
        Float[Tensor, "batch num_variables dim"], # z_t
        Float[Tensor, "batch num_variables"], # t
    ]:
        device = batch.device 
        shape = batch.z_t.shape
        
        z_noise = (torch.empty if self.cfg.deterministic else torch.randn)(
            shape, 
            device=device
        ) # template or noise
        
        if self.cfg.deterministic: # Fill template with deterministic noise
            for i, index in enumerate(batch.in_dataset_index):
                z_noise[i] = torch.randn(
                    shape[1:], # skip batch dimension
                    generator=torch.Generator(device).manual_seed(index.item()), 
                    device=device
                )
                
        t = torch.full(
            (batch.batch_size, batch.num_variables),
            fill_value=denoising_model.cfg.time_interval[1],
            device=device,
            dtype=torch.float,
        )
                
        return z_noise, t
    
    @torch.no_grad()
    def _evaluate(
        self, 
        denoising_model: DenoisingModel, 
        batch: BatchVariables,
        num_elem_to_log: int,
        logger: Logger,
    ) -> Iterator[MetricsBatch] | None:
        
        # TODO Use TaskSampler (or TimeAndMaskSampler) here, and get the mask
        clean_x = batch.z_t
        z_t, t = self._get_noise_z_and_t(denoising_model, batch)
        
        if batch.mask is not None and denoising_model.cfg.conditioning.mask:
            
            unsqueezed_mask = batch.mask.unsqueeze(-1)
            
            mask = batch.mask
            x_masked = clean_x * (1 - unsqueezed_mask)
            t =torch.minimum(t, mask)
            t = torch.clamp(t, min=denoising_model.cfg.time_interval[0])
            z_t = z_t * unsqueezed_mask + x_masked * (1 - unsqueezed_mask)
            
        else:
            mask = None
            x_masked = None
            
        label = batch.label if denoising_model.cfg.conditioning.label else None
        visualize_final = self.visualize_final
        visualize_intermediate = self.visualize_intermediate
        
        if num_elem_to_log == 0:
            visualize_final = False
            visualize_intermediate = False
        else: 
            batch_idx_to_log = torch.arange(num_elem_to_log, device=batch.device)
            
        inference_sample: InferenceBatchSample
        
        for sampler_key, sampler in self.samplers.items():
            for inference_sample in sampler(
                denoising_model,
                z_t=z_t,
                t=t,
                mask=mask,
                x_masked=x_masked,
                label=label,
                fixed_conditioning_fields=batch.fixed_conditioning_fields,
                return_intermediate=visualize_intermediate, 
                return_sigma=self.cfg.visualize_sigma,
                return_x_pred=self.cfg.visualize_x,
            ):
                if isinstance(inference_sample, IntermediateInferenceBatchSample):
                    indices_to_log = batch_idx_to_log[torch.isin(batch_idx_to_log, inference_sample.in_batch_index)]
                    
                    if len(indices_to_log) == 0:
                        continue # Nothing to log/visualize
                    
                    inference_sub_selection_mask = torch.isin(inference_sample.in_batch_index, indices_to_log)
                    
                    selected_paths = None
                    if batch.path is not None:
                        selected_paths = [batch.path[i] for i in indices_to_log]
                    
                    limited_batch_variables = BatchVariables(
                        in_dataset_index=batch.in_dataset_index[indices_to_log],
                        z_t=inference_sample.z_t[inference_sub_selection_mask],
                        label=batch.label[indices_to_log] if batch.label is not None else None,
                        path=selected_paths,
                        mask=batch.mask[indices_to_log] if batch.mask is not None else None,
                        t=inference_sample.t[inference_sub_selection_mask],
                        sigma_pred=inference_sample.sigma[inference_sub_selection_mask] if inference_sample.sigma is not None else None,
                        x_pred=inference_sample.x[inference_sub_selection_mask] if inference_sample.x is not None else None,
                        num_steps=inference_sample.step[inference_sub_selection_mask] 
                    )
                    
                    if self.cfg.visualize_x:
                        assert inference_sample.x is not None, "x should be in inference batch"
                    
                    self._persist_intermediate_sample(
                        denoising_model,
                        limited_batch_variables,
                        in_batch_index=indices_to_log,
                        sampler_key=sampler_key,
                        logger=logger,
                    )

            assert isinstance(inference_sample, FinalInferenceBatchSample)
    
            if self.cfg.calculate_metrics:
                batch_variables = BatchVariables(
                    in_dataset_index=batch.in_dataset_index,
                    z_t=inference_sample.z_t,
                    label=batch.label,
                    path=batch.path,
                    mask=batch.mask,
                    t=inference_sample.t,
                    sigma_pred=inference_sample.sigma,
                    x_pred=inference_sample.x,
                    num_steps=inference_sample.step
                )
                metrics = self._get_metrics(batch_variables, batch.label) 
                
                if metrics is not None:
                    yield MetricsBatch(
                        metrics=metrics,
                        batch_size=batch_variables.batch_size,
                        sampler_key=sampler_key,
                        evaluation_key=self.name_key
                    )

            if visualize_final:
                if self.cfg.visualize_x:
                    assert inference_sample.x is not None, "x should be in inference batch"
                        
                to_vis_batch_variables = BatchVariables(
                    in_dataset_index=batch.in_dataset_index[:num_elem_to_log],
                    z_t=inference_sample.z_t[:num_elem_to_log],
                    label=batch.label[:num_elem_to_log] if batch.label is not None else None,
                    path=batch.path[:num_elem_to_log] if batch.path is not None else None,
                    mask=None,
                    t=inference_sample.t[:num_elem_to_log] if inference_sample.t is not None else None,
                    sigma_pred=inference_sample.sigma[:num_elem_to_log] if inference_sample.sigma is not None else None,
                    x_pred=inference_sample.x[:num_elem_to_log] if inference_sample.x is not None else None,
                    num_steps=inference_sample.step[:num_elem_to_log]
                )
                
                self._persist_final_sample(
                    denoising_model, 
                    to_vis_batch_variables, 
                    sampler_key=sampler_key, 
                    in_batch_index=batch_idx_to_log,
                    logger=logger,
                )
                
    def __call__(
        self, 
        denoising_model: DenoisingModel, 
        batch: BatchUnstructuredExample,
        batch_idx: int,
        logger: Logger,
        global_rank: int = 0,
    ) -> Iterator[MetricsBatch] | None:
        """ Evaluate the model on the given batch
        """
        batch_variables = self.variable_mapper.unstructured_to_variables(batch)
        batch_size = batch_variables.batch_size
        
        if self.stage == "test" or (self.stage == "val" and global_rank == 0):
            if self.cfg.num_log_samples is None:
                num_elem_to_log = batch_size
            else:
                num_elem_to_log = max(0, min(batch_size, self.cfg.num_log_samples - batch_size * batch_idx)) 
        else:
            num_elem_to_log = 0
            
        return self._evaluate(denoising_model, batch_variables, num_elem_to_log=num_elem_to_log, logger=logger)
            
    
    @property
    def name_key(self) -> str:
        return f"{self.stage}/{self.tag}"