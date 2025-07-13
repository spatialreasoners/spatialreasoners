import pytest
import torch
import pytest
from src.sampler.sampling_schedule import (
    FixedCfg, 
    MaskGITCfg, 
    AdaptiveInverseCfg, 
    AdaptiveSoftmaxCfg, 
    GraphSequentialCfg, 
    AdaptiveSequentialCfg
)
from src.sampler import ScheduledSampler, ScheduledSamplerCfg

def get_sampling_schedule_configs():
    return [
        FixedCfg(
            num_steps=10,
        ),
        MaskGITCfg(
            num_steps_per_group=10,
            num_groups=2,
        ),
        AdaptiveInverseCfg(
            max_steps=10,
            alpha=0.5,
            div_threshold=None,
            max_approx_error=5.0e-4,
            num_search_steps=40,
            min_alpha=1.0e-5,
            min_step_size=0.1,
            finished_threshold=1.0e-2,
        ),
        AdaptiveSoftmaxCfg(
            scale=0.1,
            max_clip_iter=8,
            finished_threshold=None,
            max_steps=10,
            alpha=0.5,
        ),
        GraphSequentialCfg(
            max_steps=10,
            num_parallel=2,
            overlap=0.1,
            epsilon=1e-6,
            reverse_certainty=False,
            max_order=1,
            certainty_decay=0.5
        ),
        AdaptiveSequentialCfg(
            max_steps=10,
            num_parallel=2,
            overlap=0.1,
            epsilon=1e-6,
            reverse_certainty=False,
        )   
    ]

@pytest.mark.parametrize("sampling_schedule_cfg", get_sampling_schedule_configs())
def test_scheduled_sampler_initialization(sampling_schedule_cfg):
    patch_size = 4
    patch_grid_shape = [2, 2]
    total_patches = patch_grid_shape[0] * patch_grid_shape[1]
    dependency_matrix = torch.eye(total_patches)
    
    cfg = ScheduledSamplerCfg(
        sampling_schedule=sampling_schedule_cfg
    )

    sampler = ScheduledSampler(cfg, patch_size, patch_grid_shape, dependency_matrix)

    assert sampler.cfg == cfg
    assert sampler.patch_size == patch_size
    assert sampler.patch_grid_shape == patch_grid_shape
    assert torch.equal(sampler.dependency_matrix, dependency_matrix)
    assert sampler.sampling_schedule_class is not None
    # Check num_noise_level_values property
    assert sampler.num_noise_level_values == sampling_schedule_cfg.num_noise_level_values



