import math

import pytest
import torch

from src.sampler.sampling_schedule import Fixed, FixedCfg


@pytest.fixture
def sample_config():
    return FixedCfg(
        num_steps = 10,
    )

@pytest.fixture
def sample_instance(sample_config, mock_flow):
    batch_size = 2
    num_patches = 4
    num_patches = 4
    
    dependency_matrix = torch.eye(num_patches)
    mask = torch.ones(batch_size, num_patches)
    
    return Fixed(
        cfg=sample_config,
        num_patches=num_patches,
        model_flow=mock_flow,
        batch_size=batch_size,
        device=torch.device("cpu"),
        dependency_matrix=dependency_matrix,
        mask=mask
    )

def test_call_function(sample_instance):
    batch_size = 2
    device = torch.device("cpu")
    current_t = torch.ones(batch_size, 4, device=device)
    
    assert sample_instance.current_step == 0
    assert sample_instance.noise_levels.shape == (sample_instance.cfg.num_steps + 1,)
    
    t_new, should_denoise = sample_instance(t=current_t, sigma_theta=None)
    
    assert t_new.shape == (batch_size, 4)
    assert should_denoise.shape == (batch_size, 4)
    
    assert torch.all(t_new <= current_t)
    assert torch.all(t_new >= 0)
    assert math.isclose(t_new.std(), 0, abs_tol=1e-5), "All patches should be updated with the same noise level"
    
    assert torch.all(should_denoise) # All batch elements are unfinished
    assert sample_instance.is_unfinished_mask.equal(torch.ones(batch_size, dtype=torch.bool)) # All batch elements are unfinished
    assert sample_instance.current_step == 1
    
    
def test_finishes_after_num_steps(sample_instance):
    batch_size = 2
    num_patches = 4
    device = torch.device("cpu")
    current_t = torch.ones(batch_size, num_patches, device=device)
    
    for _ in range(sample_instance.cfg.num_steps):
        t_new, should_denoise = sample_instance(t=current_t, sigma_theta=None)
        
        assert should_denoise.any(), "At least one patch should be updated"
        assert t_new.shape == (batch_size, num_patches)
        
        assert (current_t - t_new >= 0).all(), "t_new should be less than or equal to current_t"
        assert (current_t - t_new > 0).any(), "At least one patch should be updated"
        assert (t_new >= 0).all(), "t_new should be greater than or equal to 0"
        assert (t_new <= 1).all(), "t_new should be less than or equal to 1"
        
        current_t = t_new
        
    assert sample_instance.current_step == sample_instance.cfg.num_steps
    assert sample_instance.is_unfinished_mask.equal(torch.zeros(batch_size, dtype=torch.bool)) # All batch elements are finished