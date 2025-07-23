import pytest
import torch

from src.sampler.sampling_schedule import MaskGIT, MaskGITCfg


@pytest.fixture
def sample_config():
    return MaskGITCfg(
        num_steps_per_group=2,
        num_groups=2,
    )

@pytest.fixture
def sample_instance(sample_config, mock_flow):
    batch_size = 2
    num_patches = 4
    
    dependency_matrix = torch.eye(num_patches)
    mask = None
    
    return MaskGIT(
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
    
    t_new, should_denoise = sample_instance(t=current_t, sigma_theta=None)
    
    assert t_new.shape == (batch_size, 4)
    assert should_denoise.shape == (batch_size, 4)
    
    # assert the t_new is smaller than current_t in 2 patches
    assert torch.all(t_new <= current_t)
    assert torch.all(t_new >= 0)
    assert torch.all(t_new.min(dim=1).values < 1) # At least one patch should be updated
    assert (t_new[0] == 1).sum() == 2 # 2 patches should not be updated
    assert (t_new[1] == 1).sum() == 2 # 2 patches should not be updated
    
    assert should_denoise.min() > 0 # None of the patches should be finished
    assert sample_instance.is_unfinished_mask.equal(torch.ones(batch_size, dtype=torch.bool)) # All batch elements are unfinished
    assert sample_instance.current_step == 1
    
def test_denoising_sequence(sample_instance):
    batch_size = 2
    num_patches = 4
    device = torch.device("cpu")
    current_t = torch.ones(batch_size, num_patches, device=device)
    last_unfinished_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    is_unfinished_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    
    assert sample_instance.is_unfinished_mask.equal(is_unfinished_mask)
    
    while is_unfinished_mask.any():
        sub_unfinished_mask = is_unfinished_mask[last_unfinished_mask]
        
        dropped_t = current_t[~sub_unfinished_mask]
        assert torch.allclose(dropped_t, torch.tensor(0.0, device=device))
        current_t = current_t[sub_unfinished_mask]
        
        t_new, should_denoise = sample_instance(t=current_t, sigma_theta=None)
        
        assert should_denoise.any(), "At least one patch should be updated"
        assert t_new.shape == (batch_size, num_patches)
        assert (current_t - t_new >= 0).all(), "t_new should be less than or equal to current_t"
        assert (current_t - t_new > 0).any(), "At least one patch should be updated"
        assert (t_new >= 0).all(), "t_new should be greater than or equal to 0"
        assert (t_new <= 1).all(), "t_new should be less than or equal to 1"
        assert sample_instance.current_step <= sample_instance.cfg.num_steps_per_group * sample_instance.cfg.num_groups
        
        current_t = t_new
        last_unfinished_mask = is_unfinished_mask
        is_unfinished_mask = sample_instance.is_unfinished_mask 
        
    assert sample_instance.current_step == sample_instance.cfg.num_steps_per_group * sample_instance.cfg.num_groups
    assert sample_instance.is_unfinished_mask.equal(torch.zeros(batch_size, dtype=torch.bool)) # All batch elements are finished