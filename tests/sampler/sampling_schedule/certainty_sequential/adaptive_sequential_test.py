import pytest
import torch

from src.sampler.sampling_schedule import AdaptiveSequentialCfg, AdaptiveSequential

@pytest.fixture
def sample_config():
    return AdaptiveSequentialCfg(
        max_steps = 10,
        num_parallel=2,
        overlap=0.1,
        epsilon=1e-6,
        reverse_certainty=False,
    )


@pytest.fixture
def sample_instance(sample_config, mock_flow):
    batch_size = 2
    num_patches = 4

    num_patches = 4
    
    mask = torch.ones(batch_size, num_patches)
    
    return AdaptiveSequential(
        cfg=sample_config,
        num_patches=num_patches,
        model_flow=mock_flow,
        batch_size=batch_size,
        device=torch.device("cpu"),
        mask=mask
    )


def test_initialization(sample_instance):
    assert sample_instance.selectable.shape == (2, 4)
    assert torch.all(sample_instance.selectable == torch.ones(2, 4, dtype=torch.bool))
    assert sample_instance.scheduling_matrix.shape == (11, 2, 4)  # max_steps + 1
    
    # 2 parallel blocks, 2 patches each, 6 steps out of 10
    assert torch.all(sample_instance.block_lengths == torch.tensor([6, 6], dtype=torch.int32))
    assert sample_instance.prototypes.shape == (6, 2) # max prototype length, batch_size
    
    assert torch.allclose(sample_instance.prototypes[0], torch.tensor([1, 1], dtype=torch.float32))
    assert sample_instance.current_step == 0

def test_get_inference_lengths(sample_instance):
    num_inference_blocks = torch.tensor([2, 3], dtype=torch.int32)
    lengths = sample_instance._get_inference_lengths(num_inference_blocks)
    
    expected_lengths = sample_instance.cfg.max_steps / ((num_inference_blocks - 1) * (1 - sample_instance.cfg.overlap) + 1)
    torch.testing.assert_close(lengths, expected_lengths)


def test_get_schedule_prototypes(sample_instance):
    prototype_lengths = torch.tensor([3, 4], dtype=torch.int32)
    prototypes = sample_instance._get_schedule_prototypes(prototype_lengths)
    
    assert prototypes.shape == (4, 2)  # max prototype length x batch_size
    assert prototypes.min() >= 0
    assert prototypes.max() <= 1 + sample_instance.cfg.epsilon


def test_get_next_patch_ids(sample_instance):
    patch_uncertainty = torch.tensor([[0.1, 0.3, 0.2, 0.5], [0.7, 0.2, 0.1, 0.4]])
    selectable = torch.ones(2, 4, dtype=torch.bool)
    selected_ids = sample_instance._get_next_patch_ids(patch_uncertainty, selectable)
    
    assert selected_ids.shape == (2, sample_instance.cfg.num_parallel)
    assert torch.all(selected_ids >= 0) and torch.all(selected_ids < 4)  # IDs must be within patch range


def test_call_function(sample_instance):
    batch_size = 2
    device = torch.device("cpu")
    current_t = torch.ones(batch_size, 4, device=device)
    sigma_theta = torch.arange(8, dtype=torch.float32, device=device).reshape(2, 4)
    
    t_new, should_denoise = sample_instance(t=current_t, sigma_theta=sigma_theta)
    
    assert t_new.shape == (batch_size, 4)
    assert should_denoise.shape == (batch_size, 4)
    
    # assert the t_new is smaller than current_t in 2 patches
    assert torch.all(t_new <= current_t)
    assert torch.all(t_new >= 0)
    assert torch.all(t_new.min(dim=1).values < 1) # At least one patch should be updated
    assert (t_new[0] == 1).sum() == 2 # 2 patches should not be updated
    assert (t_new[1] == 1).sum() == 2 # 2 patches should not be updated
    
    assert torch.all(t_new[:, :2] < 1) # First two patches should be updated
    assert torch.all(t_new[:, 2:] == 1) # Last two patches should not be updated
    
    assert should_denoise.min() > 0 # None of the patches should be finished
    assert sample_instance.is_unfinished_mask.equal(torch.ones(batch_size, dtype=torch.bool, device=device)) # All batch elements are unfinished
    assert sample_instance.current_step == 1
    
    
    

def test_denoising_sequence(sample_instance):
    batch_size = 2
    num_patches = 4
    device = torch.device("cpu")
    current_t = torch.ones(batch_size, num_patches, device=device)
    sigma_theta = torch.arange(8, dtype=torch.float32, device=device).reshape(2, 4)
    
    last_unfinished_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    is_unfinished_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    
    assert sample_instance.is_unfinished_mask.equal(is_unfinished_mask)
    
    while is_unfinished_mask.any():
        sub_unfinished_mask = is_unfinished_mask[last_unfinished_mask]
        
        
        dropped_t = current_t[~sub_unfinished_mask]
        assert torch.allclose(dropped_t, torch.tensor(0.0, device=device))
        
        current_t = current_t[sub_unfinished_mask]
        sigma_theta = sigma_theta[sub_unfinished_mask]
        
        t_new, should_denoise = sample_instance(t=current_t, sigma_theta=sigma_theta)
        
        assert should_denoise.any(), "At least one patch should be updated"
        assert t_new.shape == (batch_size, num_patches)
        
        assert (current_t - t_new >= 0).all(), "t_new should be less than or equal to current_t"
        assert (current_t - t_new > 0).any(), "At least one patch should be updated"
        assert (t_new >= 0).all(), "t_new should be greater than or equal to 0"
        assert (t_new <= 1).all(), "t_new should be less than or equal to 1"
        
        current_t = t_new
        last_unfinished_mask = is_unfinished_mask
        is_unfinished_mask = sample_instance.is_unfinished_mask 
        
    assert sample_instance.current_step <= sample_instance.cfg.max_steps
    assert sample_instance.is_unfinished_mask.equal(torch.zeros(batch_size, dtype=torch.bool)) # All batch elements are finished