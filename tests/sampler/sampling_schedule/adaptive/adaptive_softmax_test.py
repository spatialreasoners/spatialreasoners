import pytest
import math
import torch

from src.sampler.sampling_schedule import AdaptiveSoftmax, AdaptiveSoftmaxCfg


@pytest.fixture
def sample_config():
    return AdaptiveSoftmaxCfg(
        scale=0.1,
        max_clip_iter=8,
        finished_threshold=None,
        max_steps=10,
        alpha=0.5,
    )


@pytest.fixture
def sample_instance(sample_config, mock_flow):
    batch_size = 2
    num_patches = 4

    dependency_matrix = torch.eye(num_patches)
    mask = None

    return AdaptiveSoftmax(
        cfg=sample_config,
        num_patches=num_patches,
        model_flow=mock_flow,
        batch_size=batch_size,
        device=torch.device("cpu"),
        dependency_matrix=dependency_matrix,
        mask=mask,
    )


def test_initialization(sample_instance):
    # assert the finished_threshold is set to 1/max_steps because it is None
    assert math.isclose(sample_instance.finished_threshold, 0.1)


def test_call_function(sample_instance):
    batch_size = 2
    device = torch.device("cpu")

    current_t = torch.ones(batch_size, 4, device=device)
    sigma_theta = torch.arange(8, dtype=torch.float32, device=device).reshape(2, 4)

    t_new, should_denoise = sample_instance(t=current_t, sigma_theta=sigma_theta)

    assert t_new.shape == (batch_size, 4)
    assert should_denoise.shape == (batch_size, 4)

    assert torch.all(t_new >= 0)
    assert torch.all(t_new < 1)  # All patches should be updated

    assert torch.all(should_denoise)  # All patches should be updated
    assert sample_instance.is_unfinished_mask.equal(
        torch.ones(batch_size, dtype=torch.bool)
    )  # All batch elements are unfinished
    assert sample_instance.current_step == 1


def test_denoising_sequence(sample_instance):
    batch_size = 2
    num_patches = 4
    device = torch.device("cpu")
    current_t = torch.ones(batch_size, num_patches, device=device)
    sigma_theta = torch.ones(batch_size, num_patches, dtype=torch.float32, device=device)
    last_unfinished_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    is_unfinished_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

    assert sample_instance.is_unfinished_mask.equal(is_unfinished_mask)

    while is_unfinished_mask.any():
        sub_unfinished_mask = is_unfinished_mask[last_unfinished_mask]

        dropped_t = current_t[~sub_unfinished_mask]
        assert torch.allclose(dropped_t, torch.tensor(0.0, device=device))
        current_t = current_t[sub_unfinished_mask]

        t_new, should_denoise = sample_instance(t=current_t, sigma_theta=sigma_theta)

        assert should_denoise.any(), "At least one patch should be updated"
        assert t_new.shape == (batch_size, num_patches)
        assert (current_t > t_new).all(), "t_new should be less than current_t"
        assert (t_new >= 0).all(), "t_new should be greater than or equal to 0"
        assert (t_new <= 1).all(), "t_new should be less than or equal to 1"
        assert sample_instance.current_step <= sample_instance.cfg.max_steps

        current_t = t_new
        last_unfinished_mask = is_unfinished_mask
        is_unfinished_mask = sample_instance.is_unfinished_mask

    assert (
        sample_instance.current_step <= sample_instance.cfg.max_steps
    ), "The current step should not exceed the max steps"
    assert sample_instance.is_unfinished_mask.equal(
        torch.zeros(batch_size, dtype=torch.bool)
    ), "All batch elements are finished"
