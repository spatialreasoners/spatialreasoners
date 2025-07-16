import pytest
import torch

from src.model.variable_mapper.image_variable_mapper import (
    ImageVariableMapper,
    ImageVariableMapperCfg,
)


def get_image_variable_mapper(patch_size, num_channels):
    config = ImageVariableMapperCfg(
        name="image",
        patch_size=patch_size,
        num_channels=num_channels,
    )

    return ImageVariableMapper(config)


@pytest.mark.parametrize("patch_size", [4, 10])
@pytest.mark.parametrize("num_channels", [1, 3])
def test_image_variable_mapper(patch_size, num_channels):
    variable_mapper = get_image_variable_mapper(patch_size, num_channels)

    # Test input_to_variables
    batch_size = 2
    width = patch_size * 4
    height = patch_size * 4
    x = torch.randn(batch_size, num_channels, width, height)
    variables = variable_mapper.input_to_variables(x)

    assert variables.shape == (
        batch_size,
        (width // patch_size) * (height // patch_size),
        num_channels * patch_size**2,
    )

    # Test variables_to_input
    reconstructed_x = variable_mapper.variables_to_input(variables)

    assert reconstructed_x.shape == (batch_size, num_channels, width, height)

    # Test that the input and output are the same
    assert torch.allclose(x, reconstructed_x, atol=1e-6)
