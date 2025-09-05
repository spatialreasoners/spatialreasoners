from collections.abc import Sequence

import torch

from spatialreasoners.registry import Registry

from .variable_mapper import VariableMapper, VariableMapperCfg

_variable_mapper_registry = Registry(VariableMapper, VariableMapperCfg)

def get_variable_mapper(
    cfg: VariableMapperCfg,
    unstructured_sample_shape: Sequence[int],
) -> VariableMapper:
    return _variable_mapper_registry.build(cfg, unstructured_sample_shape=unstructured_sample_shape)


register_variable_mapper = _variable_mapper_registry.register

from .image_variable_mapper import ImageVariableMapper, ImageVariableMapperCfg
from .sudoku_variable_mapper import SudokuVariableMapper, SudokuVariableMapperCfg
from .video_pose_variable_mapper import (
    VideoPoseVariableMapper,
    VideoPoseVariableMapperCfg,
)

__all__ = [
    "VariableMapper", "VariableMapperCfg",
    "ImageVariableMapper", "ImageVariableMapperCfg",
    "SudokuVariableMapper", "SudokuVariableMapperCfg",
    "VideoPoseVariableMapper", "VideoPoseVariableMapperCfg",
    "get_variable_mapper",
    "register_variable_mapper"
]