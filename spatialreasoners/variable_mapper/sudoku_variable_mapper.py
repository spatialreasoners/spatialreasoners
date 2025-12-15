from dataclasses import dataclass
from typing import Sequence

import torch
from jaxtyping import Float
from torch import Tensor

from . import register_variable_mapper
from .image_variable_mapper import ImageVariableMapper, ImageVariableMapperCfg


@dataclass(kw_only=True, frozen=True)
class SudokuVariableMapperCfg(ImageVariableMapperCfg):
    variable_patch_size: int = 28
    
    def __post_init__(self) -> None:
        assert self.variable_patch_size == 28, "Only 28x28 patch size supported"


@register_variable_mapper("sudoku", SudokuVariableMapperCfg)
class SudokuVariableMapper(ImageVariableMapper[SudokuVariableMapperCfg]):
    def __init__(self, cfg: SudokuVariableMapperCfg, unstructured_sample_shape: Sequence[int]) -> None:
        super().__init__(cfg, unstructured_sample_shape)
        assert tuple(self.grid_shape) == (9, 9), "Only 9x9 patch grid supported"

    def _calculate_dependency_matrix(self) -> Float[Tensor, "num_variables num_variables"]:
        dependency_matrix = torch.zeros(81, 81, dtype=torch.bool)
        
        for i in range(81):
            r, c = self._from_full_idx(i)
            for j in range(81):
                r_, c_ = self._from_full_idx(j)
                if r == r_: #same row
                    dependency_matrix[i, j] = True

                if c == c_: #same column
                    dependency_matrix[i, j] = True

                if r // 3 == r_ // 3 and c // 3 == c_ // 3: #same subgrid
                    dependency_matrix[i, j] = True

        if self.cfg.mask_self_dependency:
            dependency_matrix = dependency_matrix.logical_xor(torch.eye(81))

        return dependency_matrix.float()