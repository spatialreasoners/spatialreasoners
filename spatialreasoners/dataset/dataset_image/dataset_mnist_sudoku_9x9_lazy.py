from dataclasses import dataclass
from typing import Sequence

import torch
from jaxtyping import Float
from torch import Tensor

from .. import register_dataset
from .dataset_mnist_grid_lazy import DatasetLazyGridCfg, DatasetMnistGridLazy


@dataclass(frozen=True, kw_only=True)
class DatasetMnistSudoku9x9LazyCfg(DatasetLazyGridCfg):
    given_cells_range: Sequence[int] = (0, 80)


@register_dataset("mnist_sudoku_lazy", DatasetMnistSudoku9x9LazyCfg)
class DatasetMnistSudoku9x9Lazy(DatasetMnistGridLazy[DatasetMnistSudoku9x9LazyCfg]):
    cell_size = (28, 28)
    grid_size = (9, 9)
    
    def get_dependency_matrix(
        self,
        grid_shape: tuple[int, int]
    ) -> Float[Tensor, "num_patches num_patches"] | None:
        # assert grid_shape == (9, 9), "Only 9x9 patch grid supported"  # TODO do not call dependency matrix if not necessary
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
