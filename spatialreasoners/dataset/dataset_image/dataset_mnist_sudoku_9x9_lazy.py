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