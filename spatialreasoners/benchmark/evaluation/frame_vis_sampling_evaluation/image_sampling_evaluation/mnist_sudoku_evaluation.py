from dataclasses import dataclass
from pathlib import Path

import torch
from jaxtyping import Bool, Integer, Shaped
from torch import Tensor

from spatialreasoners.type_extensions import Stage
from spatialreasoners.variable_mapper import SudokuVariableMapper

from ... import register_evaluation
from .mnist_evaluation import MnistEvaluation, MnistEvaluationCfg


@dataclass(frozen=True, kw_only=True)
class MnistSudokuEvaluationCfg(MnistEvaluationCfg):
    pass


@register_evaluation("mnist_sudoku", MnistSudokuEvaluationCfg)
class MnistSudokuEvaluation(MnistEvaluation[MnistSudokuEvaluationCfg]):
    def __init__(
        self, 
        cfg: MnistEvaluationCfg, 
        variable_mapper: SudokuVariableMapper,
        tag: str,
        output_dir: Path,
        stage: Stage = "test",
    ) -> None:
        super().__init__(cfg, variable_mapper, tag, output_dir=output_dir, stage=stage)
    
    def _classify(
        self,
        pred: Integer[Tensor, "batch grid_size grid_size"]
    ) -> tuple[
        Bool[Tensor, "batch"],
        dict[str, Shaped[Tensor, "batch"]]
    ]:
        batch_size, grid_size = pred.shape[:2]
        sub_grid_size = round(grid_size ** 0.5)
        dtype, device = pred.dtype, pred.device
        pred = pred - 1 # Shift [1, 9] to [0, 8] for indices
        ones = torch.ones((1,), dtype=dtype, device=device).expand_as(pred)
        dist = torch.zeros((batch_size,), dtype=dtype, device=device)
        for dim in range(1, 3):
            cnt = torch.full_like(pred, fill_value=-1)
            cnt.scatter_add_(dim=dim, index=pred, src=ones)
            dist.add_(cnt.abs_().sum(dim=(1, 2)))
        # Subgrids
        grids = pred.unfold(1, sub_grid_size, sub_grid_size)\
            .unfold(2, sub_grid_size, sub_grid_size).reshape(-1, grid_size, grid_size)
        cnt = torch.full_like(grids, fill_value=-1)
        cnt.scatter_add_(dim=dim, index=grids, src=ones)
        dist.add_(cnt.abs_().sum(dim=(1, 2)))
        label = dist == 0
        return label, {"distance": dist}
