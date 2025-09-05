from dataclasses import dataclass
from pathlib import Path
from math import prod
from typing import Sequence

import torch
from jaxtyping import Bool, Integer, Shaped
from torch import Tensor

from spatialreasoners.type_extensions import Stage
from spatialreasoners.variable_mapper import VariableMapper

from ... import register_evaluation
from .mnist_evaluation import MnistEvaluation, MnistEvaluationCfg


@dataclass(frozen=True, kw_only=True)
class MnistGridEvaluationCfg(MnistEvaluationCfg):
    grid_size: Sequence[int] = (9, 9)


@register_evaluation("mnist_grid", MnistGridEvaluationCfg)
class MnistGridEvaluation(MnistEvaluation[MnistGridEvaluationCfg]):
    def __init__(
        self, 
        cfg: MnistEvaluationCfg, 
        variable_mapper: VariableMapper,
        tag: str,
        output_dir: Path,
        stage: Stage = "test",
    ) -> None:
        super().__init__(cfg, variable_mapper, tag, output_dir=output_dir, stage=stage)
        self.num_cells = prod(self.cfg.grid_size)

    def _classify(
        self,
        pred: Integer[Tensor, "batch grid_height grid_width"]
    ) -> tuple[
        Bool[Tensor, "batch"],
        dict[str, Shaped[Tensor, "batch"]]
    ]:
        # -1 because we want exactly one in the end for every cnt
        cnt = torch.full(
            (pred.shape[0], self.num_cells), fill_value=-1, 
            dtype=pred.dtype, device=pred.device
        )
        pred = pred.flatten(-2) - 1
        cnt.scatter_add_(
            dim=1, 
            index=pred, 
            src=torch.ones((1,), dtype=pred.dtype, device=pred.device).expand_as(pred)
        )
        cnt.abs_()
        dist = cnt.sum(dim=1)
        label = dist == 0
        return label, {"distance": dist}
