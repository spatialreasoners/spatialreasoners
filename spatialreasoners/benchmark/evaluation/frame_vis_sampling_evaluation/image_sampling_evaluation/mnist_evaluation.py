from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import torch
from jaxtyping import Bool, Float, Integer, Shaped
from torch import Tensor

from spatialreasoners.global_cfg import get_mnist_classifier_path
from spatialreasoners.misc.mnist_classifier import MNISTClassifier, get_classifier
from spatialreasoners.type_extensions import BatchVariables, Stage
from spatialreasoners.variable_mapper.sudoku_variable_mapper import SudokuVariableMapper

from .image_sampling_evaluation import (
    ImageSamplingEvaluation,
    ImageSamplingEvaluationCfg,
)


@dataclass(frozen=True, kw_only=True)
class MnistEvaluationCfg(ImageSamplingEvaluationCfg):
    num_fill: int | Sequence[int] | None = None


T = TypeVar("T", bound=MnistEvaluationCfg)


class MnistEvaluation(ImageSamplingEvaluation[T], ABC):
    force_load_from_dataset=True
    grid_size = (9, 9)
    
    def __init__(
        self,
        cfg: MnistEvaluationCfg,
        variable_mapper: SudokuVariableMapper,
        tag: str,
        output_dir: Path,
        stage: Stage = "test",
    ) -> None:
        super().__init__(cfg, variable_mapper, tag, output_dir=output_dir, stage=stage)

    @torch.no_grad()
    def _discretize(
        self,
        classifier: MNISTClassifier,
        samples: Float[Tensor, "batch 1 height width"]
    ) -> Integer[Tensor, "batch grid_height grid_width"]:
        batch_size = samples.shape[0]
        tile_shape = tuple(s // g for s, g in zip(samples.shape[-2:], self.grid_size))
        tiles = samples.unfold(2, tile_shape[0], tile_shape[0])\
            .unfold(3, tile_shape[1], tile_shape[1]).reshape(-1, 1, *tile_shape)
        logits: Float[Tensor, "batch 10"] = classifier(tiles)
        idx = torch.topk(logits, k=2, dim=1).indices
        pred = idx[:, 0]
        # Replace zero predictions with second most probable number
        zero_mask = pred == 0
        pred[zero_mask] = idx[zero_mask, 1]
        pred = pred.reshape(batch_size, *self.grid_size)
        return pred
    
    @abstractmethod
    def _classify(
        self,
        pred: Integer[Tensor, "batch grid_height grid_width"]
    ) -> tuple[
        Bool[Tensor, "batch"],
        dict[str, Shaped[Tensor, "batch"]]
    ]:
        pass
    
    @torch.no_grad()
    def _get_metrics(
        self,
        batch_variables: BatchVariables,
        label: Integer[Tensor, "batch"] | None = None, 
    ) -> dict[str, Float[Tensor, ""]] | None:
        unstructured = self.variable_mapper.variables_to_unstructured(batch_variables)
        samples = unstructured["z_t"]
        
        classifier = get_classifier(get_mnist_classifier_path(), samples.device)
        discrete = self._discretize(classifier, samples)
        labels, metrics = self._classify(discrete)
        metrics = {k: v.float().mean() for k, v in metrics.items()}
        metrics["accuracy"] = labels.float().mean()
        return metrics
