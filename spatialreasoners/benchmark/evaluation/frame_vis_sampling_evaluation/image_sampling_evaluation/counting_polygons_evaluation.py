from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from jaxtyping import Bool, Float, Integer
from torch import Tensor

from spatialreasoners.misc.counting_polygons_classifier import get_counting_polygons_classifier
from spatialreasoners.type_extensions import BatchVariables, Stage
from spatialreasoners.variable_mapper import VariableMapper

from ... import register_evaluation
from .image_sampling_evaluation import ImageSamplingEvaluation, ImageSamplingEvaluationCfg


@dataclass(frozen=True, kw_only=True)
class CountingPolygonsEvaluationCfg(ImageSamplingEvaluationCfg):
    classifier_path: Path 
    classifier_model_base: Literal['resnet18', 'resnet50']
    are_nums_on_images: bool = False
    min_vertices: int = 3
    max_vertices: int = 7
    

@register_evaluation("counting_polygons", CountingPolygonsEvaluationCfg)
class CountingPolygonsEvaluation(ImageSamplingEvaluation[CountingPolygonsEvaluationCfg]):
    def __init__(
        self,
        cfg: CountingPolygonsEvaluationCfg,
        variable_mapper: VariableMapper,
        tag: str,
        output_dir: Path,
        stage: Stage = "test",
    ) -> None:
        super().__init__(cfg, variable_mapper, tag, output_dir=output_dir, stage=stage)
        self.classifier_path = cfg.classifier_path
        self.classifier_model = cfg.classifier_model_base
        self.are_nums_on_images = cfg.are_nums_on_images
        self.min_vertices = cfg.min_vertices
        self.max_vertices = cfg.max_vertices

    @staticmethod
    def _are_ambiguous_numbers_consistent(
        numbers_label: Integer[Tensor, "batch 2"],
        num_polygons: Integer[Tensor, "batch"],
        num_vertices: Integer[Tensor, "batch"],
    ) -> Bool[Tensor, "batch"]:
        return torch.logical_or(
            torch.logical_and(
                numbers_label[:, 0] == num_polygons, numbers_label[:, 1] == num_vertices
            ),
            torch.logical_and(
                numbers_label[:, 0] == num_vertices, numbers_label[:, 1] == num_polygons
            ),
        )

    def _get_vertices_counts(
        self, num_vertices: Integer[Tensor, "batch"], counts_range: tuple[int, int]
    ) -> dict[str, float]:
        return {
            f"relative_vertex_count_{i}": (
                (num_vertices == i).sum() / num_vertices.shape[0]
            ).item()
            for i in range(*counts_range)
        }

    def _get_polygons_counts(
        self, num_polygons: Integer[Tensor, "batch"], counts_range: tuple[int, int]
    ) -> dict[str, float]:
        return {
            f"relative_polygons_count_{i}": (
                (num_polygons == i).sum() / num_polygons.shape[0]
            ).item()
            for i in range(*counts_range)
        }

    def _is_class_label_consistent(
        self,
        class_label: Integer[Tensor, "batch"],
        num_polygons: Integer[Tensor, "batch"],
        num_vertices: Integer[Tensor, "batch"],
    ) -> Bool[Tensor, "batch"]:
        assert self.dataset.labeler is not None, "Labeler must be provided"

        generated_label = self.dataset.labeler.get_batch_labels(
            num_polygons, num_vertices
        )
        return class_label == generated_label

    @torch.no_grad()
    def _get_metrics(
        self,
        batch_variables: BatchVariables,
        label: Integer[Tensor, "batch"] | None = None, 
    ) -> dict[str, Float[Tensor, ""]] | None:
        unstructured = self.variable_mapper.variable_to_unstructured(batch_variables)
        samples = unstructured["z_t"]
        classifier = get_counting_polygons_classifier(
            model_path=self.classifier_path,
            model_base=self.classifier_model,
            device=samples.device,
            are_nums_on_images=self.are_nums_on_images,
            min_vertices=self.min_vertices,
            max_vertices=self.max_vertices,
        )

        outputs, confidences = classifier.predict(samples)

        consistency = None
        if label is not None:
            consistency = self._is_class_label_consistent(
                label, outputs["num_polygons"], outputs["num_vertices"]
            )
        elif self.are_nums_on_images:
            consistency = self._are_ambiguous_numbers_consistent(
                outputs["num_polygons_vertices"],
                outputs["num_polygons"],
                outputs["num_vertices"],
            )

        metrics = {
            "are_vertices_uniform": outputs["is_uniform"],
            **{f"{key}_confidence": value for key, value in confidences.items()},
        }
        
        if consistency is not None:
            metrics["is_class_label_consistent"] = consistency
        
        metrics = {k: v.float().mean() for k, v in metrics.items()}
        
        vertex_value_range = (classifier.min_vertices, classifier.max_vertices + 1)
        polygon_value_range = (
            classifier.min_num_polygons,
            classifier.max_num_polygons + 1,
        )

        vertex_counts = self._get_vertices_counts(
            outputs["num_vertices"], vertex_value_range
        )
        polygon_counts = self._get_polygons_counts(
            outputs["num_polygons"], polygon_value_range
        )

        metrics.update(vertex_counts)
        metrics.update(polygon_counts)

        return metrics
