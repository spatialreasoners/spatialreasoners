from functools import cache
from pathlib import Path
from typing import Dict, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from torchvision.models import resnet18, resnet50

from spatialreasoners.dataset.dataset_image.dataset_counting_polygons.labelers import (
    DatasetCountingPolygonsAmbiguousLabeler,
)


class MultiHeadLayer(nn.Module):
    def __init__(self, in_features: int, num_classes: Dict[str, int]):
        super().__init__()

        self.heads = nn.ModuleDict(
            {name: nn.Linear(in_features, num_classes[name]) for name in num_classes}
        )

    def forward(self, x):
        return {name: head(x) for name, head in self.heads.items()}


class CountingPolygonsClassifierWrapper:
    def __init__(
        self,
        model_base: Literal["resnet18", "resnet50"],
        model_path: str | Path,
        device: str | torch.device,
        min_vertices: int = 3,
        max_vertices: int = 7,
        min_num_polygons: int = 1,
        max_num_polygons: int = 9,
        are_nums_on_images: bool = True,
    ):
        self.device = device
        self.min_vertices = min_vertices
        self.max_vertices = max_vertices
        self.min_num_polygons = min_num_polygons
        self.max_num_polygons = max_num_polygons
        self.are_nums_on_images = are_nums_on_images

        self.model_base = model_base
        self.model_path = model_path

        self.num_classes = {
            "num_polygons": self.max_num_polygons - self.min_num_polygons + 1,
            "num_vertices": self.max_vertices - self.min_vertices + 1,
            "is_uniform": 2,
        }

        if self.are_nums_on_images:
            self.labeler = DatasetCountingPolygonsAmbiguousLabeler(
                min_vertices=self.min_vertices, max_vertices=self.max_vertices
            )

            self.num_classes["numbers_label"] = self.labeler.num_classes

        self.model = self.load_model()

    def load_model(self) -> nn.Module:
        model = (
            resnet18(pretrained=False)
            if self.model_base == "resnet18"
            else resnet50(pretrained=False)
        )
        model.fc = MultiHeadLayer(model.fc.in_features, self.num_classes)

        model.load_state_dict(
            torch.load(self.model_path, map_location=self.device, weights_only=True),
            strict=True,
        )
        model.to(self.device)
        model.eval()
        return model

    def predict(
        self, images: Float[Tensor, "batch 3 height width"]
    ) -> tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        with torch.no_grad():
            outputs = self.model(images)

        selected_labels = {}
        confidences = {}

        for key in outputs:
            softmaxed = F.softmax(outputs[key], dim=1)
            selected_labels[key] = softmaxed.argmax(dim=1)
            confidences[key] = softmaxed.max(dim=1).values

        outputs = {
            "is_uniform": selected_labels["is_uniform"].bool(),  # 1 is True, 0 is False
            "num_polygons": selected_labels["num_polygons"] + self.min_num_polygons,
            "num_vertices": selected_labels["num_vertices"] + self.min_vertices,
        }

        if self.are_nums_on_images:
            outputs["num_polygons_vertices"] = (
                self.labeler.label_to_num_polygons_vertices(
                    selected_labels["numbers_label"]
                )
            )

        return outputs, confidences


@cache
def get_counting_polygons_classifier(
    *args, **kwargs
) -> CountingPolygonsClassifierWrapper:
    return CountingPolygonsClassifierWrapper(*args, **kwargs)
