from abc import ABC, abstractmethod
from functools import cache
from typing import Dict, Literal

import torch
from jaxtyping import Int
from torch import Tensor


class DatasetCountingPolygonsLabeler(ABC):
    name: Literal["labeler"] = "labeler"

    def __init__(self, min_vertices: int, max_vertices: int):
        self.min_vertices = min_vertices
        self.max_vertices = max_vertices

    @abstractmethod
    def get_label(self, num_polygons: int, num_vertices: int) -> int:
        pass

    @abstractmethod
    def get_batch_labels(
        self, num_polygons: Int[Tensor, "batch"], num_vertices: Int[Tensor, "batch"]
    ) -> Int[Tensor, "batch"]:
        pass

    @abstractmethod
    def label_to_num_polygons_vertices(
        self, label: Int[Tensor, "batch"]
    ) -> Int[Tensor, "batch 2"]:
        pass

    @property
    @abstractmethod
    def num_classes(self) -> int:
        pass


class DatasetCountingPolygonsAmbiguousLabeler(DatasetCountingPolygonsLabeler):
    name: Literal["ambiguous"] = "ambiguous"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.label_id_to_pair, self.label_pair_to_id = self._load_class_labels()

    def _load_class_labels(
        self,
    ) -> tuple[Int[Tensor, "num_pairs 2"], Dict[tuple, int]]:

        # It's slow but it's only done once
        pairs_in_list = set()
        pairs = []
        for num_vertices in range(self.min_vertices, self.max_vertices + 1):
            for num_polygons in range(1, 10):
                low = min(num_vertices, num_polygons)
                high = max(num_vertices, num_polygons)

                value_pair = (low, high)

                if value_pair not in pairs_in_list:
                    pairs_in_list.add(value_pair)
                    pairs.append(value_pair)

        id_to_pair = torch.tensor(pairs, dtype=torch.int32)
        pair_to_id = {tuple(pair): i for i, pair in enumerate(pairs)}

        return id_to_pair, pair_to_id

    def get_label(self, num_polygons: int, num_vertices: int) -> int:
        low = min(num_polygons, num_vertices)
        high = max(num_polygons, num_vertices)

        return self.label_pair_to_id[(low, high)]

    def get_batch_labels(
        self, num_polygons: Int[Tensor, "batch"], num_vertices: Int[Tensor, "batch"]
    ) -> Int[Tensor, "batch"]:
        low = torch.min(num_polygons, num_vertices)
        high = torch.max(num_polygons, num_vertices)

        return torch.tensor(
            [
                self.label_pair_to_id[(low[i].item(), high[i].item())]
                for i in range(num_polygons.shape[0])
            ],
            dtype=torch.int32,
            device=num_polygons.device,
        )

    def label_to_num_polygons_vertices(
        self, label: Int[Tensor, "batch"]
    ) -> Int[Tensor, "batch 2"]:
        if (
            self.label_id_to_pair.device != label.device
        ):  # Except for first, should be on the same device
            self.label_id_to_pair = self.label_id_to_pair.to(label.device)

        return self.label_id_to_pair[label]

    @property
    def num_classes(self) -> int:
        return len(self.label_id_to_pair)


class DatasetCountingPolygonsExplicitLabeler(DatasetCountingPolygonsLabeler):
    name: Literal["explicit"] = "explicit"

    def get_label(self, num_polygons: int, num_vertices: int) -> int:
        polygons_label = num_polygons - 1  # at least 1 polygon
        vertices_label = num_vertices - self.min_vertices

        return polygons_label * self.num_vertex_classes + vertices_label

    def get_batch_labels(
        self, num_polygons: Int[Tensor, "batch"], num_vertices: Int[Tensor, "batch"]
    ) -> Int[Tensor, "batch"]:
        polygons_label = num_polygons - 1  # at least 1 polygon
        vertices_label = num_vertices - self.min_vertices

        return polygons_label * self.num_vertex_classes + vertices_label

    def label_to_num_polygons_vertices(
        self, label: Int[Tensor, "batch"]
    ) -> Int[Tensor, "batch 2"]:
        polygons_num = label // self.num_vertex_classes + 1
        vertices_num = label % self.num_vertex_classes + self.min_vertices

        return torch.stack([polygons_num, vertices_num], dim=-1)

    @property
    @cache
    def num_vertex_classes(self) -> int:
        return self.max_vertices - self.min_vertices + 1

    @property
    def num_classes(self) -> int:
        return 9 * self.num_vertex_classes


def get_labeler(labeler_name: str, *args, **kwargs) -> DatasetCountingPolygonsLabeler:
    for labeler in DatasetCountingPolygonsLabeler.__subclasses__():
        if labeler.name == labeler_name:
            return labeler(*args, **kwargs)

    raise ValueError(f"Invalid labeler name: {labeler_name}")
