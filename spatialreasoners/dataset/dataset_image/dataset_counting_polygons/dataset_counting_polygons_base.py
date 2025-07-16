import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar

import numpy as np
from jaxtyping import Float, Int
from PIL import Image, ImageDraw

from spatialreasoners.type_extensions import ConditioningCfg, Stage

from ..dataset_image import DatasetImage, DatasetImageCfg
from ..type_extensions import ImageExample
from .font_cache import FontCache
from .labelers import get_labeler


@dataclass(frozen=True, kw_only=True)
class DatasetCountingPolygonsCfg(DatasetImageCfg):
    labeler_name: Literal["explicit", "ambiguous"] | None = None
    are_nums_on_images: bool = False
    supersampling_image_size: Sequence[int] = (512, 512)
    min_vertices: int = 3
    max_vertices: int = 7
    font_name: str = "Roboto-Regular.ttf"
    mismatched_numbers: bool = False  # True only for the classifier
    allow_nonuinform_vertices: bool = (
        False  # Different vertex numbers for each polygon - only for the classifier
    )
    use_stars: bool = False # Render stars instead of polygons
    star_radius_ratio: float = 0.1 # Ratio of inner radius to outer radius
    counting_polygons_classifier_path: str = (
        ""  # Path to the counting objects classifier
    )
    counting_polygons_classifier_model_base: Literal["resnet18", "resnet50"] = "resnet50"


T = TypeVar("T", bound=DatasetCountingPolygonsCfg)


class DatasetCountingPolygonsBase(DatasetImage[T], ABC):
    training_set_ratio: float = 0.95
    circle_images_per_num_circles: int = 100_000
    circle_num_variants: int = 9
    circle_positions_file_name: str = "circle_position_radius.npy"

    @abstractmethod
    def _get_base_image(self, base_image_idx) -> Image.Image:
        pass

    @property
    @abstractmethod
    def _num_available(self) -> int:
        pass

    @abstractmethod
    def _split_idx(self, idx) -> tuple[int, int, int | None]:
        pass

    def __init__(
        self,
        cfg: DatasetCountingPolygonsCfg,
        conditioning_cfg: ConditioningCfg,
        stage: Stage,
    ) -> None:
        super().__init__(cfg=cfg, conditioning_cfg=conditioning_cfg, stage=stage)

        if self.cfg.allow_nonuinform_vertices:
            assert (
                self.cfg.mismatched_numbers
            ), "Mismatched numbers must be enabled to allow nonuniform vertices"

        self.root_path = Path(self.cfg.root)
        self.min_circle_num = 3 if self.cfg.are_nums_on_images else 1
        self.circle_positions = self._load_circle_positions()

        self.font_cache = FontCache(self.root_path / "Roboto-Regular.ttf")
        self.labeler = (
            get_labeler(
                labeler_name=cfg.labeler_name,
                min_vertices=cfg.min_vertices,
                max_vertices=cfg.max_vertices,
            )
            if cfg.labeler_name
            else None
        )

        assert self.cfg.supersampling_image_size[0] >= self.cfg.image_resolution[0]
        assert self.cfg.supersampling_image_size[1] >= self.cfg.image_resolution[1]

        self.circle_xyr_scaling_factor = np.array(
            [
                self.cfg.supersampling_image_size[0],  # x
                self.cfg.supersampling_image_size[1],  # y
                min(self.cfg.supersampling_image_size),  # radius
            ]
        )[
            np.newaxis, ...
        ]  # shape: (1, 3)
        
        self._draw_object = (
            self._draw_stars if self.cfg.use_stars else self._draw_polygons
        )

    @property
    def is_deterministic(self) -> bool:
        return self.stage != "train"

    @property
    def _training_positions_per_circles_num(self) -> int:
        return int(self.training_set_ratio * self.circle_images_per_num_circles)

    def _load_circle_positions(
        self,
    ) -> dict[int, Float[np.ndarray, "num_images num_circles_per_image 3"]]:
        # The data is in the form of a numpy array with shape (num_circles, num_circles, 3)
        # where the last dimension is the x, y, and radius of the circle
        circle_pos_radius = np.load(
            self.root_path / self.circle_positions_file_name, allow_pickle=True
        ).item()

        possible_num_circle_nums = np.arange(
            self.min_circle_num, self.min_circle_num + self.circle_num_variants
        )  # 1-9 or 3-11

        return {
            num_circles: (
                data[: self._training_positions_per_circles_num]
                if self.stage == "train"
                else data[self._training_positions_per_circles_num :]
            )
            for num_circles, data in circle_pos_radius.items()
            if num_circles in possible_num_circle_nums
        }

    def split_circles_idx(self, idx) -> tuple[int, int]:
        num_circles_idx = (
            idx // self._num_positions_per_num_circles + self.min_circle_num
        )
        circles_image_idx = idx % self._num_positions_per_num_circles

        assert circles_image_idx < self._num_overlay_images
        return num_circles_idx, circles_image_idx

    def _get_circle_xyr(
        self, num_circles_idx: int, circles_image_idx: int
    ) -> Float[np.ndarray, "num_circles_per_image 3"]:
        image_circles = self.circle_positions[num_circles_idx][
            circles_image_idx
        ]  # shape: (num_circles, 3)

        return image_circles * self.circle_xyr_scaling_factor

    @abstractmethod
    def _get_color(
        self, rng: np.random.Generator | None, base_image: Image.Image
    ) -> str | tuple[int, int, int]:
        pass

    @staticmethod
    def _get_unit_polygon_vertices(
        points_on_circle: int | np.int64, angle_offset: Float[np.ndarray, "num_polygons"]
    ) -> Float[np.ndarray, "num_polygons points_on_circle 2"]:
        base = np.arange(points_on_circle) / points_on_circle * 2 * np.pi
        angles = base[np.newaxis, ...] + np.expand_dims(angle_offset, 1)

        x = np.cos(angles)
        y = np.sin(angles)

        return np.stack([x, y], axis=-1)

    @staticmethod
    def random_choice(rng: np.random.Generator | None, *args, **kwargs):
        if rng is not None:
            return rng.choice(*args, **kwargs)
        return np.random.choice(*args, **kwargs)

    @staticmethod
    def random_integers(rng: np.random.Generator | None, *args, **kwargs):
        if rng is not None:
            return rng.integers(*args, **kwargs).item()
        return np.random.randint(*args, **kwargs)

    @staticmethod
    def random_uniform(rng: np.random.Generator | None, *args, **kwargs):
        if rng is not None:
            return rng.uniform(*args, **kwargs)
        return np.random.uniform(*args, **kwargs)

    def _draw_polygons(
        self,
        draw: ImageDraw.ImageDraw,
        vertices_per_object: Int[np.ndarray, "num_polygons"],
        polygons_xyr: Float[np.ndarray, "num_polygons 3"],
        angle_offset: Float[np.ndarray, "num_polygons"],
        color: str | tuple[int, int, int],
    ) -> None:
        drawn_elements = 0

        for vertices_num in np.unique(vertices_per_object):
            vertices_mask = vertices_per_object == vertices_num

            group_polygons_xyr = polygons_xyr[vertices_mask]
            group_angle_offset = angle_offset[vertices_mask]

            batch_polygons_vertices = self._get_unit_polygon_vertices(
                vertices_num, group_angle_offset
            )

            radii = group_polygons_xyr[:, 2]
            centers = group_polygons_xyr[:, :2]

            batch_polygons_vertices *= radii[
                :, np.newaxis, np.newaxis
            ]  # scale by corresponding radius
            batch_polygons_vertices += np.expand_dims(
                centers, 1
            )  # translate to corresponding center

            for polygon_vertices in batch_polygons_vertices:
                polygon_vertices = [
                    (x, y) for x, y in polygon_vertices
                ]  # convert to list of tuples
                draw.polygon(polygon_vertices, fill=color)
                drawn_elements += 1

        assert drawn_elements == len(polygons_xyr), "Not all polygons were drawn"
        
        
    def _draw_stars(
        self,
        draw: ImageDraw.ImageDraw,
        vertices_per_object: Int[np.ndarray, "num_polygons"],
        polygons_xyr: Float[np.ndarray, "num_polygons 3"],
        angle_offset: Float[np.ndarray, "num_polygons"],
        color: str | tuple[int, int, int],
    ) -> None:

        for vertex_num, xyr, ang_offset in zip(
            vertices_per_object, polygons_xyr, angle_offset
        ):
            center = xyr[:2]
            outer_radius = xyr[2]
            inner_radius = self.cfg.star_radius_ratio * outer_radius

            self.draw_star(
                draw=draw,
                center=center,
                outer_radius=outer_radius.item(),
                inner_radius=inner_radius.item(),
                angle_offset=ang_offset.item(),
                points=vertex_num.item(),
                color=color,
            )
        
    @staticmethod
    def draw_star(
        draw: ImageDraw.ImageDraw,
        center: Float[np.ndarray, "2"],
        outer_radius: float,
        inner_radius: float,
        angle_offset: float,
        points: int,
        color: str | tuple[int, int, int] = "black",
    ):
        """
        Draws a star using PIL.ImageDraw.

        :param draw: The ImageDraw object
        :param center: Tuple (x, y) for the center of the star
        :param outer_radius: Radius of the outer points of the star
        :param inner_radius: Radius of the inner points of the star
        :param points: Number of star points (default is 5)
        :param color: Fill color of the star
        """
        star_points = []
        angle = math.pi / points  # Angle between outer and inner points

        for i in range(2 * points):  # Loop through outer and inner points
            r = outer_radius if i % 2 == 0 else inner_radius
            theta = i * angle + angle_offset  # Current angle
            x = center[0] + r * math.cos(theta)
            y = center[1] + r * math.sin(theta)
            star_points.append((x, y))

        draw.polygon(star_points, fill=color)

    def _draw_numbers(
        self,
        draw: ImageDraw.ImageDraw,
        numbers_xyr: Float[np.ndarray, "2 3"],
        numbers: Int[np.ndarray, "2"],
        color: str | tuple[int, int, int],
    ) -> None:
        for (x, y, radius), number in zip(numbers_xyr, numbers):
            font = self.font_cache.get_font(int(radius))
            draw.text((x, y), str(number), font=font, fill=color)
            
    def _get_image_with_objects(
        self,
        vertices_per_polygon: Int[np.ndarray, "num_polygons"],
        polygons_xyr: Float[np.ndarray, "num_polygons 3"],
        angle_offset: Float[np.ndarray, "num_polygons"],
        color: str | tuple[int, int, int],
        numbers_xyr: Float[np.ndarray, "2 3"] | None,
        numbers: Int[np.ndarray, "2"] | None,
    ) -> Image.Image:
        image = Image.new("RGBA", self.cfg.supersampling_image_size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(image)

        self._draw_object(
            draw, vertices_per_polygon, polygons_xyr, angle_offset, color
        )

        if numbers_xyr is not None:
            self._draw_numbers(draw, numbers_xyr, numbers, color)
            
        resized_image = image.resize(self.cfg.image_resolution, resample=Image.BICUBIC)
        return resized_image

    def _get_overlay_image_w_label(
        self,
        num_circles_idx: int,
        circles_image_idx: int,
        full_idx: int,
        base_image: Image.Image,
    ) -> tuple[Image.Image, int | None]:

        circle_xyr = self._get_circle_xyr(num_circles_idx, circles_image_idx)
        num_circles = circle_xyr.shape[0]

        # Randomize the order of the circles
        rng = None
        if self.is_deterministic:
            rng = np.random.default_rng(full_idx)
        else:
            circle_xyr = circle_xyr[np.random.permutation(num_circles)]

        if self.cfg.are_nums_on_images:
            numbers_xyr = circle_xyr[-2:]
            polygons_xyr = circle_xyr[:-2]
            num_polygons = num_circles - 2

        else:
            numbers_xyr = None
            polygons_xyr = circle_xyr
            num_polygons = num_circles

        if all([
            self.cfg.allow_nonuinform_vertices,  # Only if the option is enabled
            (num_polygons > 1),  # one polygon -> num of vertices is always uniform
            self.random_choice(rng, [True, False]),  # Randomly choose if uniform
        ]):
            vertices_per_polygon = self.random_choice(
                rng,
                np.arange(self.cfg.min_vertices, self.cfg.max_vertices + 1),
                num_polygons,
                replace=True,
            )
        else:
            # set the number of vertices to num_vertices
            num_vertices = self.random_integers(
                rng, self.cfg.min_vertices, self.cfg.max_vertices + 1
            )
            vertices_per_polygon = np.full(num_polygons, num_vertices)

        are_uniform_vertices = (vertices_per_polygon[0] == vertices_per_polygon).all()

        if self.cfg.mismatched_numbers:
            # Randomize the numbers that will be drawn on the image (and in the label)
            conditioning_numbers = {
                "num_polygons": self.random_integers(rng, 1, 10),
                "num_vertices": self.random_integers(
                    rng, self.cfg.min_vertices, self.cfg.max_vertices + 1
                ),
            }
        else:
            conditioning_numbers = {
                "num_polygons": num_polygons,
                "num_vertices": vertices_per_polygon[0].item(),
            }

        numbers_label = None  # Default value
        if self.labeler is not None:
            numbers_label = self.labeler.get_label(**conditioning_numbers)

        if self.cfg.mismatched_numbers:
            label = {
                "num_polygons": num_polygons - 1,  # map 1-9 to 0-8 as class labels
                "num_vertices": int( # map 3-7 to 0-4 as class labels
                    vertices_per_polygon[0] - self.cfg.min_vertices
                ),  
                "is_uniform": int(are_uniform_vertices),
            }

            if numbers_label is not None:
                label["numbers_label"] = numbers_label

        else:
            label = numbers_label
            
        angle_offset = self.random_uniform(rng, 0, 2 * np.pi, size=num_polygons)
        color = self._get_color(rng, base_image)

        resized_image = self._get_image_with_objects(
            vertices_per_polygon=vertices_per_polygon,
            polygons_xyr=polygons_xyr,
            angle_offset=angle_offset,
            color=color,
            numbers_xyr=numbers_xyr,
            numbers=np.fromiter(conditioning_numbers.values(), dtype=int),
        )
        
        return resized_image, label

    @property
    def num_classes(self) -> int | dict[str, int]:
        if not self.cfg.mismatched_numbers:
            return self.labeler.num_classes if self.labeler else 0

        class_counts = {
            "num_polygons": self.circle_num_variants,
            "num_vertices": self.cfg.max_vertices - self.cfg.min_vertices + 1,
        }

        if self.labeler:
            class_counts["numbers_label"] = self.labeler.num_classes

        if self.cfg.allow_nonuinform_vertices:
            class_counts["is_uniform"] = 2

        return class_counts

    def _load(self, idx: int) -> ImageExample:
        num_circles_idx, circles_image_idx, base_image_idx = self._split_idx(idx)

        base_image = self._get_base_image(base_image_idx)

        overlay_image, label = self._get_overlay_image_w_label(
            num_circles_idx, circles_image_idx, full_idx=idx, base_image=base_image
        )

        image = Image.alpha_composite(base_image, overlay_image)
        image = image.convert("RGB")

        return {"image": image} if label is None else {"image": image, "label": label}

    @property
    def _num_positions_per_num_circles(self) -> int:
        return (
            self._training_positions_per_circles_num
            if self.stage == "train"
            else self.circle_images_per_num_circles
            - self._training_positions_per_circles_num
        )

    @property
    def _num_overlay_images(self) -> int:
        """Calculate the number of overlay images based on the number of circle xyrs
        This doesn't include the colors and the number of vertices"""
        return (
            self._num_positions_per_num_circles * self.circle_num_variants
        )  # 10 possible number of circles
