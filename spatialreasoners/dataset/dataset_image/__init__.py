
from .dataset_image import DatasetImage, DatasetImageCfg # isort: skip

from .dataset_celeba import DatasetCelebA, DatasetCelebACfg
from .dataset_cifar10 import DatasetCifar10, DatasetCifar10Cfg
from .dataset_counting_polygons import (
    DatasetCountingPolygonsBlank,
    DatasetCountingPolygonsBlankCfg,
    DatasetCountingPolygonsFFHQ,
    DatasetCountingPolygonsFFHQCfg,
)
from .dataset_even_pixels import DatasetEvenPixels, DatasetEvenPixelsCfg
from .dataset_ffhq import DatasetFFHQ, DatasetFFHQCfg
from .dataset_grid import DatasetGrid

from .dataset_imagenet import DatasetImageNet, DatasetImageNetCfg
from .dataset_mnist import DatasetMnist, DatasetMnistCfg
from .dataset_mnist_sudoku_9x9_lazy import (
    DatasetMnistSudoku9x9Lazy,
    DatasetMnistSudoku9x9LazyCfg,
)

__all__ = [
    "DatasetImage", "DatasetImageCfg",
    "DatasetCelebA", "DatasetCelebACfg",
    "DatasetCifar10", "DatasetCifar10Cfg",
    "DatasetCountingPolygonsBlank", "DatasetCountingPolygonsBlankCfg",
    "DatasetCountingPolygonsFFHQ", "DatasetCountingPolygonsFFHQCfg",
    "DatasetEvenPixels", "DatasetEvenPixelsCfg",
    "DatasetFFHQ", "DatasetFFHQCfg",
    "DatasetGrid",
    "DatasetImageNet", "DatasetImageNetCfg",
    "DatasetMnist", "DatasetMnistCfg",
    "DatasetMnistSudoku9x9Lazy", "DatasetMnistSudoku9x9LazyCfg",
]
