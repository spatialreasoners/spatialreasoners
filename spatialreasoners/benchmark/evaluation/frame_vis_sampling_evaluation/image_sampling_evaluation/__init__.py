from .counting_polygons_evaluation import (
    CountingPolygonsEvaluation,
    CountingPolygonsEvaluationCfg,
)
from .even_pixels_evaluation import EvenPixelsEvaluation, EvenPixelsEvaluationCfg
from .image_sampling_evaluation import (
    ImageSamplingEvaluation,
    ImageSamplingEvaluationCfg,
)
from .mnist_grid_evaluation import MnistGridEvaluation, MnistGridEvaluationCfg
from .mnist_sudoku_evaluation import MnistSudokuEvaluation, MnistSudokuEvaluationCfg

__all__ = [
    "Evaluation", "EvaluationCfg",
    "CountingPolygonsEvaluation", "CountingPolygonsEvaluationCfg",
    "EvenPixelsEvaluation", "EvenPixelsEvaluationCfg",
    "MnistGridEvaluation", "MnistGridEvaluationCfg",
    "MnistSudokuEvaluation", "MnistSudokuEvaluationCfg",
    "ImageSamplingEvaluation", "ImageSamplingEvaluationCfg",
    "get_evaluation",
    "register_evaluation"
]
