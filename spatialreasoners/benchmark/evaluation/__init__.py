from pathlib import Path

from spatialreasoners.registry import Registry
from spatialreasoners.type_extensions import Stage
from spatialreasoners.variable_mapper import VariableMapper

from .evaluation import Evaluation, EvaluationCfg

_evaluation_registry = Registry(Evaluation, EvaluationCfg)


def get_evaluation(
    cfg: EvaluationCfg,
    variable_mapper: VariableMapper,
    tag: str,
    output_dir: Path,
    stage: Stage = "test"
) -> Evaluation:
    return _evaluation_registry.build(cfg, variable_mapper, tag, stage=stage, output_dir=output_dir)


register_evaluation = _evaluation_registry.register


from .frame_vis_sampling_evaluation import __all__ as frame_vis_sampling_evaluation_all
from .frame_vis_sampling_evaluation.video_pose_sampling_evaluation import (
    VideoPoseSamplingEvaluation,
    VideoPoseSamplingEvaluationCfg,
)

__all__ = frame_vis_sampling_evaluation_all + [
    "Evaluation",
    "EvaluationCfg",
    "get_evaluation",
    "register_evaluation",
    "VideoPoseSamplingEvaluation",
    "VideoPoseSamplingEvaluationCfg",
]
