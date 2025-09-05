from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from jaxtyping import Float
from torch import Tensor

from spatialreasoners.denoising_model.flow import Flow


@dataclass
class LossCfg:
    weight: float | int = 1
    apply_after_step: int = 0


T = TypeVar("T", bound=LossCfg)


class Loss(Generic[T], ABC):
    def __init__(
        self,
        cfg: T,
        flow: Flow
    ) -> None:
        self.cfg = cfg
        self.flow = flow

    def is_active(self, global_step: int) -> bool:
        return self.cfg.apply_after_step <= global_step

    @abstractmethod
    def unweighted_loss(
        self,
        t: Float[Tensor, "*#batch"],
        eps: Float[Tensor, "*batch"],
        x: Float[Tensor, "*batch"],
        z_t: Float[Tensor, "*batch"],
        mean_target: Float[Tensor, "*batch"],
        mean_pred: Float[Tensor, "*batch"],
        v_pred: Float[Tensor, "*#batch"] | None,
        sigma_pred: Float[Tensor, "*#batch"] | None,
    ) -> Float[Tensor, "*batch"]:
        pass

    def __call__(
        self,
        t: Float[Tensor, "*#batch"],
        eps: Float[Tensor, "*batch"],
        x: Float[Tensor, "*batch"],
        z_t: Float[Tensor, "*batch"],
        mean_target: Float[Tensor, "*batch"],
        mean_pred: Float[Tensor, "*batch"],
        v_pred: Float[Tensor, "*#batch"] | None,
        sigma_pred: Float[Tensor, "*#batch"] | None,
        global_step: int = 0
    ) -> Float[Tensor, "*batch"] | float | int:
        # Before the specified step, don't apply the loss.
        if not self.is_active(global_step) or self.cfg.weight == 0:
            return 0
        loss = self.unweighted_loss(
            t,
            eps,
            x,
            z_t,
            mean_target,
            mean_pred,
            v_pred,
            sigma_pred,
        )
        return self.cfg.weight * loss
