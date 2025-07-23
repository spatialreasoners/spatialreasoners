from jaxtyping import Float
from pytest import fixture
from torch import Tensor

from src.model.flow import Flow


class MockFlow(Flow):
    def __init__(self):
        pass

    def a(self, t):
        pass

    def a_prime(self, t):
        pass

    def b(self, t):
        pass

    def b_prime(self, t):
        pass

    def conditional_p_ut(
        self,
        mean_theta,
        z_t,
        t,
        t_star,
        alpha,
        temperature,
        sigma_theta,
        v_theta,
    ):
        pass

    def inverse_divergence_simple(
        self,
        sigma_theta: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"],
        alpha: float | int,
        threshold: Float[Tensor, "*#batch"],
        max_approx_error: float | None = None,
        num_search_steps: int | None = None,
        t_star_min: Float[Tensor, "*batch"] | None = None,
        t_star_max: Float[Tensor, "*batch"] | None = None,
    ) -> Float[Tensor, "*batch"]:
        return t / 2

    def divergence_simple(
        self,
        sigma_theta: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"],
        t_star: Float[Tensor, "*#batch"],
        alpha: float | int,
    ) -> Float[Tensor, "*batch"]:
        return t / 2


@fixture
def mock_flow():
    return MockFlow()
