from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor

from . import Flow, FlowCfg, register_flow
from .diagonal_gaussian import DiagonalGaussian


@dataclass
class RectifiedFlowCfg(FlowCfg):
    pass


@register_flow("rectified", RectifiedFlowCfg)
class RectifiedFlow(Flow[RectifiedFlowCfg]):
    def a(
        self, 
        t: Float[Tensor, "*batch"]
    ) -> Float[Tensor, "*batch"]:
        """Define data weight a_t for t"""
        return 1-t

    def a_prime(
        self,
        t: Float[Tensor, "*batch"]
    ) -> Float[Tensor, "*batch"]:
        """Define derivative of data weight a_t for t"""
        return torch.full(
            (1,), fill_value=-1.0, device=t.device, dtype=t.dtype
        ).expand_as(t)

    def b(
        self, 
        t: Float[Tensor, "*batch"]
    ) -> Float[Tensor, "*batch"]:
        """Define noise weight b_t for t"""
        return t

    def b_prime(
        self, 
        t: Float[Tensor, "*batch"]
    ) -> Float[Tensor, "*batch"]:
        """Define derivative of noise weight b_t for t"""
        return torch.ones(
            (1,), device=t.device, dtype=t.dtype
        ).expand_as(t)
    
    def get_eps_from_ut_and_x(
        self,
        ut: Float[Tensor, "*batch"],
        x: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        return ut + x

    def get_ut_from_eps_and_x(
        self,
        eps: Float[Tensor, "*batch"],
        x: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        return eps - x      

    def get_ut_from_eps_and_zt(
        self,
        eps: Float[Tensor, "*batch"],
        zt: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        # a_prime / a * zt - (a_prime * b / a - b_prime) * eps
        return 1 / self.a(t) * (eps - zt)
    
    def get_x_from_eps_and_ut(
        self,
        eps: Float[Tensor, "*batch"],
        ut: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        return eps - ut

    def get_x_from_ut_and_zt(
        self,
        ut: Float[Tensor, "*batch"],
        zt: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
         return zt - (1 - self.a(t)) * ut

    def conditional_p_ut(
        self,
        mean_theta: Float[Tensor, "*batch"],
        z_t: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"],
        t_star: Float[Tensor, "*#batch"],
        alpha: float | int,
        temperature: float | int = 1,
        sigma_theta: Float[Tensor, "*#batch"] | None = None,
        v_theta: Float[Tensor, "*#batch"] | None = None
    ) -> DiagonalGaussian:
        # mean_theta and sigma_theta are the mean and standard deviation of ut
        gamma_p = self.gamma(t, t_star, alpha)
        a_t, a_t_star = self.a(t), self.a(t_star)
        mean = a_t_star * (z_t - t*mean_theta) \
            + (z_t + a_t*mean_theta) * gamma_p
        var = self.sigma(t, t_star, alpha, v_theta) ** 2
        if sigma_theta is not None:
            var = var + ((a_t_star*t-a_t*gamma_p) * sigma_theta) ** 2
        if temperature != 1:
            var = temperature ** 2 * var
        return DiagonalGaussian(mean, var=var)

    def a_b_prime_minus_a_prime_b(
        self,
        t: Float[Tensor, "*batch"]
    ) -> Float[Tensor, "*batch"]:
        """Computes a(t) * b'(t) - a'(t) * b(t)"""
        return torch.ones(
            (1,), device=t.device, dtype=t.dtype
        ).expand_as(t)

    def conditional_p_v(
        self,
        mean_theta: Float[Tensor, "*batch"], # This is v_theta
        z_t: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"],
        t_star: Float[Tensor, "*#batch"],
        alpha: float | int,
        temperature: float | int = 1,
        sigma_theta: Float[Tensor, "*#batch"] | None = None, # This is sigma_v_theta
        v_theta: Float[Tensor, "*#batch"] | None = None
    ) -> DiagonalGaussian:
        # Convert v_theta to eps_theta
        # eps = b(t)*z_t + a(t)*v
        mean_eps_theta = self.get_eps_from_v_and_zt(mean_theta, z_t, t)
        
        sigma_eps_theta = None
        if sigma_theta is not None:
            # sigma_eps_theta = a(t) * sigma_v_theta
            a_t = self.a(t) # a(t) = 1 - t for RectifiedFlow
            if a_t.ndim < sigma_theta.ndim:
                 a_t = a_t.view([-1] + [1]*(sigma_theta.ndim - a_t.ndim))
            sigma_eps_theta = a_t * sigma_theta

        return self.conditional_p_eps(
            mean_theta=mean_eps_theta,
            z_t=z_t,
            t=t,
            t_star=t_star,
            alpha=alpha,
            temperature=temperature,
            sigma_theta=sigma_eps_theta,
            v_theta=v_theta # This v_theta is for variance interpolation, not the predicted v.
        )

    def divergence_simple_v(
        self,
        sigma_theta: Float[Tensor, "*batch"], # This is sigma_v_theta
        t: Float[Tensor, "*#batch"],
        t_star: Float[Tensor, "*#batch"],
        alpha: float | int,
    ) -> Float[Tensor, "*batch"]:
        # Convert sigma_v_theta to sigma_eps_theta
        # sigma_eps_theta = a(t) * sigma_v_theta
        a_t = self.a(t) # a(t) = 1 - t for RectifiedFlow
        if a_t.ndim < sigma_theta.ndim:
            a_t = a_t.view([-1] + [1]*(sigma_theta.ndim - a_t.ndim))
        sigma_eps_theta = a_t * sigma_theta
        
        return self.divergence_simple_eps(
            sigma_theta=sigma_eps_theta,
            t=t,
            t_star=t_star,
            alpha=alpha
        )
