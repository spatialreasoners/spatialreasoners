from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor

from spatialreasoners.type_extensions import Parameterization

from . import Flow, FlowCfg, register_flow
from .diagonal_gaussian import DiagonalGaussian


@dataclass
class CosineFlowCfg(FlowCfg):
    skew: float = 0.


@register_flow("cosine", CosineFlowCfg)
class CosineFlow(Flow[CosineFlowCfg]):
    def __init__(
        self, 
        cfg: CosineFlowCfg,
        parameterization: Parameterization = "ut"
    ):
        super().__init__(cfg, parameterization)
        self.pi_half = torch.pi / 2
        self.skew_norm = 1 + self.cfg.skew
        self.prime_scale = self.pi_half / self.skew_norm

    def skew_t(
        self, 
        t: Float[Tensor, "*batch"]
    ) -> Float[Tensor, "*batch"]:
        return (t + self.cfg.skew) / self.skew_norm

    def a(
        self, 
        t: Float[Tensor, "*batch"]
    ) -> Float[Tensor, "*batch"]:
        """Define data weight a_t for t"""
        return torch.cos(self.pi_half * self.skew_t(t))

    def a_prime(
        self,
        t: Float[Tensor, "*batch"]
    ) -> Float[Tensor, "*batch"]:
        """Define derivative of data weight a_t for t"""
        return -self.prime_scale * torch.sin(self.pi_half * self.skew_t(t))

    def b(
        self, 
        t: Float[Tensor, "*batch"]
    ) -> Float[Tensor, "*batch"]:
        """Define noise weight b_t for t"""
        return torch.sin(self.pi_half * self.skew_t(t))

    def b_prime(
        self, 
        t: Float[Tensor, "*batch"]
    ) -> Float[Tensor, "*batch"]:
        """Define derivative of noise weight b_t for t"""
        return self.prime_scale * torch.cos(self.pi_half * self.skew_t(t))

    def get_ut_from_eps_and_x(
        self,
        eps: Float[Tensor, "*batch"],
        x: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        rad = self.pi_half * self.skew_t(t)
        return self.prime_scale * (
            torch.cos(rad) * eps \
                - torch.sin(rad) * x
        )
    
    def get_ut_from_eps_and_zt(
        self,
        eps: Float[Tensor, "*batch"],
        zt: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        # a_prime / a * zt - (a_prime * b / a - b_prime) * eps
        rad = self.pi_half * self.skew_t(t)
        sin, cos = torch.sin(rad), torch.cos(rad)
        tan = sin / cos
        return self.prime_scale * (
            (sin * tan - cos) * eps - tan * zt
        )

    def get_x_from_ut_and_zt(
        self,
        ut: Float[Tensor, "*batch"],
        zt: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        rad = self.pi_half * self.skew_t(t)
        sin, cos = torch.sin(rad), torch.cos(rad)
        return (zt - sin * (sin * zt + cos / self.prime_scale * ut)) / cos 

    def get_x_from_eps_and_zt(
        self,
        eps: Float[Tensor, "*batch"],
        zt: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        """Compute x0 from eps and zt: x0 = (zt - b(t)*eps) / a(t)"""
        a_t = self.a(t)
        b_t = self.b(t)
        # if a_t is 0 (skewed_t = 1), then zt = b_t*eps. x0 is ill-defined.
        safe_a_t = a_t + 1e-9 * (a_t == 0)
        return (zt - b_t * eps) / safe_a_t

    def get_x_from_v_and_eps(
        self,
        v: Float[Tensor, "*batch"],
        eps: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        """Compute x0 from v and eps: x0 = (a(t)*eps - v) / b(t)"""
        a_t = self.a(t)
        b_t = self.b(t)
        # if b_t is 0 (skewed_t = 0), then v = a_t*eps. x0 is ill-defined.
        safe_b_t = b_t + 1e-9 * (b_t == 0)
        return (a_t * eps - v) / safe_b_t

    def get_x_from_eps_and_ut(
        self,
        eps: Float[Tensor, "*batch"],
        ut: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        """Compute x0 from eps and ut: x0 = (ut - b_prime(t)*eps) / a_prime(t)"""
        a_prime_t = self.a_prime(t)
        b_prime_t = self.b_prime(t)
        # if a_prime_t is 0 (skewed_t = 0), then ut = b_prime_t*eps. x0 is ill-defined.
        safe_a_prime_t = a_prime_t + 1e-9 * (a_prime_t == 0)
        return (ut - b_prime_t * eps) / safe_a_prime_t

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
        b_t = self.b(t)
        mean = a_t_star * (a_t * z_t - b_t * mean_theta / self.prime_scale) \
            + (a_t * mean_theta / self.prime_scale + b_t * z_t) * gamma_p
        var = self.sigma(t, t_star, alpha, v_theta) ** 2
        if sigma_theta is not None:
            var = var + (((a_t_star*b_t-a_t*gamma_p) / self.prime_scale) * sigma_theta) ** 2
        if temperature != 1:
            var = temperature ** 2 * var
        return DiagonalGaussian(mean, var=var)

    def a_b_prime_minus_a_prime_b(
        self,
        t: Float[Tensor, "*batch"]
    ) -> Float[Tensor, "*batch"]:
        """Computes a(t) * b'(t) - a'(t) * b(t)"""
        return torch.full(
            (1,), fill_value=self.prime_scale, device=t.device, dtype=t.dtype
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
        # For CosineFlow, a(t)^2 + b(t)^2 = 1, so get_eps_from_v_and_zt simplifies from the general form.
        # eps_pred = self.b(t) * z_t + self.a(t) * v_pred
        mean_eps_theta = self.get_eps_from_v_and_zt(mean_theta, z_t, t)
        
        sigma_eps_theta = None
        if sigma_theta is not None:
            # sigma_eps_theta = a(t) * sigma_v_theta
            a_t = self.a(t)
            # Ensure a_t is broadcastable with sigma_theta if sigma_theta has extra dims (e.g. feature dim)
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
        a_t = self.a(t)
        if a_t.ndim < sigma_theta.ndim:
            a_t = a_t.view([-1] + [1]*(sigma_theta.ndim - a_t.ndim))
        sigma_eps_theta = a_t * sigma_theta
        
        return self.divergence_simple_eps(
            sigma_theta=sigma_eps_theta,
            t=t,
            t_star=t_star,
            alpha=alpha
        )

    def get_eps_from_x_and_zt(
        self,
        x0: Float[Tensor, "*batch"],
        zt: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        """Compute eps from x0 and zt: eps = (zt - a(t)*x0) / b(t)"""
        a_t = self.a(t)
        b_t = self.b(t)
        # if b_t is 0 (skewed_t = 0), then zt = a_t*x0. eps is ill-defined.
        safe_b_t = b_t + 1e-9 * (b_t == 0)
        return (zt - a_t * x0) / safe_b_t

    def get_eps_from_v_and_x0(
        self,
        v: Float[Tensor, "*batch"],
        x0: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        """Compute eps from v and x0: eps = (v + b(t)*x0) / a(t)"""
        a_t = self.a(t)
        b_t = self.b(t)
        # if a_t is 0 (skewed_t = 1), then v = -b_t*x0. eps is ill-defined.
        safe_a_t = a_t + 1e-9 * (a_t == 0)
        return (v + b_t * x0) / safe_a_t

    def get_eps_from_ut_and_x0(
        self,
        ut: Float[Tensor, "*batch"],
        x0: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        """Compute eps from ut and x0: eps = (ut - a_prime(t)*x0) / b_prime(t)"""
        a_prime_t = self.a_prime(t)
        b_prime_t = self.b_prime(t)
        # if b_prime_t is 0 (skewed_t = 1), then ut = a_prime_t*x0. eps is ill-defined.
        safe_b_prime_t = b_prime_t + 1e-9 * (b_prime_t == 0)
        return (ut - a_prime_t * x0) / safe_b_prime_t

    def get_eps_from_ut_and_zt(
        self,
        ut: Float[Tensor, "*batch"],
        zt: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        """Compute eps from ut and zt: eps = (a(t)*ut - a_prime(t)*zt) / a_b_prime_minus_a_prime_b(t)
        For CosineFlow, a_b_prime_minus_a_prime_b(t) is self.prime_scale.
        So, eps = (a(t)*ut - a_prime(t)*zt) / self.prime_scale
        """
        a_t = self.a(t)
        a_prime_t = self.a_prime(t)
        # self.prime_scale should be non-zero by construction if skew != -1
        # Adding safe_denom for robustness, though prime_scale is a scalar config value.
        safe_prime_scale = self.prime_scale + 1e-9 * (self.prime_scale == 0)
        return (a_t * ut - a_prime_t * zt) / safe_prime_scale
