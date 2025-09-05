from dataclasses import dataclass, field
from functools import partial

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from spatialreasoners.type_extensions import Parameterization

from . import Flow, FlowCfg, register_flow
from .beta_schedule import BetaScheduleCfg, LinearCfg, get_beta_schedule
from .diagonal_gaussian import DiagonalGaussian


@dataclass
class DiffusionCfg(FlowCfg):
    num_timesteps: int = 1000
    beta_schedule: BetaScheduleCfg = field(default_factory=LinearCfg)


@register_flow("diffusion", DiffusionCfg)
class Diffusion(Flow[DiffusionCfg]):
    def __init__(
        self, 
        cfg: DiffusionCfg,
        parameterization: Parameterization = "eps"
    ):
        assert parameterization != "ut", "Discrete time diffusion does not support flows"
        super().__init__(cfg, parameterization)
        register_buffer = partial(self.register_buffer, persistent=False)
        register_buffer("betas", torch.from_numpy(get_beta_schedule(cfg.beta_schedule)(cfg.num_timesteps)).float())
        register_buffer("alphas", 1.0 - self.betas)
        register_buffer("alphas_bar", torch.cumprod(self.alphas, dim=0))
        # calculations for diffusion q(x_t | x_0) and others
        register_buffer("sqrt_alphas_bar", torch.sqrt(self.alphas_bar))
        register_buffer("sqrt_one_minus_alphas_bar", torch.sqrt(1 - self.alphas_bar))
        register_buffer("min_posterior_std", 
                        torch.sqrt(self.betas[1] * (1.0 - self.alphas_bar[0]) / (1.0 - self.alphas_bar[1]))
        )
    
    def adjust_time(
        self,
        t: Float[Tensor, "*batch"]
    ) -> Float[Tensor, "*batch"]:
        return torch.where(
            t == 0, 
            0, 
            (self.cfg.num_timesteps * t).floor_().clamp_(0, self.cfg.num_timesteps-1)\
                .add_(1).div_(self.cfg.num_timesteps)
        )

    def time_to_index(
        self,
        t: Float[Tensor, "*batch"]
    ) -> Int64[Tensor, "*batch"]:
        # NOTE avoid sampling with more timesteps than num_timesteps
        return (self.cfg.num_timesteps * t).sub_(0.5).floor_().clamp_min_(0).long()

    def a(
        self, 
        t: Float[Tensor, "*batch"]
    ) -> Float[Tensor, "*batch"]:
        """Define data weight a_t for t"""
        return torch.where(t == 0, 1., self.sqrt_alphas_bar[self.time_to_index(t)])

    def a_prime(
        self,
        t: Float[Tensor, "*batch"]
    ) -> Float[Tensor, "*batch"]:
        """Define derivative of data weight a_t for t"""
        raise ValueError("Data weight (a) of discrete time diffusion is not differentiable")

    def b(
        self, 
        t: Float[Tensor, "*batch"]
    ) -> Float[Tensor, "*batch"]:
        """Define noise weight b_t for t"""
        return torch.where(t == 0, 0., self.sqrt_one_minus_alphas_bar[self.time_to_index(t)])

    def b_prime(
        self, 
        t: Float[Tensor, "*batch"]
    ) -> Float[Tensor, "*batch"]:
        """Define derivative of noise weight b_t for t"""
        raise ValueError("Noise weight (b) of discrete time diffusion is not differentiable")

    def sigma_small(
        self,
        t: Float[Tensor, "*batch"],
        t_star: Float[Tensor, "*#batch"],
        alpha: float | int
    ) -> Float[Tensor, "*batch"]:
        """Computes the lower bound of the variance for the approximate reverse distribution p"""
        sigma = super(Diffusion, self).sigma_small(t, t_star, alpha)
        sigma.clamp_min_(self.min_posterior_std)    # TODO that is not correct for alpha == 0 or t == t_star!
        return sigma

    def get_ut(
        self,
        t: Float[Tensor, "*#batch"],
        eps: Float[Tensor, "*batch"] | None = None,
        ut: Float[Tensor, "*batch"] | None = None,
        x: Float[Tensor, "*batch"] | None = None,
        zt: Float[Tensor, "*batch"] | None = None
    ) -> Float[Tensor, "*batch"]:
        raise ValueError("Discrete time diffusion does not support flows")

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
        raise ValueError("Discrete time diffusion does not support flows") 

    def divergence_simple_ut(
        self,
        sigma_theta: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"],
        t_star: Float[Tensor, "*#batch"],
        alpha: float | int,
    ) -> Float[Tensor, "*batch"]:
        raise ValueError("Discrete time diffusion does not support flows")

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
        # eps_pred = self.b(t) * z_t + self.a(t) * v_pred
        # However, conditional_p_eps expects mean_theta to be the predicted epsilon directly.
        # The formula for x_0 prediction from v is x_0 = a(t)z_t - b(t)v.
        # The formula for eps prediction from x_0 is eps = (z_t - a(t)x_0) / b(t).
        # Substituting x_0: eps = (z_t - a(t)(a(t)z_t - b(t)v)) / b(t)
        # eps = (z_t - a(t)^2 z_t + a(t)b(t)v) / b(t)
        # eps = ( (1 - a(t)^2)z_t + a(t)b(t)v ) / b(t)
        # Since 1 - a(t)^2 = 1 - alphas_bar = b(t)^2 (for Diffusion class where a(t)=sqrt_alphas_bar, b(t)=sqrt_one_minus_alphas_bar)
        # eps = ( b(t)^2 z_t + a(t)b(t)v ) / b(t)
        # eps = b(t)z_t + a(t)v
        mean_eps_theta = self.get_eps_from_v_and_zt(mean_theta, z_t, t)
        
        sigma_eps_theta = None
        if sigma_theta is not None:
            # sigma_eps_theta = a(t) * sigma_v_theta
            a_t = self.a(t)
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
