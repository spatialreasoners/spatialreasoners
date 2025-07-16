from dataclasses import dataclass, field
from functools import partial

import torch
from jaxtyping import Bool, Float, Int64
from torch import Tensor

from spatialreasoners.type_extensions import Parameterization

from . import Flow, FlowCfg, register_flow
from .beta_schedule import BetaScheduleCfg, LinearCfg, get_beta_schedule
from .diagonal_gaussian import DiagonalGaussian


@dataclass
class OriginalDiffusionCfg(FlowCfg):
    num_timesteps: int = 1000
    beta_schedule: BetaScheduleCfg = field(default_factory=LinearCfg)


# @register_flow("original_diffusion", OriginalDiffusionCfg)
class OriginalDiffusion(Flow[OriginalDiffusionCfg]):
    def __init__(
        self, 
        cfg: OriginalDiffusionCfg,
        parameterization: Parameterization = "eps"
    ):
        assert parameterization != "ut", "Discrete time diffusion does not support flows"
        super().__init__(cfg, parameterization)
        self._num_spaced_timesteps = self.cfg.num_timesteps
        register_buffer = partial(self.register_buffer, persistent=False)
        register_buffer("base_betas", torch.from_numpy(get_beta_schedule(cfg.beta_schedule)(cfg.num_timesteps)).float())
        register_buffer("base_alphas", 1.0 - self.base_betas)
        register_buffer("base_alphas_bar", torch.cumprod(self.base_alphas, dim=0))

        register_buffer("timesteps", torch.arange(self.cfg.num_timesteps))

        for name in (
            "betas",
            "log_betas",
            "alphas",
            "alphas_bar",
            "alphas_bar_prev",
            # calculations for diffusion q(x_t | x_0) and others
            "sqrt_alphas_bar",
            "sqrt_alphas_bar_prev",
            "sqrt_one_minus_alphas_bar",
            "sqrt_recip_alphas_bar",
            "sqrt_recipm1_alphas_bar",
            # calculations for posterior q(x_{t-1} | x_t, x_0)
            "post_var",
            "post_std",
            # clip log var for tilde_betas_0 = 0
            "post_var_clipped",
            "post_std_clipped",
            "post_log_var_clipped",
            # Mean coefficients for conditional p (DDIM)
            "ddim_mean_coef1",
            "ddim_mean_coef2",
            # for fixedlarge, we set the initial (log-)variance like so to get a better decoder log likelihood.
            "large_var",
            "large_std",
            "large_log_var"
        ):
            register_buffer(name, None)

        self.set_buffers(self.base_betas) 

    def set_buffers(self, betas: Float[Tensor, "t"]) -> None:
        self.betas = betas
        self.log_betas = torch.log(self.betas)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.alphas_bar_prev = torch.cat((torch.ones_like(self.alphas_bar[:1]), self.alphas_bar[:-1]))
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_alphas_bar_prev = torch.sqrt(self.alphas_bar_prev)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1 - self.alphas_bar)
        self.sqrt_recip_alphas_bar = torch.sqrt(1 / self.alphas_bar)
        self.sqrt_recipm1_alphas_bar = torch.sqrt(1 / self.alphas_bar - 1)
        self.post_var = self.betas * (1 - self.alphas_bar_prev) / (1 - self.alphas_bar)
        self.post_std = torch.sqrt(self.post_var)
        self.post_var_clipped = torch.cat((self.post_var[1:2], self.post_var[1:]))
        self.post_std_clipped = torch.sqrt(self.post_var_clipped)
        self.post_log_var_clipped = torch.log(self.post_var_clipped)
        self.ddim_mean_coef1 = self.sqrt_recip_alphas_bar * self.sqrt_alphas_bar_prev
        self.ddim_mean_coef2 = self.sqrt_recipm1_alphas_bar * self.sqrt_alphas_bar_prev
        self.large_var = torch.cat((self.post_var[1:2], self.betas[1:]))
        self.large_std = torch.sqrt(self.large_var)
        self.large_log_var = torch.log(self.large_var)
    
    @property
    def num_spaced_timesteps(self) -> int:
        return self._num_spaced_timesteps
    
    @num_spaced_timesteps.setter
    def num_spaced_timesteps(self, val: int) -> None:
        if self._num_spaced_timesteps != val:
            assert val <= self.cfg.num_timesteps, \
                "Number of spaced timesteps must always be at most number of timesteps"
            self._num_spaced_timesteps = val

            if self._num_spaced_timesteps < self.cfg.num_timesteps:
                # Update timesteps
                frac_stride = 1 if self._num_spaced_timesteps <= 1 \
                    else (self.cfg.num_timesteps - 1) / (self._num_spaced_timesteps - 1)
                self.timesteps = torch.arange(
                    0, frac_stride * self._num_spaced_timesteps, frac_stride, device=self.timesteps.device
                ).round_().long()
                
                # Update betas and all derived variables
                last_alpha_bar = 1.0
                betas = torch.empty(
                    (self._num_spaced_timesteps,), dtype=self.betas.dtype, device=self.betas.device
                )
                for i, t in enumerate(self.timesteps):
                    betas[i] = 1 - self.base_alphas_bar[t] / last_alpha_bar
                    last_alpha_bar = self.base_alphas_bar[t]
            else:
                self.timesteps = torch.arange(self.cfg.num_timesteps, device=self.timesteps.device)
                betas = self.base_betas
            
            self.set_buffers(betas)

    def adjust_time(
        self,
        t: Float[Tensor, "*batch"]
    ) -> Float[Tensor, "*batch"]:
        if self.num_spaced_timesteps < self.cfg.num_timesteps:
            t_idx = self.timesteps[self.time_to_index(t)]
        else:
            t_idx = (self.cfg.num_timesteps * t).floor_().clamp_(0, self.cfg.num_timesteps-1)
        return torch.where(t == 0, 0, (t_idx + 1) / self.cfg.num_timesteps)

    def time_to_index(
        self,
        t: Float[Tensor, "*batch"]
    ) -> Int64[Tensor, "*batch"]:
        return (self._num_spaced_timesteps * t)\
            .sub_(0.5 * self._num_spaced_timesteps / self.cfg.num_timesteps).floor_().clamp_min_(0).long()

    def assert_single_step(
        self,
        t: Float[Tensor, "*batch"],
        t_star: Float[Tensor, "*#batch"],
    ) -> tuple[
        Int64[Tensor, "*batch"],
        Bool[Tensor, "*batch"]
    ]:
        t_idx = self.time_to_index(t)
        t_star_idx = self.time_to_index(t_star)
        mask = t_idx != t_star_idx
        assert (t_idx[mask] == t_star_idx[mask] + 1).all()
        mask.logical_or_((t_idx == 0).logical_and_(t > 0))
        return t_idx, mask

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
        t_idx, mask = self.assert_single_step(t, t_star)
        return torch.where(mask, alpha * self.post_std_clipped[t_idx], 0)

    def sigma_large(
        self,
        t: Float[Tensor, "*batch"],
        t_star: Float[Tensor, "*#batch"],
        alpha: float | int
    ) -> Float[Tensor, "*batch"]:
        """Computes the upper bound of the variance for the approximate reverse distribution p"""
        t_idx, mask = self.assert_single_step(t, t_star)
        return torch.where(mask, alpha * self.large_std[t_idx], 0)

    def sigma(
        self,
        t: Float[Tensor, "*#batch"],
        t_star: Float[Tensor, "*#batch"],
        alpha: float | int,
        v_theta: Float[Tensor, "*batch"] | None = None
    ) -> Float[Tensor, "*batch"]:
        """
        Computes the variance for the approximate reverse distribution p
        NOTE assumes the normalization of v_theta to be done already
        """
        if self.cfg.variance == "fixed_small":
            return self.sigma_small(t, t_star, alpha)
        elif self.cfg.variance == "fixed_large":
            return self.sigma_large(t, t_star, alpha)
        else:
            assert v_theta is not None
            t_idx, mask = self.assert_single_step(t, t_star)
            min_log = self.post_log_var_clipped[t_idx]
            max_log = self.log_betas[t_idx]
            log_var = v_theta * max_log + (1 - v_theta) * min_log
            return torch.where(mask, alpha * torch.exp(0.5 * log_var), 0)

    def gamma(
        self,
        t: Float[Tensor, "*batch"],
        t_star: Float[Tensor, "*batch"],
        alpha: float | int
    ) -> Float[Tensor, "*batch"]:
        """Computes sqrt(b_{t*}^2-sigma_small(t,t*)^2)"""
        raise NotImplementedError()

    def get_eps_from_x_and_zt(
        self,
        x: Float[Tensor, "*batch"],
        zt: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        idx = self.time_to_index(t)
        return (self.sqrt_recip_alphas_bar[idx] * zt - x) / self.sqrt_recipm1_alphas_bar[idx]

    def get_ut(
        self,
        t: Float[Tensor, "*#batch"],
        eps: Float[Tensor, "*batch"] | None = None,
        ut: Float[Tensor, "*batch"] | None = None,
        x: Float[Tensor, "*batch"] | None = None,
        zt: Float[Tensor, "*batch"] | None = None
    ) -> Float[Tensor, "*batch"]:
        raise ValueError("Discrete time diffusion does not support flows")

    def get_x_from_eps_and_zt(
        self,
        eps: Float[Tensor, "*batch"],
        zt: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        idx = self.time_to_index(t)
        return self.sqrt_recip_alphas_bar[idx] * zt - self.sqrt_recipm1_alphas_bar[idx] * eps

    def conditional_p_eps(
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
        # mean_theta and sigma_theta are the mean and standard deviation of eps
        idx, mask = self.assert_single_step(t, t_star)
        sigma_small = alpha * self.post_std[idx]
        eps_scale = torch.sqrt(1 - self.alphas_bar_prev[idx] - sigma_small ** 2) - self.ddim_mean_coef2[idx]
        mean = self.ddim_mean_coef1[idx] * z_t + eps_scale * mean_theta
        var = self.sigma(t, t_star, alpha, v_theta) ** 2
        if sigma_theta is not None:
            var = var + (eps_scale * sigma_theta) ** 2
        if temperature != 1:
            var = temperature ** 2 * var
        mean = torch.where(mask, mean, z_t)
        var = torch.where(mask, var, 0)
        return DiagonalGaussian(mean, var=var)

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

    def conditional_q(
        self,
        x: Float[Tensor, "*batch"],
        eps: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"],
        t_star: Float[Tensor, "*#batch"],
        alpha: float | int,
    ) -> DiagonalGaussian:
        idx, mask = self.assert_single_step(t, t_star)
        sigma_small = alpha * self.post_std[idx]
        eps_scale = torch.sqrt(1 - self.alphas_bar_prev[idx] - sigma_small ** 2)
        mean = self.sqrt_alphas_bar_prev[idx] * x + eps_scale * eps
        return DiagonalGaussian(
            torch.where(mask, mean, self.get_zt(t, eps=eps, x=x)), 
            std=torch.where(mask, sigma_small, 0)
        )

    def divergence_simple_eps(
        self,
        sigma_theta: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"],
        t_star: Float[Tensor, "*#batch"],
        alpha: float | int,
    ) -> Float[Tensor, "*batch"]:
        """
        Computes the simplified KL divergence between conditional_p_eps and conditional_q
        NOTE assumes that sigma_theta is the standard deviation of the predicted **noise eps**
        """
        # TODO adapt this for other variances than fixed_small?
        
        idx, mask = self.assert_single_step(t, t_star)
        sigma_small = alpha * self.post_std_clipped[idx]
        eps_scale = torch.sqrt(1 - self.alphas_bar_prev[idx] - sigma_small ** 2) - self.ddim_mean_coef2[idx]
        kl_div =  torch.log(
            torch.sqrt(
                (eps_scale * sigma_theta) ** 2 + sigma_small ** 2
            ) / sigma_small
        )
        return torch.where(mask, kl_div, 0)

    def divergence_simple_ut(
        self,
        sigma_theta: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"],
        t_star: Float[Tensor, "*#batch"],
        alpha: float | int,
    ) -> Float[Tensor, "*batch"]:
        raise ValueError("Discrete time diffusion does not support flows")

    def on_sampling_start(self, num_noise_level_values: int | None) -> None:
        assert (
            num_noise_level_values is not None
        ), "Number of noise level values must be specified for discrete time diffusion"
        
        self.num_spaced_timesteps = num_noise_level_values

    def on_sampling_end(self) -> None:
        self.num_spaced_timesteps = self.cfg.num_timesteps
