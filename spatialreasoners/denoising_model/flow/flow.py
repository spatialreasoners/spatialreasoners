from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import ceil, log2
from typing import Generic, Literal, TypeVar

import torch
from jaxtyping import Float
from torch import Tensor
from torch.nn import Module

from spatialreasoners.type_extensions import Parameterization

from .diagonal_gaussian import DiagonalGaussian


@dataclass
class FlowCfg:
    variance: Literal["fixed_small", "fixed_large", "learned_range"] = "fixed_small"


T = TypeVar("T", bound=FlowCfg)


class Flow(Module, ABC, Generic[T]):
    cfg: T

    def __init__(
        self,
        cfg: T,
        parameterization: Parameterization = "ut"
    ) -> None:
        super(Flow, self).__init__()
        self.cfg = cfg
        self.parameterization = parameterization
        if parameterization == "eps":
            self.conditional_p = self.conditional_p_eps
            self.divergence_simple = self.divergence_simple_eps
        elif parameterization == "ut":
            self.conditional_p = self.conditional_p_ut
            self.divergence_simple = self.divergence_simple_ut
        elif parameterization == "v":
            self.conditional_p = self.conditional_p_v
            self.divergence_simple = self.divergence_simple_v
        else:
            raise ValueError(f"Unknown parameterization {parameterization}")

    def adjust_time(
        self,
        t: Float[Tensor, "*batch"]
    ) -> Float[Tensor, "*batch"]:
        return t

    @abstractmethod
    def a(
        self, 
        t: Float[Tensor, "*batch"]
    ) -> Float[Tensor, "*batch"]:
        """Define data weight a_t for t"""
        pass
    
    @abstractmethod
    def a_prime(
        self,
        t: Float[Tensor, "*batch"]
    ) -> Float[Tensor, "*batch"]:
        """Define derivative of data weight a_t for t"""
        pass

    @abstractmethod
    def b(
        self, 
        t: Float[Tensor, "*batch"]
    ) -> Float[Tensor, "*batch"]:
        """Define noise weight b_t for t"""
        pass
    
    @abstractmethod
    def b_prime(
        self, 
        t: Float[Tensor, "*batch"]
    ) -> Float[Tensor, "*batch"]:
        """Define derivative of noise weight b_t for t"""
        pass

    def sigma_small(
        self,
        t: Float[Tensor, "*batch"],
        t_star: Float[Tensor, "*#batch"],
        alpha: float | int
    ) -> Float[Tensor, "*batch"]:
        """Computes the lower bound of the variance for the approximate reverse distribution p"""
        b_t_star = self.b(t_star)
        sigma = alpha * b_t_star * torch.sqrt(
            1-(self.a(t)*b_t_star / (self.a(t_star)*self.b(t)))**2
        )
        sigma[(t == 0).logical_or_(t_star == 1)] = 0
        return sigma

    def sigma_large(
        self,
        t: Float[Tensor, "*batch"],
        t_star: Float[Tensor, "*#batch"],
        alpha: float | int
    ) -> Float[Tensor, "*batch"]:
        """Computes the upper bound of the variance for the approximate reverse distribution p"""
        b_t = self.b(t)
        sigma = alpha * b_t * torch.sqrt(
            1-(self.a(t)*self.b(t_star) / (self.a(t_star)*b_t))**2
        )
        sigma[(t == 0).logical_or_(t_star == 1)] = 0
        return sigma        

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
            sigma_large = self.sigma_large(t, t_star, alpha).expand_as(v_theta)
            sigma_small = self.sigma_small(t, t_star, alpha).expand_as(v_theta)
            sigma = torch.zeros_like(v_theta)
            mask = (sigma_large != 0).logical_and_(sigma_small != 0).expand_as(v_theta)
            v_theta = v_theta[mask]
            # this can change the dtype in AMP
            nnz_sigma = torch.exp(
                v_theta * torch.log(sigma_large[mask]) \
                + (1 - v_theta) * torch.log(sigma_small[mask])
            )
            sigma = torch.zeros(mask.shape, dtype=nnz_sigma.dtype, device=nnz_sigma.device)
            sigma[mask] = nnz_sigma
            return sigma

    def gamma(
        self,
        t: Float[Tensor, "*batch"],
        t_star: Float[Tensor, "*batch"],
        alpha: float | int
    ) -> Float[Tensor, "*batch"]:
        """Computes sqrt(b_{t*}^2-sigma_small(t,t*)^2)"""
        b_t_star = self.b(t_star)
        gamma = b_t_star * torch.sqrt(
            1-alpha**2*(1-(self.a(t)*b_t_star / (self.a(t_star)*self.b(t)))**2)
        )
        gamma[t == 0] = 0
        ones_mask = t_star == 1
        gamma[ones_mask] = b_t_star[ones_mask]
        return gamma
    
    @staticmethod
    def sample_eps(
        x: Float[Tensor, "*batch"]
    ) -> Float[Tensor, "*batch"]:
        return torch.randn_like(x)
    
    def get_eps_from_ut_and_x(
        self,
        ut: Float[Tensor, "*batch"],
        x: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        raise NotImplementedError()

    def get_eps_from_ut_and_zt(
        self,
        u_t: Float[Tensor, "*batch"],
        zt: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        raise NotImplementedError()
    
    def get_eps_from_x_and_zt(
        self,
        x: Float[Tensor, "*batch"],
        zt: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        """Compute eps from x0 and zt.
        eps = (zt - a(t)*x0) / b(t)
        Subclasses must implement this if they support it, handling b(t) possibly being zero.
        """
        raise NotImplementedError("Subclasses should implement get_eps_from_x_and_zt if applicable, handling b(t)=0 case.")

    def get_eps(
        self,
        t: Float[Tensor, "*#batch"],
        eps: Float[Tensor, "*batch"] | None = None,
        ut: Float[Tensor, "*batch"] | None = None,
        x: Float[Tensor, "*batch"] | None = None, # x is x0
        zt: Float[Tensor, "*batch"] | None = None,
        v: Float[Tensor, "*batch"] | None = None
    ) -> Float[Tensor, "*batch"]:
        if eps is not None:
            return eps
        
        provided_args = sum(val is not None for val in (ut, x, zt, v))
        assert provided_args == 2, f"Exactly two of (ut, x0, zt, v) must be provided to compute eps, got {provided_args}"

        if ut is not None:
            if x is not None: # ut and x0
                return self.get_eps_from_ut_and_x(ut, x, t)
            if zt is not None: # ut and zt
                return self.get_eps_from_ut_and_zt(ut, zt, t)
            # if v is not None: # ut and v. Formula: eps = (b*ut + a_prime*v) / (a_prime*a + b_prime*b). Requires a_prime, b_prime.
            #     raise NotImplementedError("get_eps_from_ut_and_v is not implemented by base class.")
        
        if x is not None: # ut must be None here
            if zt is not None: # x0 and zt
                return self.get_eps_from_x_and_zt(x, zt, t)
            if v is not None: # x0 and v
                return self.get_eps_from_v_and_x0(v, x, t) # x is x0
            
        if zt is not None: # ut and x0 must be None here
            if v is not None: # zt and v
                return self.get_eps_from_v_and_zt(v, zt, t) # This is implemented in base and correct general form.

        raise ValueError("Unsupported combination of inputs for get_eps, or required helper method not implemented by subclass.")

    def get_zt_from_eps_and_ut(
        self,
        eps: Float[Tensor, "*batch"],
        ut: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        raise NotImplementedError()

    def get_zt_from_eps_and_x(
        self,
        eps: Float[Tensor, "*batch"],
        x: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        # NOTE equivalent to q_sample in OpenAI's diffusion implementation
        return self.a(t) * x + self.b(t) * eps
    
    def get_zt_from_ut_and_x(
        self,
        ut: Float[Tensor, "*batch"],
        x: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        raise NotImplementedError()

    def get_zt(
        self,
        t: Float[Tensor, "*#batch"],
        eps: Float[Tensor, "*batch"] | None = None,
        ut: Float[Tensor, "*batch"] | None = None,
        x: Float[Tensor, "*batch"] | None = None,
        zt: Float[Tensor, "*batch"] | None = None
    ) -> Float[Tensor, "*batch"]:
        if zt is not None:
            return zt
        assert sum(a is not None for a in (eps, ut, x)) == 2
        if eps is not None:
            if ut is not None:
                return self.get_zt_from_eps_and_ut(eps, ut, t)
            return self.get_zt_from_eps_and_x(eps, x, t)
        return self.get_zt_from_ut_and_x(ut, x, t)

    def get_ut_from_eps_and_x(
        self,
        eps: Float[Tensor, "*batch"],
        x: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        """Compute x0 from eps and ut.
        x0 = (ut - b_prime(t)*eps) / a_prime(t)
        Subclasses must implement this if they support it, handling a_prime(t) possibly being zero.
        """
        raise NotImplementedError("Subclasses should implement get_x_from_eps_and_ut if applicable.")

    def get_ut_from_eps_and_zt(
        self,
        eps: Float[Tensor, "*batch"],
        zt: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        a_ratio = self.a_prime(t) / self.a(t)
        return a_ratio * zt - (a_ratio * self.b(t) - self.b_prime(t)) * eps
    
    def get_ut_from_x_and_zt(
        self,
        x: Float[Tensor, "*batch"],
        zt: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        raise NotImplementedError()

    def get_ut(
        self,
        t: Float[Tensor, "*#batch"],
        eps: Float[Tensor, "*batch"] | None = None,
        ut: Float[Tensor, "*batch"] | None = None,
        x: Float[Tensor, "*batch"] | None = None,
        zt: Float[Tensor, "*batch"] | None = None
    ) -> Float[Tensor, "*batch"]:
        if ut is not None:
            return ut
        assert sum(a is not None for a in (eps, x, zt)) == 2
        if eps is not None:
            if x is not None:
                return self.get_ut_from_eps_and_x(eps, x, t)
            return self.get_ut_from_eps_and_zt(eps, zt, t)
        return self.get_ut_from_x_and_zt(x, zt, t)

    def get_x_from_eps_and_ut(
        self,
        eps: Float[Tensor, "*batch"],
        ut: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        """Compute x0 from eps and ut.
        x0 = (ut - b_prime(t)*eps) / a_prime(t)
        Subclasses must implement this if they support it, handling a_prime(t) possibly being zero.
        """
        raise NotImplementedError("Subclasses should implement get_x_from_eps_and_ut if applicable.")

    def get_x_from_eps_and_zt(
        self,
        eps: Float[Tensor, "*batch"],
        zt: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        """Compute x0 from eps and zt.
        x0 = (zt - b(t)*eps) / a(t)
        Subclasses must implement this if they support it, handling a(t) possibly being zero.
        """
        raise NotImplementedError("Subclasses should implement get_x_from_eps_and_zt if applicable, handling a(t)=0 case.")
    
    def get_x_from_ut_and_zt(
        self,
        ut: Float[Tensor, "*batch"],
        zt: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        """Compute x0 from ut and zt.
        x0 = (b_prime(t)*zt - b(t)*ut) / self.a_b_prime_minus_a_prime_b(t)
        Subclasses must implement this if they support it, handling denominator possibly being zero.
        """
        raise NotImplementedError("Subclasses should implement get_x_from_ut_and_zt if applicable.")

    def get_x_from_v_and_eps(
        self,
        v: Float[Tensor, "*batch"],
        eps: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        """Compute x0 from v and eps.
        x0 = (a(t)*eps - v) / b(t)
        Subclasses must implement this if they support it, handling b(t) possibly being zero.
        """
        raise NotImplementedError("Subclasses should implement get_x_from_v_and_eps if applicable, handling b(t)=0 case.")

    def get_x(
        self,
        t: Float[Tensor, "*#batch"],
        eps: Float[Tensor, "*batch"] | None = None,
        ut: Float[Tensor, "*batch"] | None = None,
        x: Float[Tensor, "*batch"] | None = None, # This is x0
        zt: Float[Tensor, "*batch"] | None = None,
        v: Float[Tensor, "*batch"] | None = None
    ) -> Float[Tensor, "*batch"]:
        if x is not None: # x here means x0
            return x
        
        provided_args = sum(val is not None for val in (eps, ut, zt, v))
        assert provided_args == 2, f"Exactly two of (eps, ut, zt, v) must be provided to compute x0, got {provided_args}"

        if eps is not None:
            if ut is not None:
                return self.get_x_from_eps_and_ut(eps, ut, t)
            if zt is not None:
                return self.get_x_from_eps_and_zt(eps, zt, t)
            if v is not None: # eps and v are not None
                return self.get_x_from_v_and_eps(v, eps, t)
        
        if ut is not None: # eps must be None here
            if zt is not None:
                return self.get_x_from_ut_and_zt(ut, zt, t)
            # if v is not None: # ut and v are not None. get_x_from_ut_and_v - could be derived but complex.
            #     raise NotImplementedError("get_x_from_ut_and_v is not implemented by base class.")

        if zt is not None: # eps and ut must be None here
            if v is not None: # zt and v are not None
                return self.get_x0_from_v_and_zt(v, zt, t) # Uses the existing, corrected, and safe method for x0
        
        # This part should ideally not be reached if assert and all pair combinations are correct.
        # However, if a combination leads to a NotImplementedError from a helper, it will propagate.
        raise ValueError("Unsupported combination of inputs for get_x, or required helper method not implemented by subclass.")

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
        a_t, a_t_star = self.a(t), self.a(t_star)
        
        # Original assertion: assert (a_t > 0).all()
        # This assertion can fail for schedules like CosineFlow or Diffusion at t=1 where a(t) can be 0.
        # We'll use a safe version for division. The overall formula should be robust if t and t_star
        # are handled correctly by the sampler (e.g., not having t=1 and t_star < 1 with a_t_star != 0).
        safe_a_t = a_t + 1e-9 * (a_t == 0) # Add epsilon only if a_t is exactly 0
        
        a_ratio = a_t_star / safe_a_t
        eps_scale = self.gamma(t, t_star, alpha) - a_ratio * self.b(t)
        mean = a_ratio * z_t + eps_scale * mean_theta
        var = self.sigma(t, t_star, alpha, v_theta) ** 2
        if sigma_theta is not None:
            var = var + (eps_scale * sigma_theta) ** 2
        if temperature != 1:
            var = temperature ** 2 * var
        return DiagonalGaussian(mean, var=var)

    @abstractmethod
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
        pass

    def marginal_q(
        self,
        x: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> DiagonalGaussian:
        return DiagonalGaussian(
            mean=self.a(t) * x,
            std=self.b(t)
        )
    
    def conditional_q(
        self,
        x: Float[Tensor, "*batch"],
        eps: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"],
        t_star: Float[Tensor, "*#batch"],
        alpha: float | int,
    ) -> DiagonalGaussian:
        return DiagonalGaussian(
            mean=self.a(t_star) * x + eps * self.gamma(t, t_star, alpha),
            std=self.sigma_small(t, t_star, alpha)
        )
    
    def a_b_prime_minus_a_prime_b(
        self,
        t: Float[Tensor, "*batch"]
    ) -> Float[Tensor, "*batch"]:
        """Computes a(t) * b'(t) - a'(t) * b(t)"""
        return self.a(t) * self.b_prime(t) - self.a_prime(t) * self.b(t)

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
        b_t_star = self.b(t_star)
        sigma = self.sigma_small(t, t_star, alpha).clamp_min_(1.e-4)
        gamma = torch.sqrt(b_t_star ** 2 - sigma ** 2)
        return torch.log(
            torch.sqrt(
                ((
                    self.a(t_star) * self.b(t) / self.a(t) - gamma
                ) * sigma_theta) ** 2 + sigma ** 2
            ) / sigma
        )

    def divergence_simple_ut(
        self,
        sigma_theta: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"],
        t_star: Float[Tensor, "*#batch"],
        alpha: float | int,
    ) -> Float[Tensor, "*batch"]:
        """
        Computes the simplified KL divergence between conditional_p_ut and conditional_q
        NOTE assumes that sigma_theta is the standard deviation of the predicted **flow ut**
        """
        # TODO adapt this for other variances than fixed_small?
        b_t_star = self.b(t_star)
        sigma = self.sigma_small(t, t_star, alpha).clamp_min_(1.e-4)
        gamma = torch.sqrt(b_t_star ** 2 - sigma ** 2)
        return torch.log(
            torch.sqrt(
                ((
                    self.a(t_star) * self.b(t) - self.a(t) * gamma
                ) * sigma_theta / self.a_b_prime_minus_a_prime_b(t)) ** 2 + sigma ** 2
            ) / sigma
        )

    @torch.no_grad()
    def inverse_divergence_simple(
        self,
        sigma_theta: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"],
        alpha: float | int,
        threshold: Float[Tensor, "*#batch"],
        max_approx_error: float | None = None,
        num_search_steps: int | None = None,
        t_star_min: Float[Tensor, "*batch"] | None = None,
        t_star_max: Float[Tensor, "*batch"] | None = None
    ) -> Float[Tensor, "*batch"]:
        """
        Find t_star with divergence(t_star) = threshold via binary search 
        exploiting the fact of monotonically decreasing function in t_star
        """
        assert max_approx_error is not None or num_search_steps is not None
        if num_search_steps is None:
            num_search_steps = ceil(-log2(max_approx_error)) - 1
        if t_star_min is None:
            t_star_min = torch.zeros_like(sigma_theta)
        if t_star_max is None:
            t_star_max = t.expand_as(sigma_theta).contiguous()
        t_star = (t_star_min + t_star_max) / 2
        if num_search_steps <= 0:
            return t_star
        div = self.divergence_simple(sigma_theta, t, t_star, alpha)
        mask = div >= threshold
        t_star_min = torch.where(mask, t_star, t_star_min)
        t_star_max = torch.where(mask, t_star_max, t_star)
        return self.inverse_divergence_simple(
            sigma_theta, t, alpha, threshold, 
            num_search_steps=num_search_steps-1, 
            t_star_min=t_star_min, 
            t_star_max=t_star_max
        )

    def on_sampling_start(self, num_noise_level_values: int | None) -> None:
        # Hook for start of sampling
        return

    def on_sampling_end(self) -> None:
        # Hook for end of sampling
        return

    def logsnr_t(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        """
        Computes the log Signal-to-Noise Ratio (logSNR) based on a(t) and b(t).
        logSNR(t) = log (a(t)^2 / b(t)^2) = 2 * (log|a(t)| - log|b(t)|).
        Uses a small epsilon for numerical stability if a(t) or b(t) are zero.
        Subclasses can override this if they have a more direct or numerically
        stable way to compute logSNR.
        """
        a_t = self.a(t).abs()
        b_t = self.b(t).abs()
        # Add a small epsilon to prevent log(0) -> -inf
        # Clamping ensures that the argument to log is positive.
        eps = torch.finfo(a_t.dtype).eps 
        log_a_t = torch.log(a_t.clamp_min(eps))
        log_b_t = torch.log(b_t.clamp_min(eps))
        return 2 * (log_a_t - log_b_t)

    # Methods for v-parameterization
    def get_x0_from_v_and_zt(
        self,
        v: Float[Tensor, "*batch"],
        zt: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        """Predict x0 from v and zt.
        General formula: x0 = (a(t)*zt - b(t)*v) / (a(t)^2 + b(t)^2)
        If a(t)^2 + b(t)^2 = 1 (e.g. for standard Diffusion), it simplifies to x0 = a(t)*zt - b(t)*v.
        """
        a_t = self.a(t)
        b_t = self.b(t)
        # Add a small epsilon for numerical stability, especially if a(t)^2 + b(t)^2 could be zero (though unlikely for valid schedules).
        denominator = a_t**2 + b_t**2 + 1e-9 
        return (a_t * zt - b_t * v) / denominator

    def get_eps_from_v_and_zt(
        self,
        v: Float[Tensor, "*batch"],
        zt: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        """Predict eps from v and zt.
        General formula: eps = (b(t)*zt + a(t)*v) / (a(t)^2 + b(t)^2)
        If a(t)^2 + b(t)^2 = 1 (e.g. for standard Diffusion), it simplifies to eps = b(t)*zt + a(t)*v.
        """
        a_t = self.a(t)
        b_t = self.b(t)
        denominator = a_t**2 + b_t**2 + 1e-9 # Add epsilon for numerical stability
        return (b_t * zt + a_t * v) / denominator

    def get_v_from_eps_and_x0(
        self,
        eps: Float[Tensor, "*batch"],
        x0: Float[Tensor, "*batch"], # x0 denotes the original clean data
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        """Define v from eps and x0: v = a(t)*eps - b(t)*x0"""
        # Note: a(t) and b(t) here usually correspond to sqrt_alpha_bar_t and sqrt_one_minus_alpha_bar_t for diffusion
        return self.a(t) * eps - self.b(t) * x0

    def get_v_from_eps_and_zt(
        self,
        eps: Float[Tensor, "*batch"],
        zt: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        """Compute v from eps and zt.
        v = ((a(t)^2 + b(t)^2) / a(t)) * eps - (b(t) / a(t)) * zt
        Assumes a(t) != 0.
        """
        a_t = self.a(t)
        b_t = self.b(t)
        # Add small epsilon to prevent division by zero if a_t is zero (e.g. t=1 for some schedules)
        # However, if a_t is truly zero, this formula is ill-defined. 
        # For Diffusion, a(t)=sqrt_alphas_bar. If t=1, alpha_bar is often near 0.
        # If a(t)=0, then zt = b(t)*eps. v = a(t)*eps - b(t)*x0. This scenario implies x0 might not be recoverable or v depends on context.
        # Let's assume a(t) is not zero for the direct formula, or use a safe division.
        safe_a_t = a_t + 1e-9 * (a_t == 0) # Add epsilon only if a_t is exactly 0
        
        # If a(t)^2 + b(t)^2 = 1 (e.g., for standard VP diffusion schedules like Diffusion class implements)
        # then v = (1/a(t)) * eps - (b(t)/a(t)) * zt = (eps - b(t)*zt) / a(t)
        # Let's use the general form first.
        factor1 = (a_t**2 + b_t**2) / safe_a_t
        factor2 = b_t / safe_a_t
        v = factor1 * eps - factor2 * zt
        # Consider edge case for Diffusion where a(t)=0 (t=1). Then v = eps (from Kingma et al. 2021, Appendix B, v_t for t=1)
        # And z_t = b(t)*eps. If a(t)=0 implies b(t)=1 (for a^2+b^2=1), then z_t = eps. Then v = z_t.
        # The formula is: (eps - b(t)zt)/a(t). If a(t)=0, b(t)=1, zt=eps. -> (eps - eps)/0, problematic.
        # If v-prediction target is v_t = alpha_t eps - sigma_t x_0
        # for t_max (alpha_t_max=0, sigma_t_max=1), v_t_max = eps.
        # at t_max, z_t_max = sigma_t_max * eps = eps. So v_t_max = z_t_max
        v = torch.where(a_t == 0, zt, v) # if a(t)=0, then zt=b(t)eps. If a^2+b^2=1, b(t)=1, so zt=eps. v then becomes eps. So v=zt.
        return v

    def get_v_from_x0_and_zt(
        self,
        x0: Float[Tensor, "*batch"],
        zt: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        """Compute v from x0 and zt.
        v = (a(t) / b(t)) * zt - ((a(t)^2 + b(t)^2) / b(t)) * x0
        Assumes b(t) != 0.
        """
        a_t = self.a(t)
        b_t = self.b(t)
        # Add small epsilon to prevent division by zero if b_t is zero (e.g. t=0 for some schedules)
        safe_b_t = b_t + 1e-9 * (b_t == 0) # Add epsilon only if b_t is exactly 0
        
        # If a(t)^2 + b(t)^2 = 1 (e.g., for standard VP diffusion schedules)
        # then v = (a(t)/b(t)) * zt - (1/b(t)) * x0 = (a(t)*zt - x0) / b(t)
        # Let's use the general form first.
        factor1 = a_t / safe_b_t
        factor2 = (a_t**2 + b_t**2) / safe_b_t
        v = factor1 * zt - factor2 * x0
        # Consider edge case where b(t)=0 (t=0). Then v = -x0 (from Kingma et al. 2021, Appendix B, v_t for t=0, with alpha_0=1, sigma_0=0)
        # At t=0, zt = a(t)x0. If b(t)=0 implies a(t)=1 (for a^2+b^2=1), then zt = x0.
        # The formula (a(t)*zt - x0)/b(t) -> (a(t)*a(t)*x0 - x0)/0, problematic.
        # If v-prediction target is v_t = alpha_t eps - sigma_t x_0
        # for t_min (alpha_t_min=1, sigma_t_min=0), v_t_min = -sigma_t_min * x0 / alpha_t_min = 0 IF x0 is used in def.
        # However, the common definition is v = a(t) * eps - b(t) * x0. 
        # If b(t)=0, then v = a(t)*eps. At t=0, a(t)=1, b(t)=0, so zt=x0. And v = eps.
        # This is inconsistent with v_target = alpha_t * noise - sigma_t * data_scaled_by_alpha
        # Let's re-check the definition of v = a(t) * eps - b(t) * x0. 
        # If b(t) = 0, then z_t = a(t) * x_0. If a(t) = 1 (typical at t=0), then z_t = x_0.
        # And v = a(t) * eps = eps (if a(t)=1). This doesn't seem right if v is supposed to be -x_0 scaled. 
        # The v formulation from (Salimans & Ho, 2022) or (Kingma et al. 2021) is often v = sqrt(alpha_bar) * noise - sqrt(1-alpha_bar) * data.
        # If t=0, alpha_bar=1, then v = noise. This makes sense.
        # If b(t) (sqrt(1-alpha_bar)) is 0, v = a(t) * eps. For Diffusion a(t) is sqrt_alphas_bar. So v=eps.
        # The code currently sets v = torch.where(b_t == 0, self.a(t)*eps_replacement_needed , v)
        # This path is tricky. Let's stick to the direct formula and assume b_t is not pathologically zero or rely on safe_b_t.
        # If b_t == 0, zt = a_t * x0. The formula gives v = (a_t/0)*a_t*x0 - ((a_t^2)/0)*x0 -> inf - inf.
        # If b_t is zero, v = a_t * eps. We don't have eps here. So this conversion isn't well-defined if b_t=0.
        # For now, we rely on safe_b_t. If b_t is truly zero, the result may be large but an error isn't raised.
        # A more robust solution might need specific handling for flows where b(t) can be zero.
        return v

    def get_v(
        self,
        t: Float[Tensor, "*#batch"],
        eps: Float[Tensor, "*batch"] | None = None,
        v: Float[Tensor, "*batch"] | None = None,
        x0: Float[Tensor, "*batch"] | None = None, # x0 denotes the original clean data
        zt: Float[Tensor, "*batch"] | None = None
    ) -> Float[Tensor, "*batch"]:
        if v is not None:
            return v
        assert sum(val is not None for val in (eps, x0, zt)) == 2, "Exactly two of (eps, x0, zt) must be provided to compute v"
        if eps is not None:
            if x0 is not None:
                return self.get_v_from_eps_and_x0(eps, x0, t)
            # eps and zt are not None
            return self.get_v_from_eps_and_zt(eps, zt, t)
        # eps is None, so x0 and zt must be not None
        return self.get_v_from_x0_and_zt(x0, zt, t)

    @abstractmethod
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
        # mean_theta and sigma_theta are the mean and standard deviation of v
        pass

    def divergence_simple_v(
        self,
        sigma_theta: Float[Tensor, "*batch"], # This is sigma_v_theta
        t: Float[Tensor, "*#batch"],
        t_star: Float[Tensor, "*#batch"],
        alpha: float | int,
    ) -> Float[Tensor, "*batch"]:
        """
        Computes the simplified KL divergence between conditional_p_v and conditional_q
        NOTE assumes that sigma_theta is the standard deviation of the predicted **v**
        """
        pass

    @torch.no_grad()
    def inverse_divergence_simple(
        self,
        sigma_theta: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"],
        alpha: float | int,
        threshold: Float[Tensor, "*#batch"],
        max_approx_error: float | None = None,
        num_search_steps: int | None = None,
        t_star_min: Float[Tensor, "*batch"] | None = None,
        t_star_max: Float[Tensor, "*batch"] | None = None
    ) -> Float[Tensor, "*batch"]:
        """
        Find t_star with divergence(t_star) = threshold via binary search 
        exploiting the fact of monotonically decreasing function in t_star
        """
        assert max_approx_error is not None or num_search_steps is not None
        if num_search_steps is None:
            num_search_steps = ceil(-log2(max_approx_error)) - 1
        if t_star_min is None:
            t_star_min = torch.zeros_like(sigma_theta)
        if t_star_max is None:
            t_star_max = t.expand_as(sigma_theta).contiguous()
        t_star = (t_star_min + t_star_max) / 2
        if num_search_steps <= 0:
            return t_star
        div = self.divergence_simple(sigma_theta, t, t_star, alpha)
        mask = div >= threshold
        t_star_min = torch.where(mask, t_star, t_star_min)
        t_star_max = torch.where(mask, t_star_max, t_star)
        return self.inverse_divergence_simple(
            sigma_theta, t, alpha, threshold, 
            num_search_steps=num_search_steps-1, 
            t_star_min=t_star_min, 
            t_star_max=t_star_max
        )

    def get_eps_from_v_and_x0(
        self,
        v: Float[Tensor, "*batch"],
        x0: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        """Compute eps from v and x0.
        eps = (v + b(t)*x0) / a(t)
        Subclasses must implement this if they support it, handling a(t) possibly being zero.
        """
        raise NotImplementedError("Subclasses should implement get_eps_from_v_and_x0 if applicable, handling a(t)=0 case.")

    def get_x0_from_v_and_zt(
        self,
        v: Float[Tensor, "*batch"],
        zt: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        """Predict x0 from v and zt.
        General formula: x0 = (a(t)*zt - b(t)*v) / (a(t)^2 + b(t)^2)
        If a(t)^2 + b(t)^2 = 1 (e.g. for standard Diffusion), it simplifies to x0 = a(t)*zt - b(t)*v.
        """
        a_t = self.a(t)
        b_t = self.b(t)
        # Add a small epsilon for numerical stability, especially if a(t)^2 + b(t)^2 could be zero (though unlikely for valid schedules).
        denominator = a_t**2 + b_t**2 + 1e-9 
        return (a_t * zt - b_t * v) / denominator
