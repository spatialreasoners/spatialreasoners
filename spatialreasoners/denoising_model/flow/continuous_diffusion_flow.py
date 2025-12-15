from dataclasses import dataclass

import numpy as np  # For pi if torch.pi not available or for direct copy
import torch
from jaxtyping import Float
from torch import Tensor

from spatialreasoners.type_extensions import (
    Parameterization,  # Assuming this is needed for __init__ super call
)

from . import Flow, FlowCfg, register_flow
from .diagonal_gaussian import (
    DiagonalGaussian,  # Likely needed for conditional_p overrides later
)
from .flow import Flow


@dataclass
class ContinuousLinearBetaFlowCfg(FlowCfg):
    beta_min: float = 1e-4
    beta_max: float = 0.02
    # t_epsilon: float = 1e-5 # Small offset for t to avoid issues at t=0 or t=1 if needed, not using for now


@register_flow("continuous_linear_beta", ContinuousLinearBetaFlowCfg)
class ContinuousLinearBetaFlow(Flow[ContinuousLinearBetaFlowCfg]):
    def __init__(
        self, 
        cfg: ContinuousLinearBetaFlowCfg,
        parameterization: Parameterization = "eps" # Default to eps, can be changed
    ):
        super().__init__(cfg, parameterization)
        # self.t_epsilon = cfg.t_epsilon # If using t_epsilon

    def _integral_beta_sds(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        """Computes integral_0^t beta(s)ds for beta(s) = beta_min + s * (beta_max - beta_min)."""
        # Ensure t is clamped to [0, 1] for calculations if it can go outside
        # t_clamped = torch.clamp(t, 0.0, 1.0)
        # For now, assume t is already in [0,1] as per Flow convention for inputs to a(t),b(t)
        integral = self.cfg.beta_min * t + (self.cfg.beta_max - self.cfg.beta_min) * (t**2) / 2.0
        return integral

    def a(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        """Data weight a_t for t: exp(-0.5 * integral_0^t beta(s)ds)"""
        # For t=0, integral_beta_sds(0) = 0, so exp(0) = 1.
        # Using torch.where for explicit boundary condition, though calculation should yield it.
        calculated_a = torch.exp(-0.5 * self._integral_beta_sds(t))
        return torch.where(t == 0, torch.tensor(1.0, device=t.device, dtype=t.dtype), calculated_a)

    def b(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        """Noise weight b_t for t: sqrt(1 - a(t)^2)"""
        a_t = self.a(t)
        # Clamp argument of sqrt to prevent NaN due to precision issues (e.g. 1.0 - (1.0000001)^2 becoming negative)
        one_minus_a_sq = torch.clamp_min(1.0 - a_t**2, 0.0)
        calculated_b = torch.sqrt(one_minus_a_sq)
        # For t=0, a(0)=1, so b(0)=sqrt(1-1)=0.
        return torch.where(t == 0, torch.tensor(0.0, device=t.device, dtype=t.dtype), calculated_b)

    # --- Derivatives (Phase 2) ---
    def _beta_schedule_t(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        """beta(t) = beta_min + t * (beta_max - beta_min)"""
        # t_clamped = torch.clamp(t, 0.0, 1.0) # if t can go outside [0,1]
        return self.cfg.beta_min + (self.cfg.beta_max - self.cfg.beta_min) * t

    def a_prime(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        """Derivative of a(t): -0.5 * beta(t) * a(t)"""
        # If t=0, a(0)=1. a_prime(0) = -0.5 * beta_min.
        return -0.5 * self._beta_schedule_t(t) * self.a(t)

    def b_prime(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        """Derivative of b(t): -a(t)*a_prime(t) / b(t) (for a^2+b^2=1)"""
        a_t = self.a(t)
        b_t = self.b(t)
        a_prime_t = self.a_prime(t)
        
        # Handle b(t) = 0 case (at t=0 for this schedule)
        # If b_t is 0, and a_prime_t is non-zero (if beta_min > 0), then b_prime is infinite.
        # If beta_min = 0, then a_prime_t(0)=0, then b_prime(0) is 0/0.
        # Let's assume beta_min > 0.
        # For practical purposes, if b_t is very small, b_prime_t will be very large.
        # Using safe division.
        safe_b_t = b_t + 1e-9 * (b_t == 0) # Add epsilon only if b_t is exactly 0
        calculated_b_prime = -a_t * a_prime_t / safe_b_t
        
        # What should b_prime(0) be?
        # If beta_min > 0, a_prime(0) = -0.5*beta_min. a(0)=1. b(0)=0. -> infinity.
        # This means ut might not be well-behaved at t=0.
        # If we need a finite value, we might return a large number or handle it specially in ut calculations.
        # For now, let safe_b_t manage it. If b_t is truly zero, result will be large.
        return torch.where(
            b_t == 0,
            torch.sign(-a_t * a_prime_t) * torch.tensor(float('inf'), device=t.device, dtype=t.dtype), # or some large number/NaN
            calculated_b_prime
        )

    # --- UT Parameterization Support (Phase 3) ---
    def a_b_prime_minus_a_prime_b(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        """Computes a(t)*b_prime(t) - a_prime(t)*b(t)
        For a^2+b^2=1, this simplifies to -a_prime(t)/b(t) if b(t)!=0
        or a(t)/b(t) * (beta(t)/2) * (a(t)^2+b(t)^2) = a(t)/b(t) * beta(t)/2
        Let's use the direct definition: a(t)*b_prime(t) - a_prime(t)*b(t)
        """
        a_t = self.a(t)
        b_t = self.b(t)
        a_prime_t = self.a_prime(t)
        b_prime_t = self.b_prime(t) # This will use the potentially Inf b_prime(0)

        # If b_prime_t is Inf at t=0, and b_t is 0 at t=0, this term:
        # a(0)*b_prime(0) - a_prime(0)*b(0) = 1*Inf - a_prime(0)*0 = Inf.
        # This denominator for eps_pred in conditional_p_ut would be Inf.
        
        # Simpler: for a^2+b^2=1, it is -a_prime(t)/b(t)
        # safe_b_t = b_t + 1e-9 * (b_t == 0)
        # val = -a_prime_t / safe_b_t
        # return torch.where(b_t == 0, torch.sign(-a_prime_t) * torch.tensor(float('inf'), device=t.device, dtype=t.dtype), val)
        # This seems more numerically stable if b_prime is problematic.
        # Let's stick to the definition for now.
        return a_t * b_prime_t - a_prime_t * b_t


    def conditional_p_ut(
        self,
        mean_theta: Float[Tensor, "*batch"], # This is u_theta
        z_t: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"],
        t_star: Float[Tensor, "*#batch"],
        alpha: float | int,
        temperature: float | int = 1,
        sigma_theta: Float[Tensor, "*#batch"] | None = None, # This is sigma_u_theta
        v_theta: Float[Tensor, "*#batch"] | None = None
    ) -> DiagonalGaussian:
        u_pred = mean_theta
        
        a_b_m_a_p_b = self.a_b_prime_minus_a_prime_b(t)
        # If a_b_m_a_p_b can be zero or Inf, we need safe division.
        # For VP SDE, a_b_m_a_p_b = -a_prime(t)/b(t). It can be Inf at t=0.
        safe_denom = a_b_m_a_p_b + 1e-9 * (torch.abs(a_b_m_a_p_b) < 1e-9) # an alternative for safe division
        safe_denom = torch.where(torch.isinf(a_b_m_a_p_b), torch.sign(a_b_m_a_p_b) * 1e9 , safe_denom) # Cap inf
        safe_denom = torch.where(safe_denom == 0, 1e-9, safe_denom)


        # eps_pred = (a(t)*ut - a_prime(t)*zt) / (a(t)*b_prime(t) - a_prime(t)*b(t))
        mean_eps_theta = (self.a(t) * u_pred - self.a_prime(t) * z_t) / safe_denom
        
        sigma_eps_theta = None
        if sigma_theta is not None: # sigma_theta is sigma_u_theta
            # sigma_eps_theta = abs(a(t) / (a*b' - a'*b)) * sigma_u_theta
            coeff = torch.abs(self.a(t) / safe_denom)
            if self.a(t).ndim < sigma_theta.ndim: # Ensure coeff is broadcastable
                 coeff = coeff.view([-1] + [1]*(sigma_theta.ndim - coeff.ndim))
            sigma_eps_theta = coeff * sigma_theta

        return self.conditional_p_eps( # Inherited from base Flow
            mean_theta=mean_eps_theta,
            z_t=z_t,
            t=t,
            t_star=t_star,
            alpha=alpha,
            temperature=temperature,
            sigma_theta=sigma_eps_theta,
            v_theta=v_theta
        )

    def divergence_simple_ut(
        self,
        sigma_theta: Float[Tensor, "*batch"], # This is sigma_u_theta
        t: Float[Tensor, "*#batch"],
        t_star: Float[Tensor, "*#batch"],
        alpha: float | int,
    ) -> Float[Tensor, "*batch"]:
        a_b_m_a_p_b = self.a_b_prime_minus_a_prime_b(t)
        safe_denom = a_b_m_a_p_b + 1e-9 * (torch.abs(a_b_m_a_p_b) < 1e-9)
        safe_denom = torch.where(torch.isinf(a_b_m_a_p_b), torch.sign(a_b_m_a_p_b) * 1e9 , safe_denom)
        safe_denom = torch.where(safe_denom == 0, 1e-9, safe_denom)

        coeff = torch.abs(self.a(t) / safe_denom)
        if self.a(t).ndim < sigma_theta.ndim:
            coeff = coeff.view([-1] + [1]*(sigma_theta.ndim - coeff.ndim))
        sigma_eps_theta = coeff * sigma_theta
        
        return self.divergence_simple_eps( # Inherited
            sigma_theta=sigma_eps_theta,
            t=t,
            t_star=t_star,
            alpha=alpha
        )

    # --- Overriding specific helpers for completeness (Phase 4) ---
    # Many of these involve division by a(t), b(t), a_prime(t), b_prime(t)
    # which can be zero at boundaries for this flow.
    
    def get_x_from_eps_and_zt(self, eps, zt, t):
        a_t = self.a(t)
        safe_a_t = a_t + 1e-9 * (a_t == 0)
        return (zt - self.b(t) * eps) / safe_a_t

    def get_x_from_v_and_eps(self, v, eps, t):
        b_t = self.b(t)
        safe_b_t = b_t + 1e-9 * (b_t == 0)
        return (self.a(t) * eps - v) / safe_b_t

    def get_eps_from_x_and_zt(self, x0, zt, t):
        b_t = self.b(t)
        safe_b_t = b_t + 1e-9 * (b_t == 0)
        return (zt - self.a(t) * x0) / safe_b_t

    def get_eps_from_v_and_x0(self, v, x0, t):
        a_t = self.a(t)
        safe_a_t = a_t + 1e-9 * (a_t == 0)
        return (v + self.b(t) * x0) / safe_a_t
        
    # UT related helpers
    def get_x_from_eps_and_ut(self, eps, ut, t):
        a_prime_t = self.a_prime(t)
        safe_a_prime_t = a_prime_t + 1e-9 * (a_prime_t == 0)
        return (ut - self.b_prime(t) * eps) / safe_a_prime_t

    def get_x_from_ut_and_zt(self, ut, zt, t):
        denom = self.a_b_prime_minus_a_prime_b(t)
        safe_denom = denom + 1e-9 * (torch.abs(denom) < 1e-9)
        safe_denom = torch.where(torch.isinf(denom), torch.sign(denom) * 1e9, safe_denom)
        safe_denom = torch.where(safe_denom == 0, 1e-9, safe_denom)
        return (self.b_prime(t) * zt - self.b(t) * ut) / safe_denom

    def get_eps_from_ut_and_x0(self, ut, x0, t):
        b_prime_t = self.b_prime(t)
        safe_b_prime_t = b_prime_t + 1e-9 * (b_prime_t == 0)
        # If b_prime_t is Inf, result can be NaN if a_prime_t also Inf.
        # If b_prime_t=0 and a_prime_t*x0 = ut, then 0/0.
        val = (ut - self.a_prime(t) * x0) / safe_b_prime_t
        return torch.where(
            b_prime_t == 0, 
            torch.tensor(float('nan'), device=t.device, dtype=t.dtype), # Or some other indicator
            val
        )
        
    # get_eps_from_ut_and_zt is used by conditional_p_ut, formula: (a*ut - a_prime*zt) / denom
    # This one is already implicitly handled by conditional_p_ut structure.

    # No need to override get_x0_from_v_and_zt or get_eps_from_v_and_zt as base class versions are general
    # and work for a^2+b^2=1.

@dataclass
class ContinuousCosineLogSNRFlowCfg(FlowCfg):
    logsnr_min: float = -15.0
    logsnr_max: float = 15.0
    shift_factor: float = 0.125 # Added: 1.0 means no shift (log(1)=0)
    # No shift/interpolate for now, use base cosine logSNR mapping

@register_flow("continuous_cosine_logsnr", ContinuousCosineLogSNRFlowCfg)
class ContinuousCosineLogSNRFlow(Flow[ContinuousCosineLogSNRFlowCfg]):
    def __init__(
        self, 
        cfg: ContinuousCosineLogSNRFlowCfg,
        parameterization: Parameterization = "eps" # Default to eps
    ):
        super().__init__(cfg, parameterization)
        # self.t_min_angle: Tensor # Not needed to store if calculated in _get_angles
        # self.t_max_angle: Tensor # Not needed to store if calculated in _get_angles
        # Buffers for constants to ensure they are on the correct device
        self.register_buffer("const_logsnr_max", torch.tensor(self.cfg.logsnr_max, dtype=torch.float64), persistent=False)
        self.register_buffer("const_logsnr_min", torch.tensor(self.cfg.logsnr_min, dtype=torch.float64), persistent=False)
        
        # Calculate and register the actual logSNR shift value
        if cfg.shift_factor <= 0:
            raise ValueError("shift_factor must be positive.")
        actual_logsnr_shift = 2 * torch.log(torch.tensor(cfg.shift_factor, dtype=torch.float64))
        self.register_buffer("actual_logsnr_shift", actual_logsnr_shift, persistent=False)

    def _get_angles(self, t_like: Tensor) -> tuple[Tensor, Tensor]:
        """Helper to get t_min_angle and t_max_angle on the same device as t_like."""
        logsnr_max = self.const_logsnr_max.to(device=t_like.device, dtype=t_like.dtype)
        logsnr_min = self.const_logsnr_min.to(device=t_like.device, dtype=t_like.dtype)
        t_min_angle = torch.arctan(torch.exp(-0.5 * logsnr_max))
        t_max_angle = torch.arctan(torch.exp(-0.5 * logsnr_min))
        return t_min_angle, t_max_angle

    def logsnr_t(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        """Calculates logSNR for continuous time t (0-1)."""
        t_min_angle, t_max_angle = self._get_angles(t)
        # Ensure t is clamped to [0, 1] for the interpolation, though it should be by convention.
        t_clamped = torch.clamp(t, 0.0, 1.0)
        # Argument of tan can be sensitive near pi/2. Add epsilon to avoid tan(pi/2) if t_clamped results in it.
        # angle = t_min_angle + t_clamped * (t_max_angle - t_min_angle)
        # Add small epsilon to prevent tan from exploding if angle is exactly pi/2 or -pi/2
        # For t=0, angle = t_min_angle. For t=1, angle = t_max_angle.
        # Typical range for t_min_angle (logsnr_max=15): ~0.0005. For t_max_angle (logsnr_min=-15): ~1.57 (pi/2)
        # So at t=1, angle can be pi/2. tan(pi/2) is inf. log(inf) is inf. -2*log(inf) is -inf. This is expected for logsnr at t=1.
        angle = t_min_angle + t_clamped * (t_max_angle - t_min_angle)
        tan_angle = torch.tan(angle)
        # If tan_angle is extremely large (angle is pi/2), log(tan_angle) is large, -2*log is very negative.
        # If tan_angle is extremely small (angle is 0), log(tan_angle) is very negative, -2*log is very positive.
        logsnr = -2 * torch.log(tan_angle + 1e-9 * (tan_angle==0)) # Add epsilon if tan_angle is 0
        
        # Add the shift
        logsnr = logsnr + self.actual_logsnr_shift.to(device=logsnr.device, dtype=logsnr.dtype)
        return logsnr

    def _alpha_bar_t(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        logsnr = self.logsnr_t(t)
        alpha_bar = torch.sigmoid(logsnr) # Numerically stable: 1 / (1 + torch.exp(-logsnr))
        return alpha_bar

    def a(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        alpha_bar = self._alpha_bar_t(t)
        calculated_a = torch.sqrt(torch.clamp_min(alpha_bar, 0.0))
        # Ensure a(0) is 1.0. alpha_bar(0) should be sigmoid(logsnr_max). If logsnr_max is large, this is ~1.
        # Ensure a(1) is 0.0. alpha_bar(1) should be sigmoid(logsnr_min). If logsnr_min is very negative, this is ~0.
        # Let's rely on the formula and clamping for boundaries for now, but add specific overrides if needed.
        # Using torch.where for t=0 to guarantee a(0)=1, as per typical VP SDE start.
        # alpha_bar_at_0 = torch.sigmoid(self.const_logsnr_max.to(t.device, t.dtype)) # Should be near 1
        # a_at_0 = torch.sqrt(alpha_bar_at_0)
        # return torch.where(t == 0, a_at_0, calculated_a)
        # return torch.where(t == 0, torch.tensor(1.0, device=t.device, dtype=t.dtype), calculated_a)
        # Removed torch.where(t==0, ...). a(t) is now purely derived from _alpha_bar_t(t).
        # This ensures a(t)^2 + b(t)^2 = 1 based on the (possibly shifted) logSNR schedule.
        return calculated_a


    def b(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        alpha_bar = self._alpha_bar_t(t)
        calculated_b = torch.sqrt(torch.clamp_min(1.0 - alpha_bar, 0.0))
        # alpha_bar_at_0 = torch.sigmoid(self.const_logsnr_max.to(t.device, t.dtype))
        # b_at_0 = torch.sqrt(1.0 - alpha_bar_at_0) # Should be near 0
        # return torch.where(t == 0, b_at_0, calculated_b)
        # return torch.where(t == 0, torch.tensor(0.0, device=t.device, dtype=t.dtype), calculated_b)
        # Removed torch.where(t==0, ...). b(t) is now purely derived from _alpha_bar_t(t).
        return calculated_b

    def _d_logsnr_dt(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        t_min_angle, t_max_angle = self._get_angles(t)
        K1 = t_min_angle
        K2 = t_max_angle - t_min_angle
        
        t_clamped = torch.clamp(t, 0.0, 1.0)
        angle = K1 + t_clamped * K2
        
        # Derivative: -4*K2 / sin(2*angle)
        # sin(2*angle) is 0 if 2*angle = n*pi, so angle = n*pi/2.
        # angle is in [t_min_angle, t_max_angle]. t_min_angle ~ 0, t_max_angle ~ pi/2.
        # So problematic points are angle=0 (t slightly < 0 if K1=0) and angle=pi/2 (t=1).
        
        sin_2_angle = torch.sin(2 * angle)
        # safe_sin_2_angle = sin_2_angle + 1e-9 * (sin_2_angle == 0)
        # A more careful handling for t=0 and t=1 might be needed.
        # At t=0, angle = t_min_angle. If t_min_angle is very small, sin_2_angle is small. d_logsnr_dt large.
        # At t=1, angle = t_max_angle ~ pi/2. sin_2_angle ~ sin(pi) = 0. d_logsnr_dt explodes (negative infinity if K2 > 0).
        
        # Derivative of tan(x) is sec^2(x). Derivative of log(u) is u'/u.
        # d/dt [log(tan(K1+tK2))] = (1/tan(K1+tK2)) * sec^2(K1+tK2) * K2
        # = K2 * cos(angle) / sin(angle) * 1/cos^2(angle) = K2 / (sin(angle)*cos(angle))
        d_log_tan_dt = K2 / (torch.sin(angle) * torch.cos(angle) + 1e-9) # add epsilon to denominator
        d_logsnr = -2 * d_log_tan_dt
        
        # Special handling for boundaries t=0 and t=1 where derivative might be Inf/NaN
        # or needs specific definition based on limits.
        # If K2 is 0 (logsnr_min = logsnr_max), d_logsnr should be 0.
        # This formula might become 0 / (sin*cos + eps) which is 0. Correct.
        
        # If t=1, angle approx pi/2. sin(pi/2)=1, cos(pi/2)=0. Denom approx 0. -> Inf.
        # If t=0, angle approx 0. sin(0)=0, cos(0)=1. Denom approx 0. -> Inf.
        # This suggests derivatives are infinite at boundaries unless K2=0.

        # For now, let's use safe division and see. Problems might arise in a_prime, b_prime.
        # Let's define a very large number for Inf to avoid direct Inf propagation if possible.
        large_number = 1e9 # Placeholder for infinity handling

        is_t0 = (t_clamped == 0)
        is_t1 = (t_clamped == 1)
        
        # When angle is 0 or pi/2, sin(angle)*cos(angle) is 0.
        denom_zero = (torch.sin(angle) * torch.cos(angle)).abs() < 1e-7 # Check if denom is effectively zero

        # If K2 is zero, derivative is zero
        res = torch.where(K2.abs() < 1e-9, torch.tensor(0.0, device=t.device, dtype=t.dtype), -2 * d_log_tan_dt) # K2 can be tensor if logsnr_min/max are
        
        # If denom was zero AND K2 is not zero, result is Inf with sign of -2*K2
        res = torch.where(
            denom_zero & (K2.abs() >= 1e-9),
            torch.sign(-2 * K2) * torch.tensor(float('inf'), device=t.device, dtype=t.dtype),
            res
        )
        return res

    def _d_alpha_bar_dt(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        logsnr = self.logsnr_t(t)
        d_logsnr = self._d_logsnr_dt(t)
        alpha_bar = torch.sigmoid(logsnr) # self._alpha_bar_t(t) is equivalent
        # alpha_bar * (1-alpha_bar) can be written as sigmoid(logsnr) * sigmoid(-logsnr)
        # Derivative of sigmoid(x) is sigmoid(x)*(1-sigmoid(x)). So d(alpha_bar)/dt = alpha_bar*(1-alpha_bar)*d(logsnr)/dt
        d_alpha_bar = alpha_bar * (1.0 - alpha_bar) * d_logsnr
        # If d_logsnr is Inf, d_alpha_bar can be Inf * 0 if alpha_bar is 0 or 1 (at t=0, t=1).
        # e.g. t=1, alpha_bar ~0. d_logsnr ~ -Inf.  0 * -Inf -> NaN. Need to handle this.
        # Limit of x*(1-x)*(-1/x) for x->0+ is approx -1. (if d_logsnr ~ -C/alpha_bar)
        # Limit of x*(1-x)* (1/(1-x)) for x->1- is approx 1. (if d_logsnr ~ C/(1-alpha_bar))
        # This suggests d_alpha_bar_dt might be finite at boundaries.

        # Let's re-evaluate d_alpha_bar_dt more directly from definition of alpha_bar(t) for boundaries.
        # alpha_bar = 1 / (1 + exp(-logsnr)). d(alpha_bar)/d(logsnr) = exp(-logsnr) / (1+exp(-logsnr))^2 = alpha_bar * (1-alpha_bar)
        # This is correct. The NaN comes from Inf * 0.
        
        # If t=0, logsnr=logsnr_max (large pos), alpha_bar ~ 1. (1-alpha_bar) ~ 0.
        # If t=1, logsnr=logsnr_min (large neg), alpha_bar ~ 0.
        # If d_logsnr is Inf at these points, we get 0 * Inf.
        # We need the limits. For VP SDE, g^2(t) = d/dt sigma_t^2 / alpha_t^2 = beta_t.
        # sigma_t^2 = 1 - alpha_bar_t. alpha_t = sqrt(alpha_bar_t).
        # beta_t = - d/dt log(alpha_bar_t).  No this is VE.
        # For VP: beta_t = d/dt (log( (1-alpha_bar_t)/alpha_bar_t )) = d/dt (-logsnr_t)
        # So beta_t = - self._d_logsnr_dt(t).
        # And d_alpha_bar_dt = -beta_t * alpha_bar_t * (1-alpha_bar_t)
        # This definition avoids 0*Inf if beta_t is well defined at boundaries.
        # beta_t for VP from logSNR is indeed -d/dt logSNR. So beta_t = -d_logsnr.
        # Let's use this improved formulation for d_alpha_bar_dt
        beta_t_equiv = -d_logsnr # beta_t for an equivalent VP SDE defined via this alpha_bar schedule
        d_alpha_bar = beta_t_equiv * alpha_bar * (1.0 - alpha_bar)
        # Note: beta_t_equiv can be Inf. Still 0*Inf. 
        
        # The issue is that d_logsnr_dt has singularities. 
        # alpha_bar(t) is smooth [0,1]. Its derivative should be finite, zero at ends.
        # The problem might be in the formula d_logsnr_dt = -4K2/sin(2*angle).
        # Let's test with numerical derivative for d_alpha_bar_dt for a moment in thought.
        # Yes, d_alpha_bar_dt should be 0 at t=0 and t=1 because alpha_bar flattens out.

        # If d_logsnr is Inf, but alpha_bar is 0 or 1, the product is NaN.
        # Manually set derivative to 0 at t=0 and t=1.
        is_t0 = (torch.abs(t - 0.0) < 1e-7)
        is_t1 = (torch.abs(t - 1.0) < 1e-7)
        d_alpha_bar = torch.where(is_t0 | is_t1, torch.tensor(0.0, device=t.device, dtype=t.dtype), d_alpha_bar)
        # And also handle if d_logsnr resulted in NaN (e.g. if K2=0 and denom was also 0)
        d_alpha_bar = torch.where(torch.isnan(d_logsnr), torch.tensor(0.0, device=t.device, dtype=t.dtype), d_alpha_bar)
        return d_alpha_bar

    def a_prime(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        """ a_prime(t) = 0.5 * (1/a(t)) * d_alpha_bar_dt """
        a_t = self.a(t)
        d_alpha_bar = self._d_alpha_bar_dt(t)
        safe_a_t = a_t + 1e-9 * (a_t == 0) # a(t) is 0 at t=1
        # At t=1, a_t=0, d_alpha_bar_dt=0. So 0.5 * (1/eps) * 0 = 0. Correct.
        # At t=0, a_t=1, d_alpha_bar_dt=0. So 0. Correct.
        # This seems fine if d_alpha_bar_dt is correctly 0 at boundaries.
        res = 0.5 * (1.0 / safe_a_t) * d_alpha_bar
        # Explicitly set to 0 at t=0 and t=1 due to d_alpha_bar_dt behavior
        is_t0 = (torch.abs(t - 0.0) < 1e-7)
        is_t1 = (torch.abs(t - 1.0) < 1e-7)
        return torch.where(is_t0 | is_t1, torch.tensor(0.0, device=t.device, dtype=t.dtype), res)

    def b_prime(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        """ b_prime(t) = -0.5 * (1/b(t)) * d_alpha_bar_dt """
        b_t = self.b(t)
        d_alpha_bar = self._d_alpha_bar_dt(t)
        safe_b_t = b_t + 1e-9 * (b_t == 0) # b(t) is 0 at t=0
        # At t=0, b_t=0, d_alpha_bar_dt=0. So -0.5 * (1/eps) * 0 = 0. Correct.
        # At t=1, b_t=1, d_alpha_bar_dt=0. So 0. Correct.
        # This seems fine if d_alpha_bar_dt is correctly 0 at boundaries.
        res = -0.5 * (1.0 / safe_b_t) * d_alpha_bar
        is_t0 = (torch.abs(t - 0.0) < 1e-7)
        is_t1 = (torch.abs(t - 1.0) < 1e-7)
        return torch.where(is_t0 | is_t1, torch.tensor(0.0, device=t.device, dtype=t.dtype), res)

    # --- UT Parameterization Support (Phase 3) ---
    def a_b_prime_minus_a_prime_b(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        """Computes a(t)*b_prime(t) - a_prime(t)*b(t)
        For a^2+b^2=1, this simplifies to -a_prime(t)/b(t) if b(t)!=0
        or a(t)/b(t) * (beta(t)/2) * (a(t)^2+b(t)^2) = a(t)/b(t) * beta(t)/2
        Let's use the direct definition: a(t)*b_prime(t) - a_prime(t)*b(t)
        """
        a_t = self.a(t)
        b_t = self.b(t)
        a_prime_t = self.a_prime(t)
        b_prime_t = self.b_prime(t) # This will use the potentially Inf b_prime(0)

        # If b_prime_t is Inf at t=0, and b_t is 0 at t=0, this term:
        # a(0)*b_prime(0) - a_prime(0)*b(0) = 1*Inf - a_prime(0)*0 = Inf.
        # This denominator for eps_pred in conditional_p_ut would be Inf.
        
        # Simpler: for a^2+b^2=1, it is -a_prime(t)/b(t)
        # safe_b_t = b_t + 1e-9 * (b_t == 0)
        # val = -a_prime_t / safe_b_t
        # return torch.where(b_t == 0, torch.sign(-a_prime_t) * torch.tensor(float('inf'), device=t.device, dtype=t.dtype), val)
        # This seems more numerically stable if b_prime is problematic.
        # Let's stick to the definition for now.
        return a_t * b_prime_t - a_prime_t * b_t


    def conditional_p_ut(
        self,
        mean_theta: Float[Tensor, "*batch"], # This is u_theta
        z_t: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"],
        t_star: Float[Tensor, "*#batch"],
        alpha: float | int,
        temperature: float | int = 1,
        sigma_theta: Float[Tensor, "*#batch"] | None = None, # This is sigma_u_theta
        v_theta: Float[Tensor, "*#batch"] | None = None
    ) -> DiagonalGaussian:
        u_pred = mean_theta
        
        a_b_m_a_p_b = self.a_b_prime_minus_a_prime_b(t)
        # If a_b_m_a_p_b can be zero or Inf, we need safe division.
        # For VP SDE, a_b_m_a_p_b = -a_prime(t)/b(t). It can be Inf at t=0.
        safe_denom = a_b_m_a_p_b + 1e-9 * (torch.abs(a_b_m_a_p_b) < 1e-9) # an alternative for safe division
        safe_denom = torch.where(torch.isinf(a_b_m_a_p_b), torch.sign(a_b_m_a_p_b) * 1e9 , safe_denom) # Cap inf
        safe_denom = torch.where(safe_denom == 0, 1e-9, safe_denom)


        # eps_pred = (a(t)*ut - a_prime(t)*zt) / (a(t)*b_prime(t) - a_prime(t)*b(t))
        mean_eps_theta = (self.a(t) * u_pred - self.a_prime(t) * z_t) / safe_denom
        
        sigma_eps_theta = None
        if sigma_theta is not None: # sigma_theta is sigma_u_theta
            # sigma_eps_theta = abs(a(t) / (a*b' - a'*b)) * sigma_u_theta
            coeff = torch.abs(self.a(t) / safe_denom)
            if self.a(t).ndim < sigma_theta.ndim: # Ensure coeff is broadcastable
                 coeff = coeff.view([-1] + [1]*(sigma_theta.ndim - coeff.ndim))
            sigma_eps_theta = coeff * sigma_theta

        return self.conditional_p_eps( # Inherited from base Flow
            mean_theta=mean_eps_theta,
            z_t=z_t,
            t=t,
            t_star=t_star,
            alpha=alpha,
            temperature=temperature,
            sigma_theta=sigma_eps_theta,
            v_theta=v_theta
        )

    def divergence_simple_ut(
        self,
        sigma_theta: Float[Tensor, "*batch"], # This is sigma_u_theta
        t: Float[Tensor, "*#batch"],
        t_star: Float[Tensor, "*#batch"],
        alpha: float | int,
    ) -> Float[Tensor, "*batch"]:
        a_b_m_a_p_b = self.a_b_prime_minus_a_prime_b(t)
        safe_denom = a_b_m_a_p_b + 1e-9 * (torch.abs(a_b_m_a_p_b) < 1e-9)
        safe_denom = torch.where(torch.isinf(a_b_m_a_p_b), torch.sign(a_b_m_a_p_b) * 1e9 , safe_denom)
        safe_denom = torch.where(safe_denom == 0, 1e-9, safe_denom)

        coeff = torch.abs(self.a(t) / safe_denom)
        if self.a(t).ndim < sigma_theta.ndim:
            coeff = coeff.view([-1] + [1]*(sigma_theta.ndim - coeff.ndim))
        sigma_eps_theta = coeff * sigma_theta
        
        return self.divergence_simple_eps( # Inherited
            sigma_theta=sigma_eps_theta,
            t=t,
            t_star=t_star,
            alpha=alpha
        )

    # --- Overriding specific helpers for completeness (Phase 4) ---
    # Many of these involve division by a(t), b(t), a_prime(t), b_prime(t)
    # which can be zero at boundaries for this flow.
    
    def get_x_from_eps_and_zt(self, eps, zt, t):
        a_t = self.a(t)
        safe_a_t = a_t + 1e-9 * (a_t == 0)
        return (zt - self.b(t) * eps) / safe_a_t

    def get_x_from_v_and_eps(self, v, eps, t):
        b_t = self.b(t)
        safe_b_t = b_t + 1e-9 * (b_t == 0)
        return (self.a(t) * eps - v) / safe_b_t

    def get_eps_from_x_and_zt(self, x0, zt, t):
        b_t = self.b(t)
        safe_b_t = b_t + 1e-9 * (b_t == 0)
        return (zt - self.a(t) * x0) / safe_b_t

    def get_eps_from_v_and_x0(self, v, x0, t):
        a_t = self.a(t)
        safe_a_t = a_t + 1e-9 * (a_t == 0)
        return (v + self.b(t) * x0) / safe_a_t
        
    # UT related helpers
    def get_x_from_eps_and_ut(self, eps, ut, t):
        a_prime_t = self.a_prime(t)
        safe_a_prime_t = a_prime_t + 1e-9 * (a_prime_t == 0)
        return (ut - self.b_prime(t) * eps) / safe_a_prime_t

    def get_x_from_ut_and_zt(self, ut, zt, t):
        denom = self.a_b_prime_minus_a_prime_b(t)
        safe_denom = denom + 1e-9 * (torch.abs(denom) < 1e-9)
        safe_denom = torch.where(torch.isinf(denom), torch.sign(denom) * 1e9, safe_denom)
        safe_denom = torch.where(safe_denom == 0, 1e-9, safe_denom)
        return (self.b_prime(t) * zt - self.b(t) * ut) / safe_denom

    def get_eps_from_ut_and_x0(self, ut, x0, t):
        b_prime_t = self.b_prime(t)
        safe_b_prime_t = b_prime_t + 1e-9 * (b_prime_t == 0)
        # If b_prime_t is Inf, result can be NaN if a_prime_t also Inf.
        # If b_prime_t=0 and a_prime_t*x0 = ut, then 0/0.
        val = (ut - self.a_prime(t) * x0) / safe_b_prime_t
        return torch.where(
            b_prime_t == 0, 
            torch.tensor(float('nan'), device=t.device, dtype=t.dtype), # Or some other indicator
            val
        )
        
    # get_eps_from_ut_and_zt is used by conditional_p_ut, formula: (a*ut - a_prime*zt) / denom
    # This one is already implicitly handled by conditional_p_ut structure.

    # No need to override get_x0_from_v_and_zt or get_eps_from_v_and_zt as base class versions are general
    # and work for a^2+b^2=1.

    # --- V Parameterization Support ---
    def conditional_p_v(
        self,
        mean_theta: Float[Tensor, "*batch"], # This is v_theta
        z_t: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"],
        t_star: Float[Tensor, "*#batch"],
        alpha: float | int,
        temperature: float | int = 1,
        sigma_theta: Float[Tensor, "*#batch"] | None = None, # This is sigma_v_theta
        v_theta: Float[Tensor, "*#batch"] | None = None # This v_theta is for variance interpolation
    ) -> DiagonalGaussian:
        # Convert v_theta to eps_theta.
        # Since a(t)^2 + b(t)^2 = 1 for this flow, the general get_eps_from_v_and_zt simplifies to:
        # eps_pred = b(t)*z_t + a(t)*v_pred
        mean_eps_theta = self.get_eps_from_v_and_zt(mean_theta, z_t, t)
        
        sigma_eps_theta = None
        if sigma_theta is not None: # sigma_theta here is sigma_v_theta
            # sigma_eps_theta = a(t) * sigma_v_theta
            a_t = self.a(t)
            # Ensure a_t is broadcastable with sigma_theta if sigma_theta has extra dims (e.g. feature dim)
            if a_t.ndim < sigma_theta.ndim:
                 a_t = a_t.view([-1] + [1]*(sigma_theta.ndim - a_t.ndim))
            sigma_eps_theta = a_t * sigma_theta

        return self.conditional_p_eps( # Inherited from base Flow
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
        
        return self.divergence_simple_eps( # Inherited
            sigma_theta=sigma_eps_theta,
            t=t,
            t_star=t_star,
            alpha=alpha
        )
