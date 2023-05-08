from cmath import isclose
from turtle import forward
from matplotlib.pyplot import step
from numpy import choose
from sbi.inference.posteriors.vi_posterior import VIPosterior
from sbi.samplers.vi import get_default_flows, get_flow_builder

from abc import abstractmethod, ABC
from torch.distributions import Distribution
from typing import Callable, Dict, Optional, Tuple, Union
from torch import Tensor
import torch
from torch.nn import Module
import math
from torch.distributions.multivariate_normal import _batch_mahalanobis, _batch_mv
from rbi.utils.distributions import SIRDistribution, sample_sir

from rbi.utils.streaming_estimators import ExponentialMovingAverageEstimator


class Kernel(ABC, Module):
    """This is a general abstract class for an MCMC kernel."""

    _symmetric: bool
    _requires_metropolis_hasting: bool
    _jit_able: bool
    device: str

    def __init__(self) -> None:
        super().__init__()
        self._symmetric = False
        self._requires_metropolis_hasting = True
        self._jit_able = False
        self.device = "cpu"

    def to(self, device):
        self.device = device
        super().to(device)

    @abstractmethod
    def forward(self, x: Tensor, x_new: Tensor) -> Tensor:
        """Evaluates the kernel potential

        Args:
            x (Tensor): Old x
            x_new (Tensor):New x

        Returns:
            Tensor: Potential k(x, x_new)
        """
        pass

    @abstractmethod
    def sample(self, x: Tensor) -> Tensor:
        """Samples from the kernel given old x

        Args:
            x (Tensor): Old x

        Returns:
            Tensor: New x
        """
        pass

    @torch.jit.export  # type: ignore
    def update_params(self: Module, x: Tensor, x_new: Tensor, logalpha: Tensor) -> None:
        """Updates parameters of the kernel.

        Args:
            x (Tensor): Old x.
            x_new (_type_): New x.
            alpha (_type_): Acceptance probabilities.
        """
        pass

    @torch.jit.export  # type: ignore
    def init_params(self, x: Tensor) -> None:
        """Initializes the parameters"""
        pass


class GaussianKernel(Kernel):
    """A simple Gaussian kernel with isotropic fixed variance i.e. step_size."""

    step_size: float
    var: float

    def __init__(self, step_size: float = 0.5) -> None:
        super(GaussianKernel, self).__init__()
        # The step size is used to generate a random sample
        self.step_size = step_size
        # Variance is equal to step_size^2
        self.var = step_size**2
        self._symmetric = True
        self._jit_able = True

    @torch.jit.export  # type: ignore
    def forward(self, x: Tensor, x_new: Tensor) -> Tensor:
        # Returns the potential of the kernel k(x, x_new)
        return torch.sum(-((x - x_new) ** 2) / (2 * self.var), -1)

    @torch.jit.export  # type: ignore
    def sample(self, x: Tensor) -> Tensor:
        # Returns a new x sample generated from the Gaussian distribution
        return x + self.step_size * torch.randn_like(x, device=self.device)


class AdaptiveGaussianKernel(Kernel):
    """Adaptive Gaussian kernel with diagonal adaptive variance and step_size.

    This kernel adapts to the distributions of the sample space.
    The step_size  is adapted based on the acceptance rate.
    The mean and variance is estimated as a exponential moving average.

    Args:
        acceptance_probability_goal (float, optional): The target acceptance rate. Defaults to 0.25.
        initial_step_size (float, optional): The initial step size. Defaults to 0.2.
        min_var (float, optional): The minimum variance for numerical stability. Defaults to 1e-3.
        gamma (float, optional): The decay rate for the exponential moving average of variance and step size. Defaults to 0.5.

    """

    _jit_able: bool
    acceptance_probability_goal: float
    initial_step_size: float
    min_var: float
    mean: Tensor
    var: Tensor
    gamma: float
    log_step_size: Tensor
    min_log_step_size: float
    max_log_step_size: float
    adaptive_step_size: bool
    steps: int

    def __init__(
        self,
        acceptance_probability_goal: float = 0.234,
        initial_step_size: float = 2.38,
        min_var: float = 1e-3,
        gamma: float = 0.5,
        min_log_step_size: float = -4.0,
        max_log_step_size: float = 3.0,
        adaptive_step_size: bool = True,
    ) -> None:
        super().__init__()
        # Target acceptance rate
        self.acceptance_probability_goal = acceptance_probability_goal
        # Decay rate for the exponential moving average
        self.gamma = gamma
        # Minimum variance for numerical stability
        self.min_var = min_var
        # Moving average estimator for the mean
        self.mean = torch.zeros(1)
        # Moving average estimator for the variance
        self.var = torch.ones(1)
        # Initial step size
        self.initial_step_size = initial_step_size
        # Logarithm of the step size
        self.log_step_size = torch.ones(1)
        self.min_log_step_size = min_log_step_size
        self.max_log_step_size = max_log_step_size

        self.adaptive_step_size = adaptive_step_size
        self.steps = 0

        self._jit_able = True
        self._symmetric = True

    @torch.jit.export  # type: ignore
    def forward(self, x: Tensor, x_new: Tensor) -> Tensor:
        """Evaluates the kernel potential.

        Args:
            x (Tensor): Old x.
            x_new (Tensor): New x.

        Returns:
            Tensor: The potential k(x, x_new).

        """
        step_size = self.log_step_size.exp().unsqueeze(-1)
        var = step_size * self.var + self.min_var

        return torch.sum(
            -((x - x_new) ** 2) / (2 * var) - 0.5 * torch.log(var),
            -1,
        )

    @torch.jit.export  # type: ignore
    def sample(self, x: Tensor) -> Tensor:
        """Samples from the kernel given old x.

        Args:
            x (Tensor): Old x.

        Returns:
            Tensor: New x, sampled from the kernel.

        """
        step_size = self.log_step_size.exp().unsqueeze(-1)
        var = step_size * self.var + self.min_var
        return x + var.sqrt() * torch.randn_like(x, device=self.device)

    @torch.jit.export  # type: ignore
    def update_params(self, x: Tensor, x_new: Tensor, alpha: Tensor):
        # Update the mean of the distribution
        self.steps += 1

        new_mean = (self.steps - 1) / self.steps * self.mean + 1 / self.steps * x
        self.var = (self.steps - 1) / self.steps * self.var + 1 / self.steps * (
            x**2 - self.steps * new_mean**2 + (self.steps - 1) * self.mean**2
        )
        self.mean = new_mean

        # Update the log step size used in sampling
        if self.adaptive_step_size:
            self.log_step_size = self.log_step_size + self.gamma * 100 / self.steps * (
                alpha - self.acceptance_probability_goal
            )
            self.log_step_size = self.log_step_size.clamp(
                min=self.min_log_step_size, max=self.max_log_step_size
            )

    @torch.jit.export  # type: ignore
    def init_params(self, x: Tensor):
        # Initialize the mean of the distribution
        self.mean = x.to(self.device)
        self.steps = 1

        # Initialize the covariance of the distribution
        self.var = torch.ones_like(x, device=self.device)

        # Initialize the log step size used in sampling
        self.log_step_size = (
            math.log(self.initial_step_size)
            * torch.ones(x.shape[:-1], device=self.device)
            / x.shape[-1]
        )


class AdaptiveMultivariateGaussianKernel(Kernel):
    """Adaptive Multivariate Gaussian kernel with diagonal adaptive variance and step_size."""

    acceptance_probability_goal: float
    initial_step_size: float
    min_var: float
    gamma: float
    mean: Tensor
    covar: Tensor
    log_step_size: Tensor
    min_log_step_size: float
    max_log_step_size: float
    adaptive_step_size: bool
    steps: int

    def __init__(
        self,
        acceptance_probability_goal: float = 0.234,
        initial_step_size: float = 2.38,
        min_var: float = 1e-1,
        gamma: float = 0.5,
        min_log_step_size: float = -4.0,
        max_log_step_size: float = 3.0,
        adaptive_step_size: bool = True,
    ) -> None:
        # Call the parent class's __init__ method to initialize the class
        super().__init__()

        # The desired acceptance probability for the kernel
        self.acceptance_probability_goal = acceptance_probability_goal

        # The exponential decay rate for the moving average estimators
        self.gamma = gamma

        # The minimum value for the variance
        self.min_var = min_var

        # The mean and variance of the distribution
        self.mean = torch.zeros(1)
        self.covar = torch.ones(1)
        self.scale_tril = torch.ones(1)

        # The initial step size used in sampling
        self.initial_step_size = initial_step_size
        self.min_log_step_size = min_log_step_size
        self.max_log_step_size = max_log_step_size
        self.adaptive_step_size = adaptive_step_size
        self.steps = 0

        # The logarithm of the step size used in sampling
        self.log_step_size = torch.ones(1)

        self._jit_able = True
        self._symmetric = True

    def forward(self, x: Tensor, x_new: Tensor) -> Tensor:
        step_size = self.log_step_size.exp().unsqueeze(-1).unsqueeze(-1)
        scale_tril = (
            step_size * self.scale_tril
            + torch.eye(x.shape[-1], device=self.device) * self.min_var
        )

        # Calculate the log probability of `x_new` under the current distribution
        if x.shape[-1] > 1:
            M = _batch_mahalanobis(scale_tril, x_new - x)
            half_log_det = scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        else:
            M = (x_new - x) ** 2 / scale_tril.squeeze(-1) ** 2
            half_log_det = scale_tril.log().squeeze(-1)

        res = -0.5 * M - half_log_det
        return res.reshape(x.shape[0], x.shape[1])

    @torch.jit.export  # type: ignore
    def sample(self, x: Tensor) -> Tensor:
        step_size = self.log_step_size.exp().unsqueeze(-1).unsqueeze(-1)
        scale_tril = (
            step_size * self.scale_tril
            + torch.eye(x.shape[-1], device=self.device) * self.min_var
        )
        # Sample from the current distribution

        if x.shape[-1] > 1:
            return x + _batch_mv(scale_tril, torch.randn_like(x, device=self.device))
        else:
            return x + scale_tril.squeeze(-1) * torch.randn_like(x, device=self.device)

    @torch.jit.export  # type: ignore
    def update_params(self, x: Tensor, x_new: Tensor, alpha: Tensor):
        # Update the mean of the distribution
        self.steps += 1
        new_mean = (self.steps - 1) / self.steps * self.mean + 1 / self.steps * x
        self.covar = (self.steps - 1) / self.steps * self.covar + 1 / self.steps * (
            x[..., :, None] * x[..., None, :]
            - self.steps * new_mean[..., :, None] * new_mean[..., None, :]
            + (self.steps - 1) * self.mean[..., :, None] * self.mean[..., None, :]
        )
        self.mean = new_mean
        self.scale_tril = torch.linalg.cholesky(self.covar)

        # Update the log step size used in sampling
        if self.adaptive_step_size:
            self.log_step_size = self.log_step_size + self.gamma * 100 / self.steps * (
                alpha - self.acceptance_probability_goal
            )
            self.log_step_size = self.log_step_size.clamp(
                min=self.min_log_step_size, max=self.max_log_step_size
            )

    @torch.jit.export  # type: ignore
    def init_params(self, x: Tensor):
        # Initialize the mean of the distribution
        self.mean = x
        self.steps = 1

        # Initialize the covariance of the distribution
        self.covar = torch.eye(x.shape[-1], device=self.device).repeat(
            x.shape[:-1] + (1, 1)
        )
        self.scale_tril = torch.linalg.cholesky(self.covar)

        # Initialize the log step size used in sampling
        self.log_step_size = (
            math.log(self.initial_step_size)
            * torch.ones(x.shape[:-1], device=self.device)
            / x.shape[-1]
        )


class LangevianKernel(Kernel):
    step_size: Tensor
    adaptive_step_size: bool
    acceptance_probability_goal: float
    log_step_size: Tensor
    min_log_step_size: float
    max_log_step_size: float
    gamma: float
    steps: int
    _grad_cache: Tuple[Tensor, Tensor, Optional[Tensor]]

    def __init__(
        self,
        potential_fn: Callable,
        context: Optional[Tensor] = None,
        step_size: float = 1e-2,
        always_accept: bool = False,
        adaptive_step_size: bool = True,
        acceptance_probability_goal: float = 0.567,
        gamma: float = 0.5,
        min_log_step_size: float = -5.0,
        max_log_step_size: float = 1.0,
    ) -> None:
        # Call the parent class's constructor
        super().__init__()

        # Store the step size, potential function, and adaptive step size flag
        self.step_size = torch.as_tensor(step_size, device=self.device)
        self.potential_fn = potential_fn
        self.adaptive_step_size = adaptive_step_size
        self.context = context

        # Store the gradient cache
        self._grad_cache = (torch.ones(1, device=self.device), torch.ones(1, device=self.device), torch.ones(1, device=self.device))

        # Store the acceptance probability goal and the log of the step size
        self.acceptance_probability_goal = acceptance_probability_goal
        self.log_step_size = torch.ones(1)
        self.min_log_step_size = min_log_step_size
        self.max_log_step_size = max_log_step_size
        self.steps = 0

        # Store the gamma value
        self.gamma = gamma

        # Check if the "always accept" flag is set, and set the metropolis hasting flag accordingly
        if always_accept:
            self._requires_metropolis_hasting = False
            self.adaptive_step_size = False

        self._jit_able = True

    @torch.jit.export  # type: ignore
    def _eval_potential_and_grad(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        x_cached, loss, x_grad_cached = self._grad_cache

        if torch.allclose(x_cached, x):
            return loss, x_grad_cached
        else:
            x = x.detach().clone().requires_grad_(True)
            if self.context is None:
                loss = self.potential_fn(x)
            else:
                loss = self.potential_fn(self.context, x)
            x_grad = torch.autograd.grad(
                [
                    loss.sum(),
                ],
                [
                    x,
                ],
            )[0]

            self._grad_cache = (x.detach(), loss.detach(), x_grad)
            return loss.detach(), x_grad

    def forward(self, x: Tensor, x_new: Tensor) -> Tensor:
        # Determine the step size based on whether adaptive step size is enabled
        if self.adaptive_step_size:
            step_size = torch.exp(self.log_step_size).unsqueeze(-1)
        else:
            step_size = self.step_size

        # Check if the gradient for x is cached
        potential, x_grad = self._eval_potential_and_grad(x_new)

        if x_grad is not None:  # JIT compatibility
            # Return the kernel value
            return (
                -torch.sum((x_new + step_size * x_grad - x) ** 2, dim=-1, keepdim=True)
                / (4 * step_size)
            ).squeeze(-1)
        else:
            return torch.zeros(1, device=self.device)

    @torch.jit.export  # type: ignore
    def sample(self, x: Tensor) -> Tensor:
        # Determine the step size based on whether adaptive step size is enabled
        if self.adaptive_step_size:
            step_size = torch.exp(self.log_step_size).unsqueeze(-1)
        else:
            step_size = self.step_size

        # Compute the gradient for x
        potential, x_grad = self._eval_potential_and_grad(x)

        if x_grad is not None:
            x_new = (
                x.detach()
                + step_size * x_grad
                + torch.sqrt(2 * step_size) * torch.randn_like(x, device=self.device)
            )
        else:
            x_new = x
        return x_new

    @torch.jit.export  # type: ignore
    def update_params(self, x: Tensor, x_new: Tensor, alpha: Tensor):
        if self.adaptive_step_size:
            self.steps += 1
            self.log_step_size = self.log_step_size + self.gamma / self.steps * (
                alpha - self.acceptance_probability_goal
            )
            self.log_step_size = self.log_step_size.clamp(
                min=self.min_log_step_size, max=self.max_log_step_size
            )

    @torch.jit.export  # type: ignore
    def init_params(self, x: Tensor):
        self.log_step_size = torch.log(self.step_size) * torch.ones(x.shape[:-1], device=self.device)
        self.steps = 1


class HMCKernel(Kernel):
    xs: Tuple[Tensor, Tensor]
    energies: Tuple[Tensor, Tensor]
    adaptive_step_size: bool
    acceptance_probability_goal: float
    log_step_size: Tensor
    min_log_step_size: float
    max_log_step_size: float
    gamma: float
    steps: int

    def __init__(
        self,
        potential_fn: Callable,
        context: Optional[Tensor] = None,
        num_steps: int = 5,
        step_size: float = 0.1,
        acceptance_probability_goal: float = 0.65,
        adaptive_step_size: bool = True,
        min_log_step_size: float = -8.0,
        max_log_step_size: float = 0.0,
        gamma: float = 0.5,
    ):
        super().__init__()
        self.potential_fn = potential_fn
        self.context = context
        self.num_steps = num_steps
        self.step_size = torch.tensor(step_size)
        self.xs_cached = (torch.ones(1), torch.ones(1))
        self.energies = (torch.ones(1), torch.ones(1))

        self.acceptance_probability_goal = acceptance_probability_goal
        self.adaptive_step_size = adaptive_step_size
        self.log_step_size = torch.ones(1)
        self.min_log_step_size = min_log_step_size
        self.max_log_step_size = max_log_step_size
        self.gamma = gamma
        self.steps = 0

        self._jit_able = True

    @torch.jit.export  # type: ignore
    def forward(self, x: Tensor, x_new: Tensor) -> Tensor:
        if torch.allclose(x, self.xs_cached[0]):
            return -self.energies[1]
        else:
            return -self.energies[0]

    @torch.jit.export  # type: ignore
    def _compute_grad_potential(self, x: Tensor) -> Optional[Tensor]:
        if self.context is None:
            potential = self.potential_fn(x)
        else:
            potential = self.potential_fn(self.context, x)

        grad_potential = torch.autograd.grad(
            [
                -potential.sum(),
            ],
            [
                x,
            ],  # List is needed for jit compiler
        )[0]

        return grad_potential

    @torch.jit.export  # type: ignore
    def sample(self, x: torch.Tensor) -> torch.Tensor:
        # Initialize x_new as a clone of x
        x_new = x.clone()

        # Generate random momentum and calculate its kinetic energy
        momentum = torch.randn_like(x, device=self.device)
        kinetic_energy = 0.5 * momentum.pow(2).sum(dim=-1)

        if not self.adaptive_step_size:
            step_size = self.step_size
        else:
            step_size = self.log_step_size.exp().unsqueeze(-1)

        # Verlet integrator
        for i in range(self.num_steps):
            # Set x_new to require gradient computation
            x_new.requires_grad_()
            # Compute the gradient of the negative potential energy wrt x_new
            grad_potential = self._compute_grad_potential(x_new)
            if grad_potential is not None:
                # Update the momentum by half the step size times the gradient of the potential energy
                momentum -= step_size * grad_potential / 2
                # Update the position of x_new by the step size times the momentum
                x_new.data.add_(step_size * momentum)
            # Compute the gradient of the negative potential energy wrt x_new again
            grad_potential = self._compute_grad_potential(x_new)
            if grad_potential is not None:  # Jit typing...
                # Update the momentum again by half the step size times the gradient of the potential energy
                momentum -= step_size * grad_potential / 2

        # Calculate the new kinetic energy
        new_kinetic_energy = 0.5 * momentum.pow(2).sum(dim=-1)
        # Detach x_new from the computation graph and store the new kinetic energy in the cache
        x_new = x_new.detach()

        self.xs_cached = (x, x_new)
        self.energies = (
            torch.nan_to_num(kinetic_energy, nan=0.0, posinf=10000.0, neginf=0.0),
            torch.nan_to_num(new_kinetic_energy, nan=0.0, posinf=10000.0, neginf=0.0),
        )

        # Return the new position x_new
        return x_new

    @torch.jit.export  # type: ignore
    def update_params(self, x: Tensor, x_new: Tensor, alpha: Tensor):
        if self.adaptive_step_size:
            self.steps += 1
            self.log_step_size = self.log_step_size + self.gamma * 100 / self.steps * (
                alpha - self.acceptance_probability_goal
            )
            self.log_step_size = self.log_step_size.clamp(
                min=self.min_log_step_size, max=self.max_log_step_size
            ).nan_to_num(nan=self.min_log_step_size, posinf=self.min_log_step_size)

    @torch.jit.export  # type: ignore
    def init_params(self, x: Tensor):
        self.log_step_size = torch.log(self.step_size) * torch.ones(x.shape[:-1], device=self.device)
        self.steps = 1


class SliceKernel(Kernel):
    _requires_metropolis_hasting: bool
    step_size: float
    bracket_expansion_increase: float
    adaptive_direction: bool
    mean: Tensor
    covar: Tensor
    scale_tril: Tensor

    def __init__(
        self,
        potential_fn: Callable,
        context: Optional[Tensor] = None,
        step_size: float = 0.1,
        bracket_expansion_increase: float = 1.5,
        adaptive_direction: bool = False,
        min_var: float = 1e-1,
    ) -> None:
        """Performs slice sampling for each iteration. Each sample is accepted.

        Args:
            potential_fn (Callable): Potential function
            step_size (float, optional): Step size for expanding the slice. Defaults to 0.1.
            adaptive_direction (bool, optional): If slice direction should be adapted. Defaults to True.
            gamma (float, optional): Exponential moving average for learning slice directions. Defaults to 0.3.
            min_var (float, optional): Min variance for adaptive slice directions. Defaults to 5e-2.
        """
        super().__init__()
        self.potential_fn = potential_fn
        self.step_size = step_size
        self.bracket_expansion_increase = bracket_expansion_increase
        self.adaptive_directions = adaptive_direction
        self.min_var = min_var
        self.context = context
        self.mean = torch.zeros(1)
        self.covar = torch.ones(1)
        self.scale_tril = torch.ones(1)
        self.steps = 0

        self._requires_metropolis_hasting = False
        self._jit_able = True

    @torch.jit.export
    def _sample_random_directions(self, x: Tensor) -> Tensor:
        """Return a direction which defines a slice from which we sample uniformly.

        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: Direction in which to slice.
        """
        if self.adaptive_directions and self.steps > 10:
            directions = _batch_mv(
                self.scale_tril + self.min_var * torch.eye(x.shape[-1]),
                torch.randn_like(x, device=self.device),
            )
            directions /= torch.linalg.norm(directions, dim=-1, keepdim=True)
        else:
            directions = torch.randn(x.shape, device=self.device)
            directions /= torch.linalg.norm(directions, dim=-1, keepdim=True)

        return directions

    @torch.jit.export  # type: ignore
    def _eval_potential_masked(self, x: Tensor, mask: Tensor) -> Tensor:
        x_masked = x[mask]
        if self.context is None:
            potential = self.potential_fn(x_masked)
        else:
            context_masked = self.context.repeat(x.shape[0], 1, 1)[mask]
            potential = self.potential_fn(
                context_masked.unsqueeze(-2), x_masked.unsqueeze(-2)
            )  # Unsqueeze for jit support
        return potential

    @torch.jit.export  # type: ignore
    def _eval_potential(self, x: Tensor) -> Tensor:
        if self.context is None:
            potential = self.potential_fn(x)
        else:
            potential = self.potential_fn(self.context, x)
        return potential

    @torch.jit.export
    def sample(self, x: Tensor) -> Tensor:
        """Return a sample

        Args:
            x (Tensor): Old x.

        Returns:
            Tensor: New x.
        """
        self.steps += 1
        batch_shape = x.shape[:-1]
        potential = self._eval_potential(x)
        y = torch.log(1 - torch.rand_like(potential, device=self.device)) + potential

        random_directions = self._sample_random_directions(x)

        lb = torch.zeros(batch_shape, device=self.device)
        ub = torch.zeros(batch_shape, device=self.device)
        mask_lb = y < potential
        mask_ub = mask_lb.clone()

        i = 0
        while mask_lb.any():
            lb[mask_lb] -= self.step_size * (self.bracket_expansion_increase) ** i
            x_lb = x + lb.unsqueeze(-1) * random_directions
            potential_lb = self._eval_potential_masked(x_lb, mask_lb)
            mask_lb[mask_lb.clone()] = y[mask_lb] < potential_lb
            i += 1

        j = 0
        while mask_ub.any():
            ub[mask_ub] += self.step_size * (self.bracket_expansion_increase) ** j
            x_ub = x + ub.unsqueeze(-1) * random_directions
            potential_ub = self._eval_potential_masked(x_ub, mask_ub)
            mask_ub[mask_ub.clone()] = y[mask_ub] < potential_ub
            j += 1

        u = torch.rand(batch_shape, device=self.device) * (ub - lb) + lb
        x_new = x + u.unsqueeze(-1) * random_directions

        potential_new = self._eval_potential(x_new)
        mask_reject = potential_new < y

        while mask_reject.any():
            mask_smaller = u < 0

            lb[mask_reject.reshape(batch_shape) & mask_smaller] = u[
                mask_reject.reshape(batch_shape) & mask_smaller
            ]
            ub[mask_reject.reshape(batch_shape) & ~mask_smaller] = u[
                mask_reject.reshape(batch_shape) & ~mask_smaller
            ]
            u = torch.rand(batch_shape, device=self.device) * (ub - lb) + lb
            x_new_rej = x + u.unsqueeze(-1) * random_directions

            potential_new = self._eval_potential_masked(x_new_rej, mask_reject)
            x_new[mask_reject] = x_new_rej[mask_reject]
            mask_reject[mask_reject.clone()] = potential_new < y[mask_reject]

        return x_new

    @torch.jit.export
    def forward(self, x: Tensor, x_new: Tensor) -> Tensor:
        return torch.zeros(1, device=self.device)

    @torch.jit.export  # type: ignore
    def update_params(self, x: Tensor, x_new: Tensor, alpha: Tensor):
        # Update the mean of the distribution
        if self.adaptive_directions:
            self.steps += 1
            new_mean = (self.steps - 1) / self.steps * self.mean + 1 / self.steps * x
            self.covar = (self.steps - 1) / self.steps * self.covar + 1 / self.steps * (
                x[..., :, None] * x[..., None, :]
                - self.steps * new_mean[..., :, None] * new_mean[..., None, :]
                + (self.steps - 1) * self.mean[..., :, None] * self.mean[..., None, :]
            )
            self.mean = new_mean
            self.scale_tril = torch.linalg.cholesky(self.covar)

    @torch.jit.export  # type: ignore
    def init_params(self, x: Tensor):
        if self.adaptive_directions:
            # Initialize the mean of the distribution
            self.mean = x
            self.steps = 1

            # Initialize the covariance of the distribution
            self.covar = torch.eye(x.shape[-1], device=self.device).repeat(x.shape[:-1] + (1, 1))
            self.scale_tril = torch.linalg.cholesky(self.covar)


class LatentSliceKernel(Kernel):
    """Slice sampler where the brackets are latent variables that are also infered on the fly

    https://arxiv.org/pdf/2010.08509.pdf

    """

    def __init__(
        self,
        potential_fn: Callable,
        context: Optional[Tensor] = None,
        step_size: float = 0.1,
        max_resamplings=50,
    ) -> None:
        super().__init__()
        self.step_size = step_size
        self.potential_fn = potential_fn
        self.context = context
        self.s = torch.zeros(1)
        self.max_resamplings = max_resamplings

        self._requires_metropolis_hasting = False
        self._jit_able = True

    @torch.jit.export  # type: ignore
    def forward(self, x: Tensor, x_new: Tensor) -> Tensor:
        return torch.zeros(1)

    @torch.jit.export  # type: ignore
    def _eval_potential(self, x: Tensor) -> Tensor:
        if self.context is None:
            potential = self.potential_fn(x)
        else:
            potential = self.potential_fn(self.context, x)
        return potential

    @torch.jit.export  # type: ignore
    def _eval_potential_masked(self, x: Tensor, mask: Tensor) -> Tensor:
        x_masked = x[mask]
        if self.context is None:
            potential = self.potential_fn(x_masked)
        else:
            context_masked = self.context.repeat(x.shape[0], 1, 1)[mask]
            potential = self.potential_fn(
                context_masked.unsqueeze(-2), x_masked.unsqueeze(-2)
            )  # Unsqueeze for jit support
        return potential

    @torch.jit.export  # type: ignore
    def sample(self, x: Tensor) -> Tensor:
        """
        Sample from the slice sampler where the brackets are latent variables.

        Args:
            x (Tensor): The input tensor

        Returns:
            Tensor: Sampled tensor
        """

        # Calculate the potential of the input tensor
        potential = self._eval_potential(x)

        # Calculate the log probability using log(1 - random number) + potential
        y = torch.log(1 - torch.rand_like(potential, device=self.device)) + potential

        # Calculate the left bound of the bracket
        l = torch.rand_like(x, device=self.device) * self.s + (x - self.s / 2)

        # Calculate the difference between l and x
        diff = torch.abs(l - x) * 2

        # Update the value of s i.e. the latent bracket width
        random_exp = torch.log(1 - torch.rand_like(x, device=self.device)) / (-self.step_size)
        self.s = diff + random_exp

        # Calculate the right and left bounds of the bracket
        a = l - self.s / 2
        b = l + self.s / 2

        # Generate a sample within the bounds
        x_new = torch.rand_like(x, device=self.device) * (b - a) + a

        # Calculate the potential of the newly generated sample
        potential = self._eval_potential(x_new)

        # Check if the new potential is less than the log probability
        mask_reject = potential < y

        # If the potential is less than the log probability, keep re-sampling until it's not

        for i in range(self.max_resamplings):
            if not mask_reject.any():
                break

            # Check if the newly generated sample is smaller than the input
            mask_smaller = x_new < x
            # Update the left and right bounds accordingly
            shape = x.shape[:-1] + (1,)
            a[mask_reject.reshape(shape) & mask_smaller] = x_new[
                mask_reject.reshape(shape) & mask_smaller
            ]
            b[mask_reject.reshape(shape) & ~mask_smaller] = x_new[
                mask_reject.reshape(shape) & ~mask_smaller
            ]

            # Generate a new sample within the updated bounds
            x_new[mask_reject] = (
                torch.rand_like(a[mask_reject], device=self.device) * (b[mask_reject] - a[mask_reject])
                + a[mask_reject]
            )

            # Recalculate the potential of the newly generated sample
            potential = self._eval_potential_masked(x_new, mask_reject)
            # Recheck if the new potential is less than the log probability
            mask_reject[mask_reject.clone()] = potential < y[mask_reject]

        # Return the newly generated sample
        return x_new

    @torch.jit.export  # type: ignore
    def init_params(self, x: Tensor) -> None:
        random_exp = torch.log(1 - torch.rand_like(x, device=self.device)) / (-self.step_size)
        self.s = random_exp


class IndependentKernel(Kernel):
    """
    The IndependentKernel class implements a kernel that performs independent
    sampling from a given distribution.
    """

    proposal: Distribution

    def __init__(self, proposal: Distribution) -> None:
        """Implements a kernel that performs independent
        sampling from a given distribution.

            Args:
                proposal (Distribution): Distribution to sample from.
        """
        super().__init__()
        self.proposal = proposal
        self._symmetric = True

    def forward(self, x: Tensor, x_new: Tensor) -> Tensor:
        """Return the transition probability of going from `x` to `x_new`.

        Args:
            x (Tensor): Current state of the Markov chain
            x_new (Tensor): Proposed new state of the Markov chain

        Returns:
            Tensor: Transition probability of going from `x` to `x_new`.
        """
        return self.proposal.log_prob(x)

    def sample(self, x: Tensor) -> Tensor:
        """Draw a sample from the proposal distribution.

        Args:
            x (Tensor): Current state of the Markov chain

        Returns:
            Tensor: A sample drawn from the proposal distribution.
        """
        return self.proposal.sample(x.shape[:-len(self.proposal.batch_shape) - 1])


class LearnableIndependentKernel(Kernel):
    def __init__(
        self,
        proposal_type: str = "maf",
        proposal_kwargs: dict = {},
        optimizer_class=torch.optim.Adam,
        scheduler_class=torch.optim.lr_scheduler.PolynomialLR,
        lr: float = 1e-2,
        gamma: float = 0.9999,
        link_transform: Optional[torch.distributions.Transform] = None,
    ) -> None:
        super().__init__()
        self.link_transform = (
            torch.distributions.transforms.identity_transform
            if link_transform is None
            else link_transform
        )
        self.builder = get_flow_builder(proposal_type)
        self.proposal_kwargs = proposal_kwargs
        self.optimizer_class = optimizer_class
        self.scheduler_class = scheduler_class
        self.lr = lr
        self.gamma = gamma


    def update_params(self, x: Tensor, x_new: Tensor, alpha: Tensor) -> None:
        for b, (scheduler, optimizer, proposal) in enumerate(
            zip(self.schedulers, self.optimizer, self.proposals)
        ):
            optimizer.zero_grad()
            loss = -torch.mean(proposal.log_prob(x[:, b]))  # type: ignore
            loss.backward()
            optimizer.step()
            scheduler.step()

    def sample(self, x: Tensor) -> Tensor:
        x_new = torch.empty_like(x, device=self.device)
        for b, proposal in enumerate(self.proposals):
            x_new[:, b] = proposal.sample((x.shape[0],))
        return x_new

    def forward(self, x, x_new):
        log_prob = torch.empty(x.shape[:-1], device=self.device)
        for b, proposal in enumerate(self.proposals):
            log_prob[:, b] = proposal.log_prob(x[:, b])
        return log_prob

    def init_params(self, x: Tensor) -> None:
        event_shape = x.shape[-1]
        batch_shape = x.shape[-2]
        self.proposals = [
            self.builder((event_shape,), self.link_transform, device=self.device,**self.proposal_kwargs)
            for b in range(batch_shape)
        ]
        self.optimizer = [
            self.optimizer_class(self.proposals[b].parameters(), lr=self.lr)
            for b in range(batch_shape)
        ]
        self.schedulers = [self.scheduler_class(o, self.gamma) for o in self.optimizer]


class KernelScheduler(Kernel):
    steps: int
    timepoints: list[int]
    current_kernel: Kernel

    def __init__(self, kernels: list[Kernel], timepoints: list[int]) -> None:
        super().__init__()
        self.steps = 0
        self.timepoints = timepoints
        self.kernels = torch.nn.ModuleList(kernels)
        self.current_kernel = kernels[0]

        self._jit_able = False
    def to(self, device):
        for k in self.kernels:
            k.to(device)
        return super().to(device)

    def forward(self, x: Tensor, x_new: Tensor) -> Tensor:
        return self.current_kernel(x, x_new)

    def sample(self, x: Tensor) -> Tensor:
        return self.current_kernel.sample(x)

    def update_params(self, x: Tensor, x_new: Tensor, logalpha: Tensor) -> None:
        for k in self.kernels:
            k.update_params(x, x_new, logalpha)

        self.steps += 1
        if self.steps in self.timepoints:
            self.current_kernel = self.kernels[self.timepoints.index(self.steps)]
            self._symmetric = self.current_kernel._symmetric
            self._requires_metropolis_hasting = (
                self.current_kernel._requires_metropolis_hasting
            )

    def init_params(self, x: Tensor) -> None:
        self.steps = 0
        self.current_kernel = self.kernels[0]
        self._symmetric = self.current_kernel._symmetric
        self._requires_metropolis_hasting = (
            self.current_kernel._requires_metropolis_hasting
        )

        for k in self.kernels:
            k.init_params(x)
