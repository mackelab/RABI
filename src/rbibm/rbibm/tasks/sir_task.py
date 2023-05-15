from rbibm.tasks.base import InferenceTask
from rbi.utils.distributions import SIRDistribution, MCMCDistribution
from rbi.utils.mcmc import MCMC 
from rbi.utils.mcmc_kernels import AdaptiveMultivariateGaussianKernel, GaussianKernel,LearnableIndependentKernel,KernelScheduler
import torch
from torch import Tensor, nn
from pyro.distributions import ConditionalDistribution

from torchdiffeq import odeint, odeint_adjoint

from typing import Tuple, Callable
import inspect


class SirODE(nn.Module):
    def __init__(
        self,
        N: float = 5,
        I0: float = 0.25,
    ) -> None:
        super().__init__()
        self.batch_size = 1
        self.register_buffer("N", torch.as_tensor(N))
        self.register_buffer("beta", torch.ones(1))
        self.register_buffer("gamma", torch.ones(1))
        self.register_buffer("S0", torch.as_tensor(N))
        self.register_buffer("I0", torch.as_tensor(I0))
        self.register_buffer("R0", torch.zeros(1))

    @torch.jit.export  # type: ignore
    def set_theta(self, theta: Tensor) -> None:
        # This preserves stability of the ode.
        theta = theta.reshape(-1, 2)
        params = theta.split(1, dim=-1)
        self.batch_size = theta.shape[0]
        self.beta = params[0]
        self.gamma = params[1]

    @torch.jit.export  # type: ignore
    def get_initial_state(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Returns the initial state"""
        return (
            self.S0.clone().repeat(self.batch_size, 1),  # type: ignore
            self.I0.clone().repeat(self.batch_size, 1),  # type: ignore
            self.R0.clone().repeat(self.batch_size, 1),  # type: ignore
        )

    def forward(
        self, t: float, state: Tuple[Tensor, Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        S, I, R = state
        dS = -(self.beta * S * I) / self.N
        dI = (self.beta * S * I) / self.N - self.gamma * I
        dR = self.gamma * I
        return dS, dI, dR


class SIRPosterior(ConditionalDistribution):
    def __init__(self, prior, potential_fn) -> None:
        super().__init__()
        self.prior = prior
        self.potential_fn = potential_fn

        k1 = AdaptiveMultivariateGaussianKernel()
        k2 = LearnableIndependentKernel()
        k3 = GaussianKernel()

        self.k = KernelScheduler([k1,k2, k3,k2], [0, 50, 100, 150])


    def condition(self, context:Tensor):

        proposal = SIRDistribution(self.prior, self.potential_fn, context=context, K= 10)
        mcmc = MCMC(self.k , self.potential_fn, proposal, context=context ,thinning=10, warmup_steps=150, num_chains=1000, device=context.device)

        return MCMCDistribution(mcmc)
        

class SIRTask(InferenceTask):
    def __init__(
        self,
        t_max: float = 50.0,
        time_points_observed: int = 50,
        observation_noise: float = 0.2,
        atol=1e-6,
        rtol=1e-5,
    ) -> None:
        """Lotka volterra inference task

        Args:
            t_max (int, optional): Time to simulate. Defaults to 20.
            time_points_observed (int, optional): Timepoints observed for inference. Defaults to 50.
            observation_noise (float, optional): Observation noise. Defaults to 0.05.
            initial_noise (float, optional): Initial condition noise. Defaults to 0.0.
            prior_scale (float, optional): Prior scale. Defaults to 0.2.
            prior_mean (float, optional): Prior mean. Defaults to 0.0.
            odeint_kwargs (dict, optional): Kwargs for odesolver. Defaults to {}.
        """
        prior = torch.distributions.Independent(
            torch.distributions.Normal(torch.zeros(2), 2*torch.ones(2)),
            1,
        )

        self.ode = torch.jit.script(SirODE())  # type: ignore
        self.t_max = t_max
        self.time_points_observed = time_points_observed
        self.observation_noise = observation_noise
        self.atol = atol 
        self.rtol = rtol

        def ode_sol(theta):
            theta = theta.sigmoid()
            theta = theta.reshape(-1, 2)
            t = torch.linspace(
                0, self.t_max, self.time_points_observed, device=theta.device
            )
            self.ode.to(theta.device)
            self.ode.set_theta(theta)  # type: ignore
            try:
                sol = odeint(
                    self.ode, self.ode.get_initial_state(), t, atol=self.atol, rtol=self.rtol  # type: ignore
                )
            except:
                sol = odeint(
                    self.ode, self.ode.get_initial_state(), t, method="euler", options={"step_size":1e-3}, # type: ignore
                )
            return sol

        self.ode_sol = ode_sol

        def likelihood_fn(theta):
            batch_shape = theta.shape[:-1]
            obs_noise = self.observation_noise * torch.ones(1, device=theta.device)
            sol = self.ode_sol(theta)
            x = sol[1].squeeze(-1).transpose(0, 1)  # type: ignore
            mask = torch.isfinite(x)
            x[~mask] = 0.0
            x = x.reshape(*batch_shape, -1) + 1e-4
            return torch.distributions.Independent(
                torch.distributions.LogNormal(
                    x.log(),  obs_noise
                ),
                1,
            )

        super().__init__(prior, likelihood_fn, None)

        self.input_dim = time_points_observed
        self.output_dim = 2

    def get_loglikelihood_fn(self, device: str = "cpu") -> Callable:
        self.ode.to(device)  # type: ignore
        return super().get_loglikelihood_fn(device)

    def get_true_posterior(self, device: str = "cpu"):
        return SIRPosterior(self.get_prior(device), self.get_potential_fn(device))


