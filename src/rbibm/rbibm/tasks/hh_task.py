from typing import Callable, List, Optional, Tuple
from rbibm.tasks.base import InferenceTask, Simulator


from pyro.distributions import ConditionalDistribution
from sbi.utils.user_input_checks_utils import MultipleIndependent
from rbi.utils.distributions import SIRDistribution

import os

import torch
from torch import nn, Tensor

from torchdiffeq import odeint, odeint_adjoint


class HudgkinHuxleyODE(nn.Module):
    def __init__(
        self,
        g_leak: float = 0.1,
        V0: float = -70.0,
        Vt: float = -60.0,
        tau_max: float = 6e2,
        I_on: float = 10.0,
        I_off: float = 50.0,
        curr_level=5e-4,
    ) -> None:
        super().__init__()
        self.batch_size = 1
        self.register_buffer("gbar_Na", torch.ones(1))
        self.register_buffer("gbar_K", torch.ones(1))
        self.register_buffer("gbar_M", torch.ones(1))
        self.register_buffer("C", torch.ones(1))
        self.register_buffer("E_leak", torch.ones(1))
        self.register_buffer("E_Na", torch.ones(1))
        self.register_buffer(
            "E_K",
            torch.ones(
                1,
            ),
        )
        self.register_buffer("g_leak", torch.as_tensor(g_leak))
        self.register_buffer("Vt", torch.as_tensor(Vt))
        self.register_buffer("V0", torch.as_tensor(V0))
        self.register_buffer("tau_max", torch.as_tensor(tau_max))
        self.register_buffer("I_on", torch.as_tensor(I_on))
        self.register_buffer("I_off", torch.as_tensor(I_off))
        self.register_buffer("curr_level", torch.as_tensor(curr_level))
        self.register_buffer(
            "A_soma", torch.as_tensor(3.141592653589793 * ((70.0 * 1e-4) ** 2))
        )

    @torch.jit.export  # type: ignore
    def set_theta(self, theta: Tensor) -> None:
        theta = theta.reshape(-1, 7)
        params = theta.split(1, dim=-1)
        self.batch_size = theta.shape[0]
        self.gbar_Na = params[0]
        self.gbar_K = params[1]
        self.gbar_M = params[2]
        self.C = params[3]
        self.E_leak = params[4]
        self.E_Na = params[5]
        self.E_K = params[6]

    @torch.jit.export  # type: ignore
    def get_initial_state(self):
        V0 = self.V0.repeat(self.batch_size, 1)  # type: ignore
        return V0, self.n_inf(V0), self.m_inf(V0), self.h_inf(V0), self.p_inf(V0)

    @torch.jit.export  # type: ignore
    def efun(self, z: Tensor) -> Tensor:
        mask = torch.abs(z) < 1e-4
        new_z = torch.zeros_like(z, device=z.device)
        new_z[mask] = 1 - z[mask] / 2
        new_z[~mask] = z[~mask] / (torch.exp(z[~mask]) - 1)
        return new_z

    @torch.jit.export  # type: ignore
    def I_in(self, t: float) -> Tensor:
        if t > self.I_on and t < self.I_off:  # type: ignore
            return self.curr_level / self.A_soma  # type: ignore
        else:
            return torch.zeros(1, device=self.A_soma.device)  # type: ignore

    @torch.jit.export  # type: ignore
    def alpha_m(self, x: Tensor) -> Tensor:
        v1 = x - self.Vt - 13.0
        return 0.32 * self.efun(-0.25 * v1) / 0.25

    @torch.jit.export  # type: ignore
    def beta_m(self, x: Tensor) -> Tensor:
        v1 = x - self.Vt - 40
        return 0.28 * self.efun(0.2 * v1) / 0.2

    @torch.jit.export  # type: ignore
    def alpha_h(self, x: Tensor) -> Tensor:
        v1 = x - self.Vt - 17.0
        return 0.128 * torch.exp(-v1 / 18.0)

    @torch.jit.export  # type: ignore
    def beta_h(self, x: Tensor) -> Tensor:
        v1 = x - self.Vt - 40.0
        return 4.0 / (1 + torch.exp(-0.2 * v1))

    @torch.jit.export  # type: ignore
    def alpha_n(self, x: Tensor) -> Tensor:
        v1 = x - self.Vt - 15.0
        return 0.032 * self.efun(-0.2 * v1) / 0.2

    @torch.jit.export  # type: ignore
    def beta_n(self, x: Tensor) -> Tensor:
        v1 = x - self.Vt - 10.0
        return 0.5 * torch.exp(-v1 / 40)

    @torch.jit.export  # type: ignore
    def tau_n(self, x: Tensor) -> Tensor:
        return 1 / (self.alpha_n(x) + self.beta_n(x))

    @torch.jit.export  # type: ignore
    def n_inf(self, x: Tensor) -> Tensor:
        return self.alpha_n(x) / (self.alpha_n(x) + self.beta_n(x))

    @torch.jit.export  # type: ignore
    def tau_m(self, x: Tensor) -> Tensor:
        return 1 / (self.alpha_m(x) + self.beta_m(x))

    @torch.jit.export  # type: ignore
    def m_inf(self, x: Tensor) -> Tensor:
        return self.alpha_m(x) / (self.alpha_m(x) + self.beta_m(x))

    @torch.jit.export  # type: ignore
    def tau_h(self, x: Tensor) -> Tensor:
        return 1 / (self.alpha_h(x) + self.beta_h(x))

    @torch.jit.export  # type: ignore
    def h_inf(self, x: Tensor) -> Tensor:
        return self.alpha_h(x) / (self.alpha_h(x) + self.beta_h(x))

    @torch.jit.export  # type: ignore
    def p_inf(self, x: Tensor) -> Tensor:
        v1 = x + 35.0
        return 1.0 / (1.0 + torch.exp(-0.1 * v1))

    @torch.jit.export  # type: ignore
    def tau_p(self, x: Tensor) -> Tensor:
        v1 = x + 35.0
        return self.tau_max / (3.3 * torch.exp(0.05 * v1) + torch.exp(-0.05 * v1))

    def forward(
        self, t: float, state: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        V, n, m, h, p = state

        dV = (
            (m**3) * self.gbar_Na * h * (self.E_Na - V)
            + (n**4) * self.gbar_K * (self.E_K - V)
            + self.gbar_M * p * (self.E_K - V)
            + self.g_leak * (self.E_leak - V)
            + self.I_in(t)
        )
        dV = self.C * dV
        dn = (self.n_inf(V) - n) / self.tau_n(V)
        dm = (self.m_inf(V) - m) / self.tau_m(V)
        dh = (self.h_inf(V) - h) / self.tau_h(V)
        dp = (self.p_inf(V) - p) / self.tau_p(V)

        return dV, dn, dm, dh, dp


class MultiDistribution(torch.distributions.Distribution):
    def __init__(self, qs) -> None:
        self.qs = qs
        super().__init__(batch_shape=qs[0].batch_shape, event_shape=qs[0].event_shape)

    def log_prob(self, val):
        log_probs = []
        for q in self.qs:
            log_probs.append(q.log_prob(val))
        return torch.concat(log_probs, dim=-len(self.event_shape) - 1)

    def sample(self, shape=()):
        samples = []
        for q in self.qs:
            samples.append(q.sample(shape).unsqueeze(-len(self.event_shape) - 1))

        return torch.concat(samples, dim=-len(self.event_shape) - 1)


class HHPosterior(ConditionalDistribution):
    def __init__(self, potential_fn, device, *args) -> None:
        super().__init__()
        self.potential_fn = potential_fn
        self.device = device
        self._args = tuple(args)
        path = (
            os.path.dirname(os.path.abspath(__file__))
            + "/reference_models/hh_reference_models.pkl"
        )
        model_bank = torch.load(path)
        print(self._args)
        if self._args in model_bank:
            self.model = model_bank[self._args].to(device)
        else:
            raise NotImplementedError("No posterior for this...")

    def condition(self, context):
        context = context.reshape(-1, context.shape[-1])
        with torch.no_grad():
            qs = []
            for x_o in context:
                q = self.model(x_o.squeeze())
                q_sir = SIRDistribution(
                    q, self.potential_fn, context=x_o.squeeze(), K=8
                )
                qs.append(q_sir)

        if len(qs) > 1:
            return MultiDistribution(qs)
        else:
            return qs[0]


class HHTask(InferenceTask):
    """Hudgkin huxley inference task."""

    def __init__(
        self,
        observation_noise: float = 0.5,
        prior_min: List[float] = [1.0, 1e-4, 1e-4, 0.9, -80, 10, -120],
        prior_max: List[float] = [100.0, 5.0, 0.5, 1.2, -50, 80, -90],
        time_points_observed: int = 200,
        I_on: float = 10.0,
        g_leak: float = 0.1,
        Vt: float = -60.0,
        I_off: float = 50.0,
        t_max: float = 60.0,
    ) -> None:
        """Hudgkin huxley inference task.

        Args:
            observation_noise (float, optional): Observation noise. Defaults to 0.5.
            prior_min (List[float], optional): Prior minimum bounds. Defaults to [1.0, 1e-4, 1e-4, 0.9, -80, 10, -120].
            prior_max (List[float], optional): Prior maximum bounds.. Defaults to [100.0, 5.0, 0.5, 1.2, -50, 80, -90].
            time_points_observed (int, optional): Time points observed. Defaults to 200.
            internal_noise (float, optional): Initial condition noise. Defaults to 0.0.
            I_on (float, optional): Input voltage start. Defaults to 10.0.
            g_leak (float, optional): Fixed parameter g_leak. Defaults to 0.1.
            Vt (float, optional): Fixed parameter Vt. Defaults to -60.0.
            I_off (float, optional): Input voltage stop. Defaults to 50.0.
            t_max (float, optional): Simulated form [0,t_max]. Defaults to 60.0.
        """

        self.prior_min = prior_min
        self.prior_max = prior_max
        prior = torch.distributions.Independent(
            torch.distributions.Uniform(
                low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max)
            ),
            1,
        )
        self.observation_noise = observation_noise
        self.ode = torch.jit.script(  # type: ignore
            HudgkinHuxleyODE(g_leak=g_leak, I_on=I_on, I_off=I_off, Vt=Vt)
        )
        self.t_max = t_max
        self.I_on = I_on
        self.I_off = I_off
        self.g_leak = g_leak
        self.Vt = Vt
        self.time_points_observed = time_points_observed
        self.input_dim = time_points_observed
        self.output_dim = 7

        def solve_ode(theta):
            batch_shape = theta.shape[:-1]
            theta = theta.reshape(-1, theta.shape[-1])
            self.ode.set_theta(theta)  # type: ignore
            x0 = self.ode.get_initial_state()  # type: ignore
            try:
                V, _, _, _, _ = odeint(  # type: ignore
                    self.ode,
                    x0,
                    torch.linspace(
                        0.0, self.t_max, self.time_points_observed, device=theta.device
                    ),
                    method="bosh3",
                    atol=1e-4,
                    rtol=1e-3,
                )
            except:
                V, _, _, _, _ = odeint(  # type: ignore
                    self.ode,
                    x0,
                    torch.linspace(
                        0.0, self.t_max, self.time_points_observed, device=theta.device
                    ),
                    method="euler",
                    options={"step_size": 1e-3},
                )
            V = V.squeeze(-1).transpose(0, 1)
            mask = torch.isfinite(V)
            V[~mask] = -70.0
            return V.reshape(*batch_shape, -1)

        def simulator(theta):
            V = solve_ode(theta)
            V += torch.randn_like(V) * self.observation_noise
            return V

        def likelihood_fn(theta):
            y = solve_ode(theta)
            return torch.distributions.Independent(
                torch.distributions.Normal(y, self.observation_noise), 1
            )

        super().__init__(prior, likelihood_fn, simulator)

    def get_simulator(
        self, batch_size: Optional[int] = 100000, device: str = "cpu"
    ) -> Simulator:
        self.ode.to(device)  # type: ignore
        return super().get_simulator(batch_size, device)

    def get_loglikelihood_fn(self, device: str = "cpu") -> Callable:
        self.ode.to(device)  # type: ignore
        return super().get_loglikelihood_fn(device)

    def get_true_posterior(self, device: str = "cpu"):
        return HHPosterior(
            self.get_potential_fn(device=device),
            device,
            self.observation_noise,
            tuple(self.prior_min),
            tuple(self.prior_max),
            self.time_points_observed,
            self.I_on,
            self.g_leak,
            self.Vt,
            self.I_off,
            self.t_max,
        )


class SimpleHHTask(InferenceTask):
    """Hudgkin huxley inference task."""

    def __init__(
        self,
        observation_noise: float = 0.5,
        prior_min: List[float] = [0.0, 0.0],
        prior_max: List[float] = [100.0, 100.0],
        time_points_observed: int = 500,
        I_on: float = 10.0,
        g_leak: float = 0.1,
        Vt: float = -60.0,
        I_off: float = 50.0,
        t_max: float = 60.0,
        gbar_M=0.3,
        tau_max=6e2,
        E_leak=-70.0,
        C=1.0,
        E_Na=53,
        E_K=-107,
    ) -> None:
        """Hudgkin huxley inference task.

        Args:
            observation_noise (float, optional): Observation noise. Defaults to 0.5.
            prior_min (List[float], optional): Prior minimum bounds. Defaults to [1.0, 1e-4, 1e-4, 0.9, -80, 10, -120].
            prior_max (List[float], optional): Prior maximum bounds.. Defaults to [100.0, 5.0, 0.5, 1.2, -50, 80, -90].
            time_points_observed (int, optional): Time points observed. Defaults to 200.
            internal_noise (float, optional): Initial condition noise. Defaults to 0.0.
            I_on (float, optional): Input voltage start. Defaults to 10.0.
            g_leak (float, optional): Fixed parameter g_leak. Defaults to 0.1.
            Vt (float, optional): Fixed parameter Vt. Defaults to -60.0.
            I_off (float, optional): Input voltage stop. Defaults to 50.0.
            t_max (float, optional): Simulated form [0,t_max]. Defaults to 60.0.
        """

        self.prior_min = prior_min
        self.prior_max = prior_max
        prior = torch.distributions.Independent(
            torch.distributions.Uniform(
                low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max)
            ),
            1,
        )
        self.observation_noise = observation_noise
        self.ode = torch.jit.script(  # type: ignore
            HudgkinHuxleyODE(g_leak=g_leak, I_on=I_on, I_off=I_off, Vt=Vt)
        )
        self.t_max = t_max
        self.time_points_observed = time_points_observed
        self.input_dim = time_points_observed
        self.output_dim = 2

        theta_almost = torch.zeros(5)
        theta_almost[0] = gbar_M
        theta_almost[1] = C
        theta_almost[2] = E_leak
        theta_almost[3] = E_Na
        theta_almost[4] = E_K

        def simulator(theta):
            batch_shape = theta.shape[:-1]
            theta_rest = theta_almost.to(theta.device).repeat(*batch_shape, 1)
            theta = torch.hstack([theta, theta_rest])
            self.ode.set_theta(theta)  # type: ignore
            x0 = self.ode.get_initial_state()  # type: ignore
            V, _, _, _, _ = odeint(  # type: ignore
                self.ode,
                x0,
                torch.linspace(
                    0.0, self.t_max, self.time_points_observed, device=theta.device
                ),
                method="bosh3",
                atol=1e-4,
                rtol=1e-3,
            )
            V = V.squeeze(-1).transpose(0, 1)
            mask = torch.isfinite(V)
            V[~mask] = -70.0
            V = V + torch.randn_like(V, device=theta.device) * self.observation_noise
            return V.reshape(*batch_shape, -1)

        def likelihood_fn(theta):
            y = simulator(theta)
            return torch.distributions.Independent(
                torch.distributions.Normal(y, self.observation_noise), 1
            )

        super().__init__(prior, likelihood_fn, simulator)

    def get_simulator(
        self, batch_size: Optional[int] = None, device: str = "cpu"
    ) -> Simulator:
        self.ode.to(device)  # type: ignore
        return super().get_simulator(batch_size, device)

    def get_loglikelihood_fn(self, device: str = "cpu") -> Callable:
        self.ode.to(device)  # type: ignore
        return super().get_loglikelihood_fn(device)
