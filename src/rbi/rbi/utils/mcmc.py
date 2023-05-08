from warnings import warn
import warnings

from tqdm import tqdm
import torch
from torch.distributions import Distribution
from typing import Callable
import inspect
from rbi.utils.autograd_tools import batch_hessian

from rbi.utils.mcmc_kernels import *


class MCMC:
    def __init__(
        self,
        kernel: Kernel,  # Transition kernel
        potential_fn: Callable,  # callable function representing the potential energy of the target distribution
        proposal: Distribution,  # Distribution that is used as the initial proposal for the MCMC chains
        context: Optional[Tensor] = None,
        warmup_steps: int = 100,  # number of warmup steps
        thinning: int = 5,  # thinning factor
        num_chains: int = 100,  # number of MCMC chains
        progress_bar: bool = True,  # If we should show a progress bar.
        jit: bool = False,  # If we should try to optimize runtime using just in time compilation.
        device: str = "cpu", # Device on which we compute
    ) -> None:
        super().__init__()
        self.kernel = kernel
        self.potential_fn = potential_fn
        self.warmup_steps = warmup_steps
        self.thinning = thinning
        self.num_chains = num_chains
        self.proposal = proposal
        self.progress_bar = progress_bar
        self.context = context

        if self.context is None:
            self.batch_shape = torch.Size((1,))
            self.event_shape = self.proposal.event_shape
        else:
            self.context = self.context.reshape(-1, self.context.shape[-1])
            self.batch_shape = self.context.shape[:-1]
            self.event_shape = self.proposal.event_shape

        self.device=device
        self.kernel.to(self.device)

    

        self._check_potential_function()

        if jit:
            self._jit_components()

    def run(self, num_samples: int) -> Tensor:
        """
        Run MCMC to generate a specified number of samples.

        Arguments:
            num_samples: int, number of samples to be generated

        Returns:
            Tensor of shape `(num_samples, *x.shape[1:])` representing the samples
        """

        num_chains = self.num_chains
        if num_samples < self.num_chains:
            num_chains = num_samples
        else:
            num_chains = self.num_chains
        num_iters = self.warmup_steps + (num_samples // num_chains + 1) * self.thinning

        accepted_samples = []

        # If context is given we require a proposal with same batch size!!!
        
        x = self._get_inital_samples(num_chains)
        x.to(self.device)

        if self.progress_bar:
            pbar = tqdm(range(num_iters))
        else:
            pbar = range(num_iters)

        self.kernel.init_params(x)  # type: ignore

        for i in pbar:
            x_new = self.kernel.sample(x)  # type: ignore
            # print(x.shape, x_new.shape)
            if self.kernel._requires_metropolis_hasting:  # type: ignore
                logalpha = self._compute_log_acceptance_probability(x, x_new)
                mask = torch.rand_like(logalpha, device=self.device).log() < logalpha
                x[mask] = x_new[mask]
            else:
                logalpha = torch.zeros(x.shape[:-1])
                x = x_new
            
            alpha = logalpha.exp().clamp(min=0., max=1.).nan_to_num(nan = 0., posinf=1., neginf=0.)
            self.kernel.update_params(x, x_new, alpha)  # type: ignore

            if self.progress_bar:
                pbar.set_description(  # type: ignore
                    f"Accept. prob. {float(alpha.mean().cpu()):0.2f} ({float(alpha.quantile(0.105).cpu()):0.2f} - {float(alpha.quantile(0.95).cpu()):0.2f})"
                )

            if i > self.warmup_steps:
                if (i % self.thinning) == 0:
                    accepted_samples.append(x.clone())

        return torch.vstack(accepted_samples)[:num_samples]
    
    def _get_inital_samples(self, num_chains):
        if self.proposal.batch_shape.numel() == 1:
            x = self.proposal.sample((num_chains, ) + self.batch_shape)  # type: ignore
        elif self.proposal.batch_shape == self.batch_shape:
            x = self.proposal.sample((num_chains, ) )   # type: ignore
        else:
            raise NotImplementedError("Wrong")
        
        return x

    def _compute_log_acceptance_probability(self, x: Tensor, x_new: Tensor) -> Tensor:
        """
        Compute the acceptance probability of the new sample.

        Arguments:
            x: Tensor, current sample
            x_new: Tensor, new sample

        Returns
        """
        if self.context is None:
            logalpha = self.potential_fn(x_new) - self.potential_fn(x)  # type: ignore
        else:
            logalpha = self.potential_fn(self.context, x_new) - self.potential_fn(  # type: ignore
                self.context, x
            )

        logalpha = logalpha.reshape((-1,)+ self.batch_shape)

        if not self.kernel._symmetric:  # type: ignore
            logalpha += self.kernel(x, x_new) - self.kernel(x_new, x)  # type: ignore
        return logalpha.clamp(max=0.0).detach()

    def _check_potential_function(self):
        args = inspect.getfullargspec(self.potential_fn).args
        if len(args) == 2 and self.context is None:
            raise ValueError(
                "You passed a potential function which requires an context, please provide it during initialization."
            )
        elif len(args) == 1 and self.context is not None:
            warn.warn(
                "You passed a context, but your potential function only takes a single argument..."
            )
        elif len(args) > 2:
            raise ValueError(
                "Mhhh, we only support a potential function that either takes only a single argument f(theta) as input or a context f(context, theta)"
            )

        if len(args) == 2:
            self._requires_context = True
        else:
            self._requires_context = False

    def _jit_components(self):
        if self.context is None:
            x = self._get_inital_samples(self.num_chains)
            self.potential_fn = torch.jit.trace(self.potential_fn, x)  # type: ignore
        else:
            x = self._get_inital_samples(self.num_chains)
            self.potential_fn = torch.jit.trace(self.potential_fn, (self.context, x))  # type: ignore

        if hasattr(self.kernel, "potential_fn"):
            self.kernel.potential_fn = self.potential_fn  # type: ignore

        if self.kernel._jit_able:  # type: ignore
            with warnings.catch_warnings():
                self.kernel = torch.jit.script(self.kernel)  # type: ignore
        else:
            pass
