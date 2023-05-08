from logging import logProcesses
from typing import Callable, Union, Any


from pyro.distributions.conditional import (
    ConstantConditionalTransform,
    ConditionalTransform,
)
from torch.distributions import MixtureSameFamily, Distribution
from torch.distributions.utils import lazy_property
from sbi.utils.user_input_checks_utils import MultipleIndependent


import torch

from typing import List

import math
import numpy as np


def sample_lp_uniformly(
    N: int = 1000,
    d: int = 2,
    p: Union[float, str] = 2.0,
    eps: Union[float, torch.Tensor] = 0.5,
    device: Union[str, Any] = "cpu",
):
    if p == "inf" or p == torch.inf or p == np.inf:
        return torch.rand(N, d, device=device) * eps * 2 - eps
    elif isinstance(p, float) or isinstance(p, int):
        G = GeneralizeGamma(
            1 / p * torch.ones(d, device=device), p * torch.ones(d, device=device)
        )
        xi = G.sample((N,))
        signs = (torch.rand((N, d), device=device) - 0.5).sign()
        x = xi * signs
        z = torch.rand(N, 1, device=device) ** (1 / d)
        x_final = eps * z * (x / torch.linalg.norm(x, keepdim=True, ord=p, dim=-1))
        return x_final

    else:
        raise ValueError("Unknown order...")


def sample_sir(
    shape,
    proposal,
    potential_fn,
    K,
    batch_shape = (1,),
    device="cpu",
):
    proposed_samples = proposal.sample((K,) +shape + batch_shape)
    log_potential = potential_fn(proposed_samples).squeeze()
    log_potential = torch.nan_to_num(
        log_potential, nan=-10000, posinf=10000, neginf=-10000
    )
    logq = proposal.log_prob(proposed_samples).squeeze()
    logq = torch.nan_to_num(logq, nan=-10000, posinf=10000, neginf=-10000)

    log_importance_weights = log_potential - logq
    cum_importance_weights_normalized = torch.softmax(log_importance_weights, 0).cumsum(
        0
    )
    u = torch.rand(cum_importance_weights_normalized.shape[1:], device=device)
    while u.ndim < cum_importance_weights_normalized.ndim:
        u = u.unsqueeze(0)

    mask = torch.cumsum(cum_importance_weights_normalized >= u, 0) == 1.0
    mask = mask.permute(*torch.arange(mask.ndim - 1, -1, -1))
    proposed_samples = proposed_samples.permute(
        *torch.arange(proposed_samples.ndim - 1, -1, -1)
    ).squeeze()
    samples = proposed_samples[:, mask].reshape(
        proposal.event_shape + batch_shape[::-1] + shape
    )
    samples = samples.permute(*torch.arange(samples.ndim - 1, -1, -1))
    samples = samples.permute(*torch.arange(len(shape)-1, -1, -1), *torch.arange(len(shape), samples.ndim))
    return samples


class SIRDistribution(torch.distributions.Distribution):
    arg_constraints = {}

    def __init__(
        self,
        proposal: torch.distributions.Distribution,
        potential_fn: Callable,
        context=None,
        K=4,
        evidence_approx_samples=10,
        validate_args=False,
    ):
        self.proposal = proposal
        self.potential_fn = potential_fn
        self.context = context
        self.K = K
        self.evidence_approx_samples = evidence_approx_samples

        if self.context is not None:
            batch_shape = self.context.shape[:-1]
        else:
            batch_shape = torch.Size((1,))
        super().__init__(batch_shape, proposal.event_shape, validate_args=validate_args)

    def _eval_potential(self, x):
        if self.context is None:
            return self.potential_fn(x)
        else:
            return self.potential_fn(self.context, x)

    @lazy_property
    def _evidence_approx(self):
        proposed_samples = self.proposal.sample(
            (self.evidence_approx_samples, self.K - 1)
        )
        logweights_samples = (
            self._eval_potential(proposed_samples).squeeze()
            - self.proposal.log_prob(proposed_samples).squeeze()
        )

        return torch.logsumexp(logweights_samples - math.log(self.K), 1)

    def sample(self, shape=()):
        shape = torch.Size(shape)
        return sample_sir(shape, self.proposal, self._eval_potential, self.K, batch_shape=self.batch_shape, device=self.context.device)

    def log_prob(self, val):
        log_potential = self._eval_potential(val).squeeze()
        logq = self.proposal.log_prob(val).squeeze()
        log_weights = log_potential - logq

        # Evidence approx
        denominator = torch.zeros(
            (self.evidence_approx_samples, 2) + val.shape[:-1]
        ).squeeze()
        evidence = self.evidence_approx
        while evidence.ndim < denominator.ndim - 1:
            evidence = evidence.unsqueeze(1)

        denominator[:, 0, ...] = evidence
        denominator[:, 1, ...] = log_weights.unsqueeze(0) - math.log(self.K)
        total_logweights = torch.logsumexp(denominator, 1)

        logprob = torch.logsumexp(
            log_potential.unsqueeze(0)
            - total_logweights
            - math.log(self.evidence_approx_samples),
            0,
        )
        return logprob
    
class MCMCDistribution(Distribution):
    arg_constraints = {}
    def __init__(self, mcmc, validate_args=None):
        self.mcmc = mcmc 
        super().__init__(mcmc.batch_shape, mcmc.event_shape, validate_args)

    def sample(self, sample_shape=()):
        shape = torch.Size(sample_shape)
        num_samples = shape.numel()

        samples = self.mcmc.run(num_samples)
        return samples.reshape(*shape,*self.batch_shape,*self.event_shape)
    
    def log_prob(self, value):
        raise NotImplementedError("No log prob")


# class SIRDistirbution(torch.distributions.Distribution):
#     arg_constraints = {}

#     def __init__(
#         self,
#         proposal: torch.distributions.Distribution,
#         potential_fn: Callable,
#         context: torch.Tensor,
#         K: int = 4,
#         evidence_approx_mc_samples=1000,
#         validate_args=None,
#     ):
#         self.proposal = proposal
#         self.potential_fn = potential_fn
#         self.K = K
#         self.context = context
#         self.evidence_approx_mc_samples = evidence_approx_mc_samples
#         # Evidence approx
#         proposed_samples = self.proposal.sample(
#             (self.evidence_approx_mc_samples, self.K - 1)
#         )
#         logweights_samples = (
#             self.potential_fn(self.context, proposed_samples).squeeze()
#             - self.proposal.log_prob(proposed_samples).squeeze()
#         )
#         self.evidence_approx = torch.logsumexp(logweights_samples - math.log(self.K), 1)

#         super().__init__(
#             self.proposal.batch_shape, self.proposal.event_shape, validate_args
#         )

#     def sample(self, shape=()):
#         proposed_samples = self.proposal.sample((self.K,) + shape)
#         log_potential = self.potential_fn(self.context, proposed_samples).squeeze()
#         log_potential = torch.nan_to_num(
#             log_potential, nan=-10000, posinf=10000, neginf=-10000
#         )
#         logq = self.proposal.log_prob(proposed_samples).squeeze()
#         logq = torch.nan_to_num(logq, nan=-10000, posinf=10000, neginf=-10000)

#         log_importance_weights = log_potential - logq
#         cum_importance_weights_normalized = torch.softmax(
#             log_importance_weights, 0
#         ).cumsum(0)
#         u = torch.rand(cum_importance_weights_normalized.shape[1:])
#         while u.ndim < cum_importance_weights_normalized.ndim:
#             u = u.unsqueeze(0)

#         mask = torch.cumsum(cum_importance_weights_normalized >= u, 0) == 1.0
#         samples = (
#             proposed_samples.squeeze()
#             .T[:, mask.T]
#             .reshape(self.event_shape + self.batch_shape + shape)
#             .T
#         )

#         return samples

#     def log_prob(self, val):
#         log_potential = self.potential_fn(self.context, val).squeeze()
#         logq = self.proposal.log_prob(val).squeeze()
#         log_weights = log_potential - logq

#         # Evidence approx
#         denominator = torch.zeros((self.mc_samples, 2) + val.shape[:-1]).squeeze()
#         evidence = self.evidence_approx
#         while evidence.ndim < denominator.ndim - 1:
#             evidence = evidence.unsqueeze(1)

#         denominator[:, 0, ...] = evidence
#         denominator[:, 1, ...] = log_weights.unsqueeze(0) - math.log(self.K)
#         total_logweights = torch.logsumexp(denominator, 1)

#         logprob = torch.logsumexp(
#             log_potential.unsqueeze(0) - total_logweights - math.log(self.mc_samples), 0
#         )
#         return logprob


class MCMCDistirbution(torch.distributions.Distribution):
    arg_constraints = {}

    def __init__(
        self,
        proposal: torch.distributions.Distribution,
        potential_fn: Callable,
        context: torch.Tensor,
        kernel_std=0.1,
        warmup_steps=500,
        num_chains=100,
        thinning=3,
        mc_samples=100,
        validate_args=None,
    ):
        self.proposal = proposal
        self.potential_fn = potential_fn
        self.context = context
        self.mc_samples = mc_samples
        self.warmup_steps = warmup_steps
        self.thinning = thinning
        self.num_chains = num_chains
        self.kernel_std = kernel_std
        # Evidence approx
        proposed_samples = self.proposal.sample((self.mc_samples,))
        logweights_samples = (
            self.potential_fn(self.context, proposed_samples).squeeze()
            - self.proposal.log_prob(proposed_samples).squeeze()
        )
        self.evidence_approx = torch.logsumexp(
            logweights_samples - math.log(self.mc_samples), 0
        )
        super().__init__(
            self.proposal.batch_shape, self.proposal.event_shape, validate_args
        )

    def sample(self, shape=()):
        num_samples = torch.Size(shape).numel()
        if num_samples < self.num_chains:
            num_chains = num_samples
        else:
            num_chains = self.num_chains
        num_iters = self.warmup_steps + (num_samples // num_chains) * self.thinning
        step_size = self.kernel_std

        accepted_sampels = []
        x = self.proposal.sample((num_chains,))
        for i in range(num_iters):
            x_tilde = x + torch.randn_like(x) * step_size
            logp_x = self.potential_fn(self.context, x)
            logp_x_tilde = self.potential_fn(self.context, x_tilde)
            logalpha = torch.min(torch.exp(logp_x_tilde - logp_x), torch.ones(1))
            u = torch.rand_like(logalpha)
            mask_accept = u <= logalpha
            x[mask_accept] = x_tilde[mask_accept]
            acceptance_ratio = mask_accept.float().mean()

            # Adaptive step size
            if acceptance_ratio < 0.3:
                step_size = 0.01 * step_size + step_size * 0.8
            elif acceptance_ratio > 0.4:
                step_size = 0.01 * step_size + step_size * 1.2

            if i > self.warmup_steps:
                if (i % self.thinning) == 0:
                    accepted_sampels.append(x)

        return torch.vstack(accepted_sampels)

    def log_prob(self, val):
        log_potential = self.potential_fn(self.context, val).squeeze()
        return log_potential - self.evidence_approx

class MixtureDistribution(torch.distributions.Distribution):
    arg_constraints = {}
    has_rsample = True
    def __init__(self, mixture_distribution, component_distirbutions) -> None:

        self.component_distirbutions = component_distirbutions
        self.mixture_distribution = mixture_distribution
        super().__init__(batch_shape=self.component_distirbutions[0].batch_shape, event_shape=self.component_distirbutions[0].event_shape)

    def sample(self, shape=()):
        classes = self.mixture_distribution.sample(shape).flatten()
        classes, counts = classes.unique(return_counts=True)
        samples = []
        for c, counts in zip(classes.tolist(), counts.tolist()):
            samples.append(self.component_distirbutions[c].sample((counts,)))
        samples = torch.concat(samples,0)
        samples = samples[torch.randperm(samples.shape[0], device=samples.device)]
        return samples.reshape(shape + self.component_distirbutions[0].batch_shape + self.component_distirbutions[0].event_shape)

    def rsample(self, shape=()):
        classes = self.mixture_distribution.sample(shape).flatten()
        classes, counts = classes.unique(return_counts=True)
        samples = []
        for c, counts in zip(classes.tolist(), counts.tolist()):
            samples.append(self.component_distirbutions[c].rsample((counts,)))
        samples = torch.concat(samples,0)
        samples = samples[torch.randperm(samples.shape[0], device=samples.device)]
        return samples.reshape(shape +  self.component_distirbutions[0].batch_shape + self.component_distirbutions[0].event_shape)

    def log_prob(self, val):
        logits = self.mixture_distribution.logits
        if self.component_distirbutions[0].event_shape == torch.Size([]):
            shape = val.shape
        else:
            shape = val.shape[:-len(self.component_distirbutions[0].event_shape)]
        component_log_probs = torch.empty((len(self.component_distirbutions), ) + shape, device=val.device)
        for i,p in enumerate(self.component_distirbutions):
            component_log_probs[i] = p.log_prob(val) + logits[i]
        return torch.logsumexp(component_log_probs, 0)

    def expand(self, batch_shape):
        return self

    

class DeltaDistribution(torch.distributions.Distribution):
    """Point mass distribution"""

    arg_constraints = {}

    def __init__(self, loc, validate_args=False) -> None:
        self.loc = loc
        super().__init__(batch_shape=self.loc.shape, validate_args=validate_args)

    def sample(self, shape=()):
        shape = torch.Size(shape)
        loc = self.loc.unsqueeze(0)
        ndim = self.loc.ndim
        samples = loc.repeat([shape.numel()] + [1 for n in range(ndim)])
        return samples.reshape(shape + self.batch_shape + self.event_shape)

    def log_prob(self, x):
        x, loc = torch.broadcast_tensors(x, self.loc)
        return torch.isclose(x, loc).float().log()


class EmpiricalDistribution(MixtureSameFamily):
    arg_constraints = {}
    """ Multiple point masses"""

    def __init__(self, samples, event_dim=1, samples_dim=-2, validate_args=False):
        component_distribution = torch.distributions.Independent(
            DeltaDistribution(samples), event_dim
        )
        mixture_distribution = torch.distributions.Categorical(
            logits=torch.zeros(samples.shape[: samples_dim + 1])
        )
        super().__init__(mixture_distribution, component_distribution, validate_args)

    def expand(self, batch_shape, _instance=None):
        return super().expand(batch_shape, _instance)


class MultiIndependent(torch.distributions.Distribution):
    """ " Takes a list of distributions and concatenates them."""

    arg_constraints = {}

    def __init__(self, ps: List[Distribution]) -> None:
        self.ps = ps
        super().__init__()

    def sample(self, shape=()):
        shape = torch.Size(shape)
        samples = []
        for p in self.ps:
            samples.append(p.sample(shape))
        return torch.concat(samples, dim=-1)

    def log_prob(self, x):
        log_probs = 0
        cum_event_shape = 0
        for p in self.ps:
            event_shape = p.event_shape[0]
            log_probs += p.log_prob(
                x[..., cum_event_shape : cum_event_shape + event_shape]
            )
            cum_event_shape += event_shape
        return log_probs


# Some modifications to make it more vectorizable


from pyro.distributions import (
    MixtureOfDiagNormals,  # type: ignore
    ConditionalDistribution,
    TransformedDistribution,  # type: ignore
)
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from pyro.distributions.util import sum_leftmost


class ConstantConditionalDistribution(ConditionalDistribution):
    """Compatibility with arbitrary batch dimensions."""

    def __init__(self, base_dist):
        assert isinstance(base_dist, torch.distributions.Distribution)
        self.base_dist = base_dist

    def condition(self, context):
        if len(context.shape) > 1 and context.shape[0] > 1:
            batch_shape = context.shape[:-1]
            sample_shape = self.base_dist.sample().shape
            return self.base_dist.expand((*batch_shape, *sample_shape))
        else:
            return self.base_dist


class ConditionalTransformedDistribution(ConditionalDistribution):
    """Compatibility with arbitrary batch dimensions."""

    def __init__(self, base_dist, transforms):
        self.base_dist = (
            base_dist
            if isinstance(base_dist, ConditionalDistribution)
            else ConstantConditionalDistribution(base_dist)
        )
        self.transforms = [
            t
            if isinstance(t, ConditionalTransform)
            else ConstantConditionalTransform(t)
            for t in transforms
        ]

    def condition(self, context):
        base_dist = self.base_dist.condition(context)
        transforms = [t.condition(context) for t in self.transforms]
        return TransformedDistribution(base_dist, transforms)

    def clear_cache(self):
        for t in self.transforms:
            if isinstance(t, ConditionalTransform):
                if hasattr(t, "clear_cache"):
                    t.clear_cache()  # type: ignore
            if isinstance(t, ConstantConditionalDistribution):
                if hasattr(t.transform, "clear_cache"):  # type: ignore
                    t.transform.clear_cache()  # type: ignore


class GeneralizeGamma(Distribution):
    def __init__(self, a, c, validate_args=False):
        if isinstance(a, torch.Tensor):
            self.a = a
        else:
            self.a = torch.tensor([a])

        if isinstance(c, torch.Tensor):
            self.c = c
        else:
            self.c = torch.tensor([c])

        super().__init__()
        self.validate_args = validate_args

    def log_prob(self, value):
        loggamma = torch.special.gammaln(self.a)
        logx = value.log(value)

        return self.c - loggamma + (self.c * self.a - 1) * logx - value**self.c

    def sample(self, shape):
        proposal = torch.distributions.Gamma(self.a, 1.0).sample(shape)
        return proposal ** (1 / self.c)


class MixtureOfDiagNormals(MixtureOfDiagNormals):
    def rsample(self, sample_shape=torch.Size()):
        batch_size = self.locs.shape[:-2]
        num_components = self.locs.shape[-2]
        sample_shape = torch.Size(sample_shape)
        num_elements = sample_shape.numel()

        shape_to_call_sample = torch.Size((num_elements,))

        which = self.categorical.sample(shape_to_call_sample)
        which = which.reshape(*shape_to_call_sample, -1)
        which = torch.transpose(which, 0, 1)

        samples = _MixDiagNormalSample.apply(
            self.locs.reshape(-1, num_components, self.dim),
            self.coord_scale.reshape(-1, num_components, self.dim),
            self.component_logits.reshape(-1, num_components),
            self.categorical.probs.reshape(-1, num_components),  # type: ignore
            which,
            shape_to_call_sample + (self.dim,),
        )

        return samples.transpose(0, 1).reshape(*sample_shape, *batch_size, self.dim)


class _MixDiagNormalSample(Function):
    # Modified for batch compatibility, from pyro.
    @staticmethod
    def forward(ctx, locs, scales, component_logits, pis, which, noise_shape):
        dim = scales.size(-1)
        white = locs.new(noise_shape).normal_()
        n_unsqueezes = locs.dim() - which.dim()
        for _ in range(n_unsqueezes):
            which = which.unsqueeze(-1)
        which_expand = which.expand(tuple(which.shape[:-1] + (dim,)))
        loc = torch.gather(locs, -2, which_expand)  # .squeeze(-2)
        sigma = torch.gather(scales, -2, which_expand)  # .squeeze(-2)
        z = loc + sigma * white
        ctx.save_for_backward(z, scales, locs, component_logits, pis)
        return z

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        z, scales, locs, logits, pis = ctx.saved_tensors
        z = z.transpose(0, 1)
        dim = scales.size(-1)
        K = logits.size(-1)
        g = grad_output  # l b i
        g = g.unsqueeze(-2)  # l b 1 i
        g = g.transpose(0, 1)
        batch_dims = locs.dim() - 2

        locs_tilde = locs / scales  # b j i
        sigma_0 = torch.min(scales, -2, keepdim=True)[0]  # b 1 i
        z_shift = (z.unsqueeze(-2) - locs) / sigma_0  # l b j i
        z_tilde = z.unsqueeze(-2) / scales - locs_tilde  # l b j i

        mu_cd = locs.unsqueeze(-2) - locs.unsqueeze(-3)  # b c d i
        mu_cd_norm = torch.pow(mu_cd, 2.0).sum(-1).sqrt()  # b c d
        mu_cd /= mu_cd_norm.unsqueeze(-1)  # b c d i
        diagonals = torch.empty((K,), dtype=torch.long, device=z.device)
        torch.arange(K, out=diagonals)
        mu_cd[..., diagonals, diagonals, :] = 0.0

        mu_ll_cd = (locs.unsqueeze(-2) * mu_cd).sum(-1)  # b c d
        z_ll_cd = (z.unsqueeze(-2).unsqueeze(-2) * mu_cd).sum(-1)  # l b c d
        z_perp_cd = (
            z.unsqueeze(-2).unsqueeze(-2) - z_ll_cd.unsqueeze(-1) * mu_cd
        )  # l b c d i
        z_perp_cd_sqr = torch.pow(z_perp_cd, 2.0).sum(-1)  # l b c d

        shift_indices = torch.empty((dim,), dtype=torch.long, device=z.device)
        torch.arange(dim, out=shift_indices)
        shift_indices = shift_indices - 1
        shift_indices[0] = 0

        z_shift_cumsum = torch.pow(z_shift, 2.0)
        z_shift_cumsum = z_shift_cumsum.sum(-1, keepdim=True) - torch.cumsum(
            z_shift_cumsum, dim=-1
        )  # l b j i
        z_tilde_cumsum = torch.cumsum(torch.pow(z_tilde, 2.0), dim=-1)  # l b j i
        z_tilde_cumsum = torch.index_select(z_tilde_cumsum, -1, shift_indices)
        z_tilde_cumsum[..., 0] = 0.0
        r_sqr_ji = z_shift_cumsum + z_tilde_cumsum  # l b j i

        log_scales = torch.log(scales)  # b j i
        epsilons_sqr = torch.pow(z_tilde, 2.0)  # l b j i
        log_qs = (
            -0.5 * epsilons_sqr - 0.5 * math.log(2.0 * math.pi) - log_scales
        )  # l b j i
        log_q_j = log_qs.sum(-1, keepdim=True)  # l b j 1
        q_j = torch.exp(log_q_j)  # l b j 1
        q_tot = (pis * q_j.squeeze(-1)).sum(-1)  # l b
        q_tot = q_tot.unsqueeze(-1)  # l b 1

        root_two = math.sqrt(2.0)
        shift_log_scales = log_scales[..., shift_indices]
        shift_log_scales[..., 0] = 0.0
        sigma_products = torch.cumsum(shift_log_scales, dim=-1).exp()  # b j i

        reverse_indices = torch.tensor(
            range(dim - 1, -1, -1), dtype=torch.long, device=z.device
        )
        reverse_log_sigma_0 = sigma_0.log()[..., reverse_indices]  # b 1 i
        sigma_0_products = torch.cumsum(reverse_log_sigma_0, dim=-1).exp()[
            ..., reverse_indices - 1
        ]  # b 1 i
        sigma_0_products[..., -1] = 1.0
        sigma_products *= sigma_0_products

        logits_grad = torch.erf(z_tilde / root_two) - torch.erf(
            z_shift / root_two
        )  # l b j i
        logits_grad *= torch.exp(-0.5 * r_sqr_ji)  # l b j i

        logits_grad = (logits_grad * g / sigma_products).sum(-1)  # l b j
        logits_grad = sum_leftmost(logits_grad / q_tot, -1 - batch_dims)  # b j
        logits_grad *= 0.5 * math.pow(2.0 * math.pi, -0.5 * (dim - 1))  # type: ignore
        logits_grad = -pis * logits_grad
        logits_grad = logits_grad - logits_grad.sum(-1, keepdim=True) * pis

        mu_ll_dc = torch.transpose(mu_ll_cd, -1, -2)
        v_cd = torch.erf((z_ll_cd - mu_ll_cd) / root_two) - torch.erf(
            (z_ll_cd + mu_ll_dc) / root_two
        )
        v_cd *= torch.exp(-0.5 * z_perp_cd_sqr)  # l b c d
        mu_cd_g = (g.unsqueeze(-2) * mu_cd).sum(-1)  # l b c d
        v_cd *= (
            -mu_cd_g
            * pis.unsqueeze(-2)
            * 0.5
            * math.pow(2.0 * math.pi, -0.5 * (dim - 1))
        )  # l b c d
        v_cd = pis * sum_leftmost(v_cd.sum(-1) / q_tot, -1 - batch_dims)
        logits_grad += v_cd

        prefactor = pis.unsqueeze(-1) * q_j * g / q_tot.unsqueeze(-1)
        locs_grad = sum_leftmost(prefactor, -2 - batch_dims)
        scales_grad = sum_leftmost(prefactor * z_tilde, -2 - batch_dims)

        locs_grad[~torch.isfinite(locs_grad)] = 0.0
        scales_grad[~torch.isfinite(scales_grad)] = 0.0
        logits_grad[~torch.isfinite(logits_grad)] = 0.0

        return locs_grad, scales_grad, logits_grad, None, None, None
