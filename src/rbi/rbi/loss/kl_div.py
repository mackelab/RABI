import torch
from torch import Tensor
from torch.distributions import (
    Distribution,
    kl_divergence,
    register_kl,
    TransformedDistribution,
)

from rbi.utils.distributions import EmpiricalDistribution

# Add mixtures

MC_SAMPLES = 10


@register_kl(Distribution, Distribution)
def general_kl_divergence(p: Distribution, q: Distribution) -> Tensor:  # type: ignore
    global MC_SAMPLES

    p_batch = p.batch_shape
    q_batch = q.batch_shape

    batch_shape = torch.broadcast_shapes(p_batch, q_batch)
    p = p.expand(batch_shape)
    q = q.expand(batch_shape)

    if p.has_rsample:
        samples = p.rsample((MC_SAMPLES,))  # type: ignore
    else:
        samples = p.sample((MC_SAMPLES,))  # type: ignore
    logq = q.log_prob(samples)
    logp = p.log_prob(samples)

    return torch.mean(logp - logq, 0)


@register_kl(TransformedDistribution, TransformedDistribution)
def general_kl_divergence(p: TransformedDistribution, q: TransformedDistribution):
    # Flows can be VRAM hungry...
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    global MC_SAMPLES

    p_batch = p.batch_shape
    q_batch = q.batch_shape

    batch_shape = torch.broadcast_shapes(p_batch, q_batch)
    p = p.expand(batch_shape)
    q = q.expand(batch_shape)
    
    samples = p.rsample((MC_SAMPLES,))  # type: ignore
    logq = q.log_prob(samples)
    logp = p.log_prob(samples)

    return torch.mean(logp - logq, 0)


@register_kl(EmpiricalDistribution, Distribution)
def kl_empirical(p: EmpiricalDistribution, q: Distribution):
    samples = p.sample((MC_SAMPLES,))
    logq = q.log_prob(samples)
    logp = p.log_prob(samples)

    return torch.mean(logp - logq, 0)


def set_mc_budget(n: int):
    global MC_SAMPLES
    MC_SAMPLES = n
