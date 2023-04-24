from ast import Param
import torch
from torch import Tensor
from rbi.models import IndependentGaussianNet, MultivariateGaussianNet
from rbi.utils.autograd_tools import batch_jacobian
from torch.distributions import Independent, Distribution

from typing import Callable, Union, Optional, Any


# TODO REFACTOR

_SAMPLING_TRANSFORM_JACOBIAN = {}       # Registers fisher informations



def register_sampling_transform(
    type_p: type,
) -> Callable:
    """Register a sampling transform jacobian.

    Args:
        type_p (type): Type on which to act

    Returns:
        Callable: Decorating function.
    """


    def decorator(fun):
        _SAMPLING_TRANSFORM_JACOBIAN[type_p] = fun
        return fun

    return decorator


def sampling_transform_jacobian(p: Union[Distribution, Callable], parameters: Optional[Tensor] = None, outputs: Optional[Distribution] = None, typ="full", **kwargs) -> Tensor:
    """Returns the quality, metric as well as a short description."""
    
    if type(p) in _SAMPLING_TRANSFORM_JACOBIAN:
        fn = _SAMPLING_TRANSFORM_JACOBIAN[type(p)]
    else:
        fn = _SAMPLING_TRANSFORM_JACOBIAN[Any]
    return fn(p, parameters = parameters, outputs=outputs,typ=typ, **kwargs)



@register_sampling_transform(MultivariateGaussianNet)
def get_multivaraite_gaussian_transform_jacobian(net, parameters, **kwargs):
    J_mu = batch_jacobian(lambda x: net(x).mean, parameters)
    J_scale = batch_jacobian(lambda x: net(x).stddev, parameters)

    return (
        torch.transpose(J_mu, -2, -1) @ J_mu
        + torch.transpose(J_scale, -2, -1) @ J_scale
    )

@register_sampling_transform(IndependentGaussianNet)
def get_gaussian_transform_jacobian(net, parameters, **kwargs):
    J_mu = batch_jacobian(lambda x: net(x).base_dist.loc, parameters)
    J_scale = batch_jacobian(lambda x: net(x).base_dist.scale, parameters)
    return (
        torch.transpose(J_mu, -2, -1) @ J_mu
        + torch.transpose(J_scale, -2, -1) @ J_scale
    )

@register_sampling_transform(Any)
def get_monte_carlo_transform_jacobain_estimate(
    net, parameters, mc_samples=10, create_graph=True, **kwargs
):
    batch_shape = parameters.shape[0]
    input_shape = parameters.shape[-1]

    if len(parameters.shape) == 2:
        parameters = parameters.unsqueeze(1)

    if len(parameters.shape) != 3:
        raise ValueError("The input shape must be (batch_shape, input_shape)")

    x_rep = parameters.repeat(1, mc_samples, 1)
    try:
        J = batch_jacobian(
            lambda x: net(x).rsample(),
            x_rep.reshape(-1, input_shape),
            create_graph=create_graph,
        ).reshape(batch_shape, mc_samples, net.output_dim, net.input_dim)
    except NotImplementedError:
        raise ValueError(
            "The resulting probability distribution must have rsample implemented!"
        )

    return torch.mean(torch.transpose(J, dim0=-2, dim1=-1) @ J, 1)
