
import torch
from torch.distributions import (
    Bernoulli,
    Categorical,
    Distribution,
    Normal,
    Independent,
    MultivariateNormal,
)
from torch import Tensor
from typing import Callable, Union, Optional, Any

from rbi.models.base import ParametricProbabilisticModel, PyroFlowModel
from rbi.models import BernoulliNet, CategoricalNet, IndependentGaussianNet
from rbi.utils.distributions import MixtureOfDiagNormals
from rbi.models.parametetric_families import MixtureDiagGaussianModel, MultivariateGaussianNet
from rbi.utils.autograd_tools import (
    batch_jacobian,
    batch_hessian,
    batched_jacobian_outer_product,
    batched_jacobian_outer_product,
)

from torch import Tensor 
from torch.nn import Module

_FISHER_REGISTRY = {}       # Registers fisher informations



def register_fisher_information(
    type_p: type,
) -> Callable:
    """Register a fisher information matrix.

    Args:
        type_p (type): Type on which to act

    Returns:
        Callable: Decorating function.
    """


    def decorator(fun):
        _FISHER_REGISTRY[type_p] = fun
        return fun

    return decorator


def fisher_information_matrix(p: Union[Distribution, Callable], parameters: Optional[Tensor] = None, outputs: Optional[Distribution] = None, typ="full", **kwargs) -> Tensor:
    """Returns the quality, metric as well as a short description."""
    
    if type(p) in _FISHER_REGISTRY:
        fn = _FISHER_REGISTRY[type(p)]
    else:
        fn = _FISHER_REGISTRY[Any]
    return fn(p, parameters = parameters, outputs=outputs,typ=typ, **kwargs)


@register_fisher_information(Bernoulli)
def bernoulli_fisher(dist: Bernoulli, jitter:float=1e-20, typ="full", **kwargs) -> Tensor:
    """Returns Bernoulli Fisher information.

    Args:
        dist (Bernoulli): Bernoulli distribution
        jitter (float, optional): For numerical stability. Defaults to 1e-20.

    Returns:
        Tensor: Fisher information.
    """
    p = dist.probs
    p = p.clip(min = jitter)  # type: ignore
    F =  (1 / (p * (1 - p)))
    if typ == "full":
        F = F.unsqueeze(-1)
    
    return F


@register_fisher_information(Categorical)
def categorical_fisher(dist: Categorical, typ: str="full",jitter: float=1e-20, **kwargs) -> Tensor:
    """Returns Categorical Fisher information.

    Args:
        dist (Categorical): Categorical distribution
        typ (str, optional): Return typ of the FIM either 'full', 'diag' or 'trace'.
        jitter (float, optional): For numerical stability. Defaults to 1e-20.

    Returns:
        Tensor: Fisher information.
    """
    p = dist.probs
    p = p.clip(min = jitter)  # type: ignore
    diag = 1 / (p + jitter)
    if typ == "full":
        return torch.diag_embed(diag)
    elif typ == "diag":
        return diag 
    elif typ == "trace":
        return diag.sum(-1)
    else:
        raise NotImplementedError()


@register_fisher_information(Normal)
def normal_fisher(dist: Normal, typ:str = "full", jitter: float=1e-20, **kwargs) -> Tensor:
    """Return Normal Fisher information matrix

    Args:
        dist (Normal): Normal distributions
        typ (str, optional): Return typ of the FIM either 'full', 'diag' or 'trace'.
        jitter (float, optional): For numerical stability. Defaults to 1e-20.

    Raises:
        NotImplementedError: Unknown typ.

    Returns:
        Tensor: Fisher information.
    """
    var = dist.scale**2
    diag_fisher = torch.hstack([1 / var + jitter, 2 / var + jitter])
    if typ == "full":
        return torch.diag_embed(diag_fisher)
    elif typ == "diag":
        return diag_fisher
    elif typ == "trace":
        return diag_fisher.sum(-1)
    else:
        raise NotImplementedError()

@register_fisher_information(Independent)
def independent_fisher(dist: Independent, **kwargs) -> Tensor:
    """Fisher information of factored joint.

    Args:
        dist (Independent): Independent distribution.

    Returns:
        Tensor: Fisher information
    """
    base_dist = dist.base_dist
    return fisher_information_matrix(base_dist, **kwargs)

@register_fisher_information(BernoulliNet)
@register_fisher_information(CategoricalNet)
@register_fisher_information(IndependentGaussianNet)
def compute_reparameterized_fisher(generator: ParametricProbabilisticModel, parameters: Tensor, outputs: Optional[Distribution] = None, typ:str = "full", **kwargs) -> Tensor:
    """Computes the exact fisher information by F_x = J_f^T F_theta J_f

    Args:
        generator (ParametricProbabilisticModel): Function that outputs a probability distribution
        parameters (Tensor): Parameters as input for the function
        outputs (Optional[Distribution], optional): Outputs, if already computed to avoid additional forward pass. Defaults to None.
        typ (str, optional): Typ of matrix. Defaults to "full".

    Raises:
        NotImplementedError: Unknown typ.

    Returns:
        Tensor: Fisher information
    """
    create_graph = kwargs.pop("create_graph", True)
    if outputs is None:
        out = generator(parameters)
    else:
        out = outputs
    

    Js = batch_jacobian(generator.forward_parameters, parameters, create_graph=create_graph)
    F = fisher_information_matrix(out, **kwargs)
    F_x = torch.transpose(Js, dim0=-2, dim1=-1) @ F @ Js

    if typ == "full":
        return F_x
    elif typ == "diag":
        return torch.diagonal(F_x, dim1=-2, dim2=-1)
    elif typ == "trace":
        return torch.diagonal(F_x, dim1=-2, dim2=-1).sum(-1)
    else:
        raise NotImplementedError


@register_fisher_information(MixtureDiagGaussianModel)
@register_fisher_information(MultivariateGaussianNet)
@register_fisher_information(Any)
def monte_carlo_fisher(
    generator: Callable, parameters: Tensor, output: Optional[Distribution] = None, mc_samples: int =20, create_graph: bool=True, typ:str = "full", method: str = "score_based",
**kwargs) -> Tensor:
    """ Computes a MC estimate of the FIM

    Args:
        generator (Callable): Function that returns a distribution.
        parameters (Tensor): Input parmas to the function.
        mc_samples (int, optional): Monte carlo samples. Defaults to 20.
        create_graph (bool, optional): If gradients should be traced. Defaults to True.
        typ (str, optional): Typ of output can be 'full', 'diag' or 'trace'. Defaults to "full".
        method (str, optional): Method to compute either 'score_base' or 'hessian_based'. Defaults to "score_based".

    Raises:
        NotImplementedError: Unknown typ.
        NotImplementedError: Unknown typ.
        NotImplementedError: Unknown method.

    Returns:
        Tensor: _description_
    """
    if method == "score_based":
        if output is None:
            output = generator(parameters)
        if output.has_rsample:
            xs = output.rsample((mc_samples,))  # type: ignore
        else:
            xs = output.sample((mc_samples,))  # type: ignore
            
        score = score_function(xs, parameters, generator, create_graph=create_graph)
        
        if typ == "full":
            return torch.einsum("mbj, mbi -> bji", score, score) / mc_samples
        elif typ == "diag":
            return torch.mean(score**2,0)
        elif typ == "trace":
            return torch.mean(score**2,0).sum(-1)
        else:
            raise NotImplementedError("Unknown typ.")
    elif method == "hessian_based":
        def func(x):
            q = generator(x)
            return q.log_prob(q.sample((mc_samples,))).mean(dim=0).unsqueeze(-1)

        Hs = batch_hessian(func, parameters, create_graph=create_graph)
        if typ == "full":
            return -Hs
        elif typ == "diag":
            # This can may be imporved using hvp
            return torch.diagonal(-Hs, dim1 = -2, dim2 = -1)
        elif typ == "diag":
            # This can may be imporved using hvp
            return torch.diagonal(-Hs, dim1 = -2, dim2 = -1).sum(-1)
        else:
            raise NotImplementedError("Unknown typ.")
    else:
        raise NotImplementedError("Unknown method.")

def score_function(
    x: Tensor,
    parameter: Tensor,
    generator: Callable,
    create_graph: bool =True,
    retain_graph: bool =True,
) -> Tensor:
    """ Returns the score of a probability density i.e. the derivative of the logprob with respect to parameters.

    Args:
        x (Tensor): Evaluation points i.e. samples.
        parameter (Tensor): Parameters
        generator (Callable): Callable that return a distribution.
        create_graph (bool, optional): If it should be differentiable. Defaults to True.
        retain_graph (bool, optional): If tensors should be retained. Defaults to True.

    Returns:
        Tensor: _description_
    """
    assert (
        len(parameter.shape) == 2
    ), "Should be of dimension [param_batch, parameter_dim]"
    batch_size = x.shape[0]

    parameters = (
        parameter.clone().repeat(batch_size, *(1,) * len(x.shape[1:])).requires_grad_()
    )

    p = generator(parameters)
    y = p.log_prob(x)
    scores = torch.autograd.grad(
        y.sum(), parameters, create_graph=create_graph, retain_graph=retain_graph
    )[0]

    return scores

