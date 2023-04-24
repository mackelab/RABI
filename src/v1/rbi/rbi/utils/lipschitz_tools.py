from typing import List, Union
from torch.nn import Module
from torch import Tensor
import torch
from torch import nn

from torch.nn.utils.parametrize import remove_parametrizations
from rbi.models.module_parameterizations import linf_lipschitz_bound, l2_lipschitz_bound, l1_lipschitz_bound, frob_norm_bound, nuc_norm_bound, scaling, learnable_scaling, ScaleWeights, LearnableScaleWeights
from rbi.models.activations import GroupSort, PartialNonlinearity
from rbi.utils.autograd_tools import batch_jacobian
import numpy as np


# Some lipschitz constants
L_dict = {nn.Softmax: 0.5, nn.Sigmoid: 0.25, nn.Tanh: 1., nn.ReLU: 1., nn.LeakyReLU: 1., nn.ELU: 1., nn.Softplus: 1., nn.GELU: 1.13, nn.Identity: 1, GroupSort: 1., PartialNonlinearity: 1.}

def lipschitz_neural_net(net: Module, L:float, ord1:float=2.0, ord2:float=2.0, name:str="weight", check_if_lipschitz:bool=False, learnable_L: bool = False,**kwargs):
    """This function takes a normal network and reparameterizes it such that it has a Lipschitz constant bounded by L.

    Args:
        net (Module): Neural net.
        L (float): Lipschitz bound.
        ord1 (float, optional): Input order of metric. Defaults to 2.0.
        ord2 (float, optional): Output order of metric. Defaults to 2.0.
        name (str, optional): Name of parameters. Defaults to "weight".
        check_if_lipschitz (bool, optional): Check if lipschitz bound holds. Defaults to False.
        learnable_L (bool, optional): If L should be learnable. Defaults to False.

    Raises:
        NotImplementedError: Order
        ValueError: Order
    """
    remove_all_parameterizations(net, name)
    modules = list(net.modules())
    learnable_linear_layer = [l for l in modules if hasattr(l, name)]
    nonlinearities = [l for l in modules if not l.__class__ in list(L_dict.values())]
    L_nonlin = np.prod([L_dict[l.__class__] for l in nonlinearities if l.__class__ in L_dict])  # type: ignore
    num_linear_layers = len(learnable_linear_layer)
    if num_linear_layers == 0.:
        num_linear_layers = 1.
    layer_wise_L = float(((1/L_nonlin) * L) ** (1 / num_linear_layers))

    for l in modules:
        if ord1 == 2.0 and ord2 == 2.0:
            if l in learnable_linear_layer:
                normalization_method = kwargs.pop("method", "spectral")
                l2_lipschitz_bound(
                    l,
                    L=layer_wise_L,
                    method=normalization_method,
                    learnable_L=learnable_L,
                    **kwargs
                )
        elif ord1 == 1.0 and ord2 == 1.0:
            if l in learnable_linear_layer:
                l1_lipschitz_bound(l, L=layer_wise_L,learnable_L=learnable_L, **kwargs)
        elif ord1 == torch.inf and ord2 == torch.inf:
            if l in learnable_linear_layer:
                linf_lipschitz_bound(l, L=layer_wise_L, learnable_L=learnable_L, **kwargs)
        else:
            raise NotImplementedError("The orders are not implemented :(")

    if check_if_lipschitz:
        cond = check_lipschitz_continuouity(net, L, ord1 = ord1, ord2=ord2)
        if not cond:
            raise ValueError("Check failed, you may disable it.")

def check_lipschitz_continuouity(net: Module, L:float, ord1:float=2.0, ord2: float=2.0, mc_samples=1000) -> bool:
    """Check if lipschitz continuouty holds.

    Args:
        net (Module): Neural net
        L (float): lipschitz bound
        ord1 (float, optional): Input order of metric. Defaults to 2.0.
        ord2 (float, optional): Output order of metric. Defaults to 2.0.

    Raises:
        NotImplementedError: Order

    Returns:
        bool: If satisfied
    """
    modules = list(net.modules())
    linear_layer = [l for l in modules if isinstance(l, nn.Linear)]
    input_dim = linear_layer[0].in_features
    device = linear_layer[0].weight.device
    x = torch.randn((mc_samples, input_dim), requires_grad=True, device=device)*20
    if ord1 == ord2:
        max_jac_norm = torch.linalg.matrix_norm(batch_jacobian(net, x), ord=ord2).max()

        return max_jac_norm < L
    else:
        raise NotImplementedError()

def weight_norm_constrained_neural_net(net: Module, values: List[float], ord: Union[float, str], name: str="weight", learnable_scale=False, **kwargs):
    """Constraints matrix norms of weights by a certain specified value.

    Args:
        net (Module): Neural net.
        values (List[float]): Matrix norm for each linear layer.
        ord (Union[float, str]): Order of matrix norm, can be 1,2,inf,fro or nuc.
        name (str, optional): Weight. Defaults to "weight".
        learnable_scale (bool, optional): If bound values should be learnable. Defaults to False.

    Raises:
        ValueError: _description_
        NotImplementedError: _description_
    """
    remove_all_parameterizations(net, name)
    modules = list(net.modules())
    learnable_linear_layer = [l for l in modules if hasattr(l, name)]
    if isinstance(values, float):
        values = torch.tensor([values] * len(learnable_linear_layer))  # type: ignore
    elif len(values) == 1:
        values = torch.tensor([values[0]] * len(learnable_linear_layer))  # type: ignore
    else:
        if len(learnable_linear_layer) != len(values):
            raise ValueError(
                "The number of values must be one or equaling the length of linear layers."
            )

    i = 0
    for l in modules:
        if l in learnable_linear_layer:
            if ord == 2.0:
                normalization_method = kwargs.pop("method", "spectral")
                l2_lipschitz_bound(
                    l, L=values[i], method=normalization_method, learnable_L=learnable_scale, **kwargs
                )
            elif ord == 1.0:
                l1_lipschitz_bound(l, L=values[i], learnable_L=learnable_scale, **kwargs)
            elif ord == torch.inf:
                linf_lipschitz_bound(l, L=values[i], learnable_L=learnable_scale,**kwargs)
            elif ord == "fro":
                frob_norm_bound(l, name, **kwargs)
                learn_scale = kwargs.get("learnable_L", False)
                if not learn_scale:
                    scaling(l, name, values[i])
                else:
                    learnable_scaling(l, name, values[i])
            elif ord == "nuc":
                nuc_norm_bound(l, name, **kwargs)
                learn_scale = kwargs.get("learnable_L", False)
                if not learn_scale:
                    scaling(l, name, values[i])
                else:
                    learnable_scaling(l, name, values[i])
            else:
                raise NotImplementedError("The orders are not implemented :(")
            i += 1


def collect_lipschitz(net: Module) -> Tensor:
    """This method collects lipschitz constant constants of weights.

    Args:
        net (Module): Neural net

    Returns:
        Tensor: Scalings found
    """
    modules = list(net.modules())
    scalings = []
    device = "cpu"
    for m in modules:
        if isinstance(m, ScaleWeights) or isinstance(m, LearnableScaleWeights):
            scalings.append(m.s)
            device = m.s.device
        elif m.__class__ in L_dict:
            scalings.append(torch.as_tensor(L_dict[m.__class__], device=device))  # type: ignore
    return torch.hstack(scalings)

def collect_scalings(net: Module) -> Tensor:
    """This method collects scaling constants of weights.

    Args:
        net (Module): Neural net

    Returns:
        Tensor: Scalings found
    """
    modules = list(net.modules())
    scalings = []
    for m in modules:
        if isinstance(m, ScaleWeights) or isinstance(m, LearnableScaleWeights):
            scalings.append(m.s)
    return torch.hstack(scalings)


def remove_all_parameterizations(net: Module, name: str = "weight"):
    """Removes all parameterization from all modules.

    Args:
        net (Module): Neural net
        name (str, optional): Parameter of module to which this applies. Defaults to "weight".
    """
    modules = list(net.modules())
    for l in modules:
        if "Parametrized" in l.__class__.__name__:
            remove_parametrizations(l, name)
