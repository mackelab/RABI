import torch
from pyro.distributions import ConditionalTransformModule
from pyro.nn import ConditionalAutoRegressiveNN, ConditionalDenseNN, MaskedLinear
from rbi.models.activations import PartialNonlinearity
from rbi.models.base import ParametricProbabilisticModel, PyroFlowModel
from rbi.models.module import Reshape, ZScoreLayer
from rbi.utils.autograd_tools import batch_jacobian
from rbi.utils.hook_control_center import (
    add_forward_hook,
    disabled_hooks,
    enabled_hooks,
    remove_forward_hooks,
)
from torch import nn

# TODO To be refactored..

def get_lipschitz_upper_bound_for_module(module, ord=2.0):
    if isinstance(module, MaskedLinear):
        return torch.linalg.matrix_norm(module.weight * module.mask, ord=ord)

    elif isinstance(module, nn.Linear):
        return torch.linalg.matrix_norm(module.weight, ord=ord)
    elif isinstance(module, nn.Sequential):
        l = 1.0
        for m in module:
            l = l * get_lipschitz_upper_bound_for_module(m, ord=ord)
        return l
    elif isinstance(module, ConditionalTransformModule):
        return get_lipschitz_upper_bound_for_module(module.nn)
    elif isinstance(module, ConditionalAutoRegressiveNN) or isinstance(
        module, ConditionalDenseNN
    ):
        lip_f = get_lipschitz_upper_bound_for_module(module.f)
        l = 1.0
        for layer in module.layers:
            l = l * get_lipschitz_upper_bound_for_module(layer) * lip_f
        return l

    elif isinstance(module, nn.ReLU):
        return 1.0
    elif isinstance(module, nn.Tanh):
        return 1.0
    elif isinstance(module, nn.Sigmoid):
        return 1.0
    elif isinstance(module, nn.Softplus):
        return 1.0
    elif isinstance(module, Reshape):
        return 1.0
    elif isinstance(module, nn.Identity):
        return 1.0
    elif isinstance(module, ZScoreLayer):
        return 1.0 / module.std**2
    elif isinstance(module, PartialNonlinearity):
        l = -torch.inf
        for act in module.nonlinearities:
            l_pro = get_lipschitz_upper_bound_for_module(act)
            if l_pro > l:
                l = l_pro
        return l
    else:
        return torch.nan


def get_lipschitz_bound_per_module(model, ord=2):
    new = dict()
    for name, m in model.named_modules():
        new[name] = get_lipschitz_upper_bound_for_module(m, ord=ord)
    return new


def get_lipschitz_upper_bound(net, ord=2.0):
    lip_constants = get_lipschitz_bound_per_module(net, ord=ord)

    L_embedding = lip_constants["embedding_net"]
    if isinstance(net, ParametricProbabilisticModel):
        L_net = lip_constants["net"]

        return L_embedding, float(L_net.detach())
    elif isinstance(net, PyroFlowModel):
        L_flow = lip_constants["t0"]
        i = 1
        while f"t{i}" in lip_constants:
            L_flow *= lip_constants[f"t{i}"]

        return L_embedding, float(L_flow.detach())


def get_lipschitz_lower_bound(net, input_dim, embedded_dim, ord=2.0):
    if isinstance(net, ParametricProbabilisticModel):
        if isinstance(net.embedding_net, nn.Identity):
            L_embedding = 1.0
        else:
            L_embedding = compute_lipschitz_lower_bound(
                net.embedding_net, input_dim, ord=ord
            )
        L_net = compute_lipschitz_lower_bound(net.net, embedded_dim, ord=ord)

        return L_embedding, float(L_net.detach())
    elif isinstance(PyroFlowModel):
        raise ValueError()


def lip_loss(func, x, ord=2.0):
    jacs = batch_jacobian(func, x, create_graph=True)
    return -torch.linalg.matrix_norm(jacs, ord=ord)


def compute_lipschitz_lower_bound(
    func,
    input_dim,
    num_samples=1000,
    num_samples_to_optimize=100,
    optim_iters=500,
    ord=2.0,
):
    add_forward_hook(func, "relu_to_softplus")

    x = torch.randn(num_samples, input_dim) * 5
    with disabled_hooks():
        loss = lip_loss(func, x, ord=ord)
    idx = loss.argsort()

    # First stage
    x = x[idx[:num_samples_to_optimize]].detach().clone().requires_grad_(True)
    optim = torch.optim.Adam([x], lr=1e-1)
    with enabled_hooks():
        for i in range(optim_iters):
            optim.zero_grad()
            l = lip_loss(func, x).mean()
            l.backward()
            optim.step()

    with disabled_hooks():
        loss = lip_loss(func, x, ord=ord)
    idx = loss.argsort()

    x = x[idx[:10]].detach().clone().requires_grad_(True)
    optim = torch.optim.Adam([x], lr=1e-3)
    with enabled_hooks():
        for i in range(optim_iters):
            optim.zero_grad()
            l = lip_loss(func, x).min()
            l.backward()
            optim.step()

    with disabled_hooks():
        loss = lip_loss(func, x, ord=ord)

    L = -loss.min()
    remove_forward_hooks(func)
    return L
