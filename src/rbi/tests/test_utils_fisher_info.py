import pytest
from rbi.models.parametetric_families import (
    BernoulliNet,
    IndependentGaussianNet,
    MultivariateGaussianNet,
)

from rbi.utils.fisher_info import (
    fisher_information_matrix,
    monte_carlo_fisher,
    score_function,
    compute_reparameterized_fisher,
)

import torch
from torch.distributions import Bernoulli, Categorical, Normal, MultivariateNormal

from rbi.models.base import ParametricProbabilisticModel, PyroFlowModel



def test_fisher_on_all_models(model, device):
  
    input_dim = model.input_dim
    net = model.to(device)
    x = torch.randn((2, input_dim), device=device)
    y = net(x)
    F = fisher_information_matrix(net, x, y, mc_samples=1000)
    F_diag = fisher_information_matrix(net, x, y, typ="diag", mc_samples=1000)
    F_trace =  fisher_information_matrix(net, x, y, typ="trace", mc_samples=1000)

    assert len(F.shape) == 3, "The output must have 3 dimensions ..."
    if isinstance(net, ParametricProbabilisticModel):
        assert (
            F.shape[0] == x.shape[0]
            and F.shape[1] == x.shape[1]
            and F.shape[2] == x.shape[1]
        ), "The shapes do not match"
    assert (
        torch.diagonal(F, dim1=-2, dim2=-1) >= 0
    ).all(), "The Fisher information matrix is psd thus must have postitive diagonal"
    assert (
        torch.isclose(F,torch.transpose(F, -2, -1)).all()
    ).all(), "The Fisher information must be symmetric"
    assert torch.isclose((torch.diagonal(F, dim1=-2, dim2=-1) -F_diag)**2, torch.zeros(1, device=device), atol=5e-1).any(), "This should be the same, typ diag may wrong"
    assert torch.isclose((torch.diagonal(F, dim1=-2, dim2=-1).sum(-1) - F_trace)**2, torch.zeros(1, device=device), atol=5e-1).any(), "This should be the same, typ trace may wrong"



def test_analytic_bernoulli_monte_carlo_fisher(batch_shape):
    probs = torch.sigmoid(torch.randn(batch_shape + (1,)))
    p = Bernoulli(probs)
    F_info = (1 / (p.probs * (1 - p.probs))).unsqueeze(-1)  # type: ignore
    F = fisher_information_matrix(p)

    assert torch.isclose(F_info, F, atol=1e-3).all(), "Wrong estimate for Bernoulli Fisher info."
    assert F.shape == F_info.shape, "Shapes wrong"


def test_score_estimation():
    p_true = torch.rand((10, 1))

    def generator(p):
        return Bernoulli(p)

    p = generator(p_true)
    x = p.sample((20,))  # type: ignore
    score_p = x / p_true - (1 - x) / (1 - p_true)
    score = score_function(x, p_true, generator, create_graph=False)

    assert score.shape == score_p.shape, "Shapes mismatch"
    assert torch.isclose(score_p, score).all(), "Score mismatch"


def test_monte_carlo_fisher_mvn(dims):
    L = torch.randn((dims, dims))
    cov = L.T @ L + torch.eye(dims) * 0.01

    def generator(mu):
        return MultivariateNormal(mu, cov)

    mu = torch.randn((1, dims))
    p = generator(mu)

    F_est = monte_carlo_fisher(generator, mu, mc_samples=8000, create_graph=False)
    F_true = p.precision_matrix

    assert torch.isclose(
        F_est, F_true, rtol=0.2, atol=0.2  # type: ignore
    ).all(), "Monte Carlo Fisher information estimate is wrong "

    # For batched parameters
    
    mu = torch.randn((5, dims))
    p = generator(mu)

    F_est = monte_carlo_fisher(generator, mu, mc_samples=8000, create_graph=False)
    F_true = p.precision_matrix

    assert torch.isclose(
        F_est, F_true, rtol=0.2, atol=0.2  # type: ignore
    ).all(), "Monte Carlo Fisher information estimate is wrong under batched input..."


@pytest.mark.parametrize(
    "model", [BernoulliNet, IndependentGaussianNet]
)
def test_monte_carlo_fisher_gradient(model):
    net = model(2, 1, hidden_dims=[10, 10], nonlinearity=torch.nn.ReLU)

    x = torch.ones((1, 2))
    F_x = compute_reparameterized_fisher(
        net, x, net(x), create_graph=True
    )
    param_grad = torch.autograd.grad(
        F_x.sum(), net.parameters(), retain_graph=True, allow_unused=True
    )

    F_x2 = monte_carlo_fisher(net,x, mc_samples=50000, create_graph=True)
    param_grad2 = torch.autograd.grad(
        F_x2.sum(), net.parameters(), retain_graph=True, allow_unused=True
    )

    assert torch.isclose(
        F_x, F_x2, atol=0.01, rtol=0.1
    ).all(), "Fisheres do not match..."
    for p1, p2 in zip(param_grad, param_grad2):
        if p1 is not None and p2 is not None:
            assert torch.isclose(
                p1, p2, atol=0.1, rtol=0.5
            ).all(), "Fisheres gradient do not match..."

