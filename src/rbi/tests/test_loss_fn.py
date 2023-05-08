

import torch

from rbi.loss.train_loss import NLLLoss, NegativeElboLoss
from rbi.loss import (
    LogLikelihoodLoss,
    NegativeLogLikelihoodLoss,
    C2ST,
    C2STBayesOptimal,
    C2STKnn,

)


def test_general_NLL_loss(nllloss, device):
    net = nllloss.model.to(device)
    input_dim = net.input_dim

    loss_fn = NLLLoss(net).to(device)

    x = torch.randn(100, input_dim, device=device)
    q = net(x)

    theta = q.sample()

    l1 = loss_fn(x, theta)

    assert l1.numel() == 1, "Should be reduced to one element"
    assert torch.isfinite(l1), "Well specified inputs should have finite loss..."

    loss_fn.reduction = None  # type: ignore
    l2 = loss_fn(x, theta)

    assert l2.shape[0] == 100, "Reduction should be disabled"
    assert l1 == torch.mean(l2), "Base reduction should be the mean..."



def test_general_ELBO_loss(negative_elbo_loss):
    net, loss_class, prior, loglikelihood_fn, potential_fn, device = negative_elbo_loss
    input_dim = net.input_dim
    loss_fn = loss_class(net, prior=prior, loglikelihood_fn=loglikelihood_fn).to(device)
    loss_fn2 = loss_class(net, potential_fn=potential_fn).to(device)

    x = torch.randn(10, input_dim, device=device)

    l1 = loss_fn(x, None)
    l2 = loss_fn2(x, None)

    assert (
        l1.shape == l2.shape
    ), "Joint or prior contrastive methods should have same shape"
    assert torch.isfinite(l1), "Prior contrastive method should be finite"
    assert torch.isfinite(l2), "Joint contrastive method should be finite"

    loss_fn = NegativeElboLoss(
        net, prior=prior, loglikelihood_fn=loglikelihood_fn, mc_samples=1000
    ).to(device)
    loss_fn2 = NegativeElboLoss(net, potential_fn=potential_fn, mc_samples=1000).to(device)

    x = torch.randn(3, input_dim, device=device)
    l1 = loss_fn(x, None)
    l2 = loss_fn2(x, None)

    assert torch.isclose(
        l1, l2, atol=5e-1
    ), "For large MC samples both methods should have similar values (this can randomly fail)."

    loss_fn = NegativeElboLoss(
        net, prior=prior, loglikelihood_fn=loglikelihood_fn, mc_samples=1
    ).to(device)
    loss_fn2 = NegativeElboLoss(net, potential_fn=potential_fn, mc_samples=1).to(device)

    x = torch.randn(10, input_dim, device=device)
    l1 = loss_fn(x, None)
    l2 = loss_fn2(x, None)

    assert torch.isfinite(l1) and torch.isfinite(
        l2
    ), "Single sample MC estimate did not work..."


def test_general_eval_loss_with_models(model, eval_loss, device):
    model = model.to(device)
    input_dim = model.input_dim
    loss_fn = eval_loss().to(device)

    if eval_loss in [NegativeLogLikelihoodLoss, LogLikelihoodLoss]:
        x = torch.randn(10, input_dim, device=device)
        q = model(x)
        theta = q.sample()
        l = loss_fn(q, theta)
    else:
        l = loss_fn(model(torch.randn(10, input_dim, device=device)), model(torch.randn(10, input_dim, device=device)))
    assert l.numel() == 1 and torch.isfinite(l), "Loss should be finite"



def test_eval_divergences(divergence, example_distributions):
    l = divergence(mc_samples=1000)

    Q1, Q2, Q3 = example_distributions

    l1 = l(Q1, Q1)
    l2 = l(Q2, Q2)
    l3 = l(Q3, Q3)

    assert (
        torch.isclose(l1,torch.zeros(1)) and torch.isclose(l2,torch.zeros(1)) and torch.isclose(l3,torch.zeros(1))
    ), "Divergence between the same distribution should be zero..."

    qs = [Q1, Q2, Q3]
    for i in range(3):
        for j in range(i + 1, 3):
            q1 = qs[i]
            q2 = qs[j]
            assert (
                not torch.isclose(l(q1,q2),torch.zeros(1))
            ), "Divergence between different distributions should be larger than zero"


def test_c2sts(c2sts, example_distributions):
    l = c2sts(mc_samples=1000)

    Q1, Q2, Q3 = example_distributions

    l1 = l(Q1, Q1)
    l2 = l(Q2, Q2)
    l3 = l(Q3, Q3)

    assert (
        torch.isclose(l1, torch.ones(1) * 0.5, atol=1e-1)
        and torch.isclose(l2, torch.ones(1) * 0.5, atol=1e-1)
        and torch.isclose(l3, torch.ones(1) * 0.5, atol=1e-1)
    ), "C2ST between the same distribution should be 0.5..."

    qs = [Q1, Q2, Q3]
    for i in range(3):
        for j in range(i + 1, 3):
            q1 = qs[i]
            q2 = qs[j]
            assert (
                l(q1, q2) > 0
            ), "Divergence between different distributions should be larger than zero"


def test_different_c2sts_should_be_same(example_distributions):
    l1 = C2ST(mc_samples=1000)
    l2 = C2STKnn(mc_samples=1000)
    l3 = C2STBayesOptimal(mc_samples=1000)

    Q1, Q2, Q3 = example_distributions
    
    qs = [Q1, Q2, Q3]
    c2sts1 = []
    c2sts2 = []
    c2sts3 = []
    for i in range(3):
        for j in range(i + 1, 3):
            q1 = qs[i]
            q2 = qs[j]

            c2sts1.append(l1(q1, q2))
            c2sts2.append(l2(q1, q2))
            c2sts3.append(l3(q1, q2))

    for c1, c2, c3 in zip(c2sts1, c2sts2, c2sts3):
        assert torch.isclose(c1, c2, atol=1e-1), "C2STs differ"
        assert torch.isclose(c2, c3, atol=1e-1), "C2STs differ"
        assert torch.isclose(c3, c1, atol=1e-1), "C2STs differ"
