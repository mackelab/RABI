from torch.distributions import biject_to
import torch

from rbi.attacks import L2UniformNoiseAttack, L2PGDAdamAttack
from rbi.loss import ReverseKLLoss


def test_approximation_metric(model, task, approximation_metric, device):

    prior = task.get_prior(device=device)
    simulator = task.get_simulator(device=device)

    model = model(
        task.input_dim,
        task.output_dim,
        output_transform=biject_to(prior.support),
        hidden_dims=[10],
    ).to(device)

    try:
        post = task.get_posterior(device=device)
    except:
        post = None

    try:
        potential_fn = task.get_potential_fn(device=device)
    except:
        potential_fn = None

    m = approximation_metric(
        model,
        prior=prior,
        simulator=simulator,
        ground_truth=post,
        potential_fn=potential_fn,
        device=device,
        mc_samples=3,
    )

    if m.requires_posterior and post is None:
        return

    theta = prior.sample((5,))
    x = simulator(theta)
    x = (x - x.mean(keepdim=True, dim=0))/x.std(dim=0, keepdim=True)

    if not m.requires_thetas:
        outs = m.eval(x)
    else:
        outs = m.eval(x, theta)

    assert torch.isfinite(outs), "Metric should be finite"


def test_rob_metric(model, task, rob_metric, device):
    prior = task.get_prior(device=device)
    simulator = task.get_simulator(device=device)

    model = model(
        task.input_dim, task.output_dim, output_transform=biject_to(prior.support)
    ).to(device)

    attack1 = L2UniformNoiseAttack(model, eps=0.1)
    attack2 = L2PGDAdamAttack(model, ReverseKLLoss(mc_samples=1), nb_iter=1, eps=0.1)

    m1 = rob_metric(model, attack1, mc_samples=50, attack_attemps=1, device=device)
    m2 = rob_metric(model, attack2, mc_samples=50, attack_attemps=1, device = device)

    theta = prior.sample((10,))
    x = simulator(theta)

    val1 = m1.eval(x)
    val2 = m2.eval(x)

    assert torch.isfinite(val1), "Metric should be finite"
    assert torch.isfinite(val2), "Metric should be finite"
