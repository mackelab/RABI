from random import gauss
import pytest

import torch


def benchmark_simulator(task, device):
    prior = task.get_prior(device=device)
    simulator = task.get_simulator(device=device)
    thetas = prior.sample((10000,))
    xs = simulator(thetas)


def check_base_properties_task(task, device):
    """This check base properties all tasks must satisfy"""
    output_dim = task.output_dim
    input_dim = task.input_dim

    prior = task.get_prior(device=device)
    theta = prior.sample()
    thetas = prior.sample((10000,))

    assert thetas.shape[-1] == output_dim, "Prior samples have wrong shape"
    assert theta.shape[-1] == output_dim, "Prior samples have wrong shape"
    assert (
        torch.isfinite(theta).all() and torch.isfinite(thetas).all()
    ), "Should be finite"

    simulator = task.get_simulator(device=device)
    x = simulator(theta)
    xs = simulator(thetas)

    assert x.shape[-1] == input_dim, "Simulations have wrong shape"
    assert xs.shape[-1] == input_dim, "Simulations have wrong shape"
    assert torch.isfinite(x).all() and torch.isfinite(xs).all(), "Should be finite"

    return theta, thetas, x, xs


def check_potential_properties_task(task, device, x, xs, theta, thetas):
    """This check base properties all tasks with a tractable potential must satisfy"""
    ll = task.get_loglikelihood_fn(device=device)

    likelihood1 = ll(theta)
    likelihood2 = ll(thetas)

    out1 = likelihood1.log_prob(x)
    out2 = likelihood1.log_prob(xs)
    out3 = likelihood2.log_prob(x)
    out4 = likelihood2.log_prob(xs)

    assert out1.shape[:-1] == x.shape[:-1], "Wrong shape of likelihood function"
    assert out2.shape == xs.shape[:-1], "Wrong shape of likelihood function"
    assert out3.shape[0] == thetas.shape[0], "Wrong shape of likelihood function"
    assert out4.shape[0] == thetas.shape[0], "Wrong shape of likelihood function"

    potential = task.get_potential_fn(device=device)

    out1 = potential(x, theta)
    out2 = potential(xs, theta)
    out3 = potential(x, thetas)
    out4 = potential(xs, thetas)

    assert out1.shape[:-1] == x.shape[:-1], "Wrong shape of potential function"
    assert out2.shape == xs.shape[:-1], "Wrong shape of potential function"
    assert out3.shape[0] == thetas.shape[0], "Wrong shape of potential function"
    assert out4.shape[0] == thetas.shape[0], "Wrong shape of potential function"

    return potential, ll


def check_posterior_properties(task, device, potential, x, xs, theta, thetas):
    """This checks ground truth posterior consistency."""
    posterior = task.get_true_posterior(device=device)

    q1 = posterior.condition(x)
    q2 = posterior.condition(xs)

    try:
        mean_q1 = q1.mean
        mean_q2 = q2.mean
        std_q1 = q1.stddev
        std_q2 = q2.stddev
    except:
        samples1 = q1.sample((100,))
        samples2 = q2.sample((100,))
        mean_q1 = samples1.mean(0)
        mean_q2 = samples2.mean(0)
        std_q1 = samples1.std(0)
        std_q2 = samples2.std(0)

    assert (
        (mean_q1 - theta).abs() < 5 * std_q1
    ).all(), "Should be within confidence interval"
    assert (
        (mean_q2 - thetas).abs() < 3 * std_q2
    ).float().mean() > 0.95, "Should be within confidence interval"

    thetas_post = q1.sample((1000,))
    pot_prob = potential(x, thetas_post)
    post_prob = q1.log_prob(thetas_post)

    var = (pot_prob - post_prob).var()
    assert (
        pot_prob - post_prob
    ).var() < 1e-2, f"Should be sampe up to a constant... but is {var}"


def test_gaussian_linear(gaussian_linear, device):
    task = gaussian_linear
    theta, thetas, x, xs = check_base_properties_task(task, device)

    potential, ll = check_potential_properties_task(task, device, x, xs, theta, thetas)

    check_posterior_properties(task, device, potential, x, xs, theta, thetas)

def test_glr(glr_rbf, device):
    task = glr_rbf
    theta, thetas, x, xs = check_base_properties_task(task, device)

    potential, ll = check_potential_properties_task(task, device, x, xs, theta, thetas)

    check_posterior_properties(task, device, potential, x, xs, theta, thetas)


@pytest.mark.slow
def test_lotka_volterra(lotka_volterra, device):
    task = lotka_volterra

    theta, thetas, x, xs = check_base_properties_task(task, device)

    potential, ll = check_potential_properties_task(task, device, x, xs, theta, thetas)


@pytest.mark.slow
def test_hh_task(hh, device):
    task = hh

    check_base_properties_task(task, device)

    theta, thetas, x, xs = check_base_properties_task(task, device)

    potential, ll = check_potential_properties_task(task, device, x, xs, theta, thetas)


def test_vae_task(vae, device):
    task = vae

    check_base_properties_task(task, device)


def test_square_task(square_task, device):
    task = square_task

    theta, thetas, x, xs = check_base_properties_task(task, device)

    potential, ll = check_potential_properties_task(task, device, x, xs, theta, thetas)

    # TODO
    # check_posterior_properties(task, device, potential, x, xs, theta, thetas)


def test_benchmark_lv_simulator(benchmark, lotka_volterra_default, device):
    benchmark(lambda: benchmark_simulator(lotka_volterra_default, device))


def test_benchmark_hh_simulator(benchmark, hh_default, device):
    benchmark(lambda: benchmark_simulator(hh_default, device))


def test_benchmark_gl_simulator(benchmark, gaussian_linear_default, device):
    benchmark(lambda: benchmark_simulator(gaussian_linear_default, device))


def test_benchmark_vae_simulator(benchmark, vae_default, device):
    benchmark(lambda: benchmark_simulator(vae_default, device))
