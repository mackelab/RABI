import matplotlib.pyplot as plt
from rbibm.utils.utils_data import query
from rbibm.plots.utils import (
    maybe_get_model_by_id,
    maybe_get_x_tilde_from_id,
    maybe_get_task_by_id,
)
from rbibm.plots.predictives_per_task import get_predicitve_plotting_function
from rbibm.plots.custom_distribution_plots import *

import torch
from sbi.analysis import pairplot, marginal_plot
import matplotlib
from matplotlib.colors import to_rgb,to_hex

from matplotlib.colors import ListedColormap

import math
from scipy.stats import gaussian_kde


def set_normal_adversarial_colors(color_normal, color_adversarial):
    plt.rcParams["axes.prop_cycle"]._left[0] = {"color": to_hex(color_normal)}
    plt.rcParams["axes.prop_cycle"]._left[1] = {"color": to_hex(color_adversarial)}

def plot_posterior(
    name,
    task=None,
    model_name=None,
    metric_approx_clean=None,
    metric_approx_tilde=None,
    defense=None,
    loss=None,
    type="pairplot",
    verbose=True,
    model=None,
    n_samples=15000,
    x_o=None,
    theta_o=None,
    device="cpu",
    plotting_kwargs={},
    **kwargs,
):

    df_s = query(
        name,
        task=task,
        metric_approx_clean=metric_approx_clean,
        model_name=model_name,
        metric_approx_tilde=metric_approx_tilde,
        loss=loss,
        defense=defense,
        **kwargs,
    )

    if len(df_s) == 0:
        raise ValueError("No data for these keywords....")

    id = df_s.iloc[0].id
    if verbose:
        print("Following data row is used:")
        print(
            f"Id: {df_s.iloc[0].id}, Task: {df_s.iloc[0].task}, Model: {df_s.iloc[0].model_name}, Defense: {df_s.iloc[0].defense}, Loss: {df_s.iloc[0].loss}, N_train: {df_s.iloc[0].N_train}"
        )
    task = maybe_get_task_by_id(name, id, task)
    model = maybe_get_model_by_id(name, model_id=id, model=model).to(device)

    if task is not None:
        prior = task.get_prior(device=device)
        simulator = task.get_simulator(device=device)

    if x_o is None and task is not None:
        theta_o = prior.sample()
        x_o = simulator(theta_o)

    if x_o is None and task is None:
        raise ValueError("Either give me the task you trained on or an obsevation...")

    samples = model(x_o).sample((n_samples,))

    if type == "pairplot":
        fig, axes = custom_pairplot(
            [samples.reshape(n_samples, -1).cpu()],
            points=[theta_o.reshape(1, -1).cpu()],
            **plotting_kwargs,
        )
    elif type == "marginalplot":
        fig, axes = custom_marginal_plot(
            [samples.reshape(n_samples, -1).cpu()],
            points=[theta_o.reshape(1, -1).cpu()],
            **plotting_kwargs,
        )
    elif type == "pairplot_sbi":
        fig, axes = pairplot(
            [samples.reshape(n_samples, -1).cpu()],
            points=[theta_o.reshape(1, -1).cpu()],
            **get_plotting_kwargs_pairplot(plotting_kwargs),
        )
    elif type == "marginalplot_sbi":
        fig, axes = marginal_plot(
            [samples.reshape(n_samples, -1).cpu()],
            points=[theta_o.reshape(1, -1).cpu()],
            **get_plotting_kwargs_marginal_plot(plotting_kwargs),
        )
    else:
        raise ValueError("Unknown posterior")
    return fig, axes


def plot_adversarial_posterior(
    name,
    task=None,
    model_name=None,
    metric_approx_clean=None,
    defense=None,
    loss=None,
    type="pairplot",
    verbose=True,
    model=None,
    idx_adv_example=0,
    n_samples=15000,
    x=None,
    theta=None,
    x_tilde=None,
    device="cpu",
    plotting_kwargs={},
    plot_adv_example=None,
    **kwargs,
):

    df_s = query(
        name,
        task=task,
        metric_approx_clean=metric_approx_clean,
        model_name=model_name,
        metric_approx_tilde=metric_approx_clean,
        loss=loss,
        defense=defense,
        **kwargs,
    )

    if len(df_s) == 0:
        raise ValueError("No data for these keywords....")

    id = df_s.iloc[0].id
    id_adversarial = df_s.iloc[0].id_adversarial
    if verbose:
        print("Following data row is used:")
        print(
            f"Id: {df_s.iloc[0].id}, Task: {df_s.iloc[0].task}, Model: {df_s.iloc[0].model_name}, Defense: {df_s.iloc[0].defense}, Loss: {df_s.iloc[0].loss}, N_train: {df_s.iloc[0].N_train}, Id adversarial ={id_adversarial}, Attack:{df_s.iloc[0].attack},Attack loss_fn:{df_s.iloc[0].attack_loss_fn}, Metric: {df_s.iloc[0].metric_rob} "
        )

    model = maybe_get_model_by_id(name, model_id=id, model=model).to(device)
    x, theta, x_tilde = maybe_get_x_tilde_from_id(
        name, id_adversarial, x, theta, x_tilde
    )

    x = x[idx_adv_example].to(device)
    if theta is not None:
        theta = theta[idx_adv_example].to(device)
    x_tilde = x_tilde[idx_adv_example].to(device)

    q = model(x)
    q_tilde = model(x_tilde)

    samples = q.sample((n_samples,))
    samples_tilde = q_tilde.sample((n_samples,))

    if type == "pairplot":
        fig, axes = custom_pairplot(
            [
                samples.reshape(n_samples, -1).cpu(),
                samples_tilde.reshape(n_samples, -1).cpu(),
            ],
            points=[theta.reshape(1, -1).cpu()],
            **plotting_kwargs,
        )
    elif type == "marginalplot":
        fig, axes = custom_marginal_plot(
            [
                samples.reshape(n_samples, -1).cpu(),
                samples_tilde.reshape(n_samples, -1).cpu(),
            ],
            points=[theta.reshape(1, -1).cpu()],
            **plotting_kwargs,
        )
    elif type == "2djointplot":
        fig, axes = custom_2d_joint_plot(
            [
                samples.reshape(n_samples, -1).cpu(),
                samples_tilde.reshape(n_samples, -1).cpu(),
            ],
            points=[theta.reshape(1, -1).cpu()],
            **plotting_kwargs,
        )
    elif type == "pairplot_sbi":
        fig, axes = pairplot(
            [
                samples.reshape(n_samples, -1).cpu(),
                samples_tilde.reshape(n_samples, -1).cpu(),
            ],
            points=[theta.reshape(1, -1).cpu()],
            **get_plotting_kwargs_pairplot(plotting_kwargs),
        )
    elif type == "marginalplot_sbi":
        fig, axes = marginal_plot(
            [
                samples.reshape(n_samples, -1).cpu(),
                samples_tilde.reshape(n_samples, -1).cpu(),
            ],
            points=[theta.reshape(1, -1).cpu()],
            **get_plotting_kwargs_marginal_plot(plotting_kwargs),
        )
    else:
        raise ValueError("Unknown posterior")

    if plot_adv_example is not None:
        plot_adversarial_example_on_top(fig, plot_adv_example, task, x, x_tilde)

    return fig, axes


def plot_posterior_predictive(
    name,
    task=None,
    model_name=None,
    metric_approx_clean=None,
    defense=None,
    loss=None,
    verbose=True,
    model=None,
    x_o=None,
    theta_o=None,
    device="cpu",
    plotting_kwargs={},
    **kwargs,
):

    df_s = query(
        name,
        task=task,
        metric_approx_clean=metric_approx_clean,
        model_name=model_name,
        metric_approx_tilde=metric_approx_clean,
        loss=loss,
        defense=defense,
        **kwargs,
    )

    plot_fn = get_predicitve_plotting_function(task)

    if len(df_s) == 0:
        raise ValueError("No data for these keywords....")

    id = df_s.iloc[0].id
    if verbose:
        print("Following data row is used:")
        print(
            f"Id: {df_s.iloc[0].id}, Task: {df_s.iloc[0].task}, Model: {df_s.iloc[0].model_name}, Defense: {df_s.iloc[0].defense}, Loss: {df_s.iloc[0].loss}, N_train: {df_s.iloc[0].N_train}"
        )
    task = maybe_get_task_by_id(name, id, task)
    model = maybe_get_model_by_id(name, model_id=id, model=model).to(device)

    if task is not None:
        prior = task.get_prior(device=device)
        simulator = task.get_simulator(device=device)

    if x_o is None and task is not None:
        theta_o = prior.sample()
        x_o = simulator(theta_o)

    return plot_fn(model, task, x_o, **plotting_kwargs)


def plot_adversarial_posterior_predictive(
    name,
    task=None,
    model_name=None,
    metric_approx_clean=None,
    defense=None,
    loss=None,
    verbose=True,
    model=None,
    idx_adv_example=0,
    x=None,
    theta=None,
    x_tilde=None,
    device="cpu",
    plotting_kwargs={},
    **kwargs,
):

    df_s = query(
        name,
        task=task,
        metric_approx_clean=metric_approx_clean,
        model_name=model_name,
        metric_approx_tilde=metric_approx_clean,
        loss=loss,
        defense=defense,
        **kwargs,
    )

    plot_fn = get_predicitve_plotting_function(task)

    if len(df_s) == 0:
        raise ValueError("No data for these keywords....")

    id = df_s.iloc[0].id
    id_adversarial = df_s.iloc[0].id_adversarial
    if verbose:
        print("Following data row is used:")
        print(
            f"Id: {df_s.iloc[0].id}, Task: {df_s.iloc[0].task}, Model: {df_s.iloc[0].model_name}, Defense: {df_s.iloc[0].defense}, Loss: {df_s.iloc[0].loss}, N_train: {df_s.iloc[0].N_train}, Id adversarial ={id_adversarial}, Attack:{df_s.iloc[0].attack},Attack loss_fn:{df_s.iloc[0].attack_loss_fn}, Metric: {df_s.iloc[0].metric_rob} "
        )
    task = maybe_get_task_by_id(name, id, task)
    model = maybe_get_model_by_id(name, model_id=id, model=model).to(device)
    x, theta, x_tilde = maybe_get_x_tilde_from_id(
        name, id_adversarial, x, theta, x_tilde
    )

    return plot_fn(
        model,
        task,
        torch.stack([x[idx_adv_example], x_tilde[idx_adv_example]]),
        **plotting_kwargs,
    )


def add_plot_predictive(fig, x, x_tilde, **kwargs):
    name = kwargs.pop("name")
    _, axes = plot_adversarial_posterior_predictive(name,x=x, x_tilde=x_tilde, **kwargs)


def plot_adversarial_example_on_top(fig, type, task, x, x_tilde, **kwargs):
    # For some tasks this can be done better
    if type == "x":
        return add_adversarial_example_and_x_general(fig, x, x_tilde, **kwargs)
    elif type == "diff":
        return add_adversarial_example_and_x_difference(fig, x, x_tilde, **kwargs)
    elif type == "predicitve":
        return add_plot_predictive(fig, x, x_tilde, **kwargs)
