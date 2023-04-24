from rbibm.plots.custom_distribution_plots import *

import matplotlib.pyplot as plt
from math import sqrt
from torchvision.utils import make_grid

from matplotlib.colors import to_rgb
import torch
from torch import Tensor

import numpy as np

from typing import Callable, Union

from scipy.signal import savgol_filter
from rbi.models.base import ParametricProbabilisticModel, PyroFlowModel
from rbibm.tasks.base import Task


def get_predicitve_plotting_function(task: str) -> Callable:
    """Returns a nice plotting function for the predictive distribution of certain tasks.

    Args:
        task (str): Name of the tast

    Returns:
        Callable: Function that plots the predictive.
    """
    if task == "gaussian_linear":
        return plot_gl_predictives
    elif task == "lotka_volterra":
        return plot_lv_predictives
    elif task == "vae_task":
        return plot_vae_predictives
    elif task == "hudgkin_huxley":
        return plot_hh_predictives
    elif task == "spatial_sir":
        return plot_spatial_sir_predictives
    elif task == "rbf_regression":
        return plot_glr_predictives
    elif task == "sir":
        return plot_sir_predictives
    elif task == "pyloric":
        return plot_pyloric_predictives
    else:
        return plot_gl_predictives


def plot_gl_predictives(
    model: Union[ParametricProbabilisticModel, PyroFlowModel],
    task: Task,
    xs: Tensor,
    type: str = "marginalplot",
    n_samples: int = 10000,
    colors=None,
    device: str = "cpu",
    **kwargs
):
    """Plot the predictive distribution of a Gaussian Linear task.

    Args:
        model (Union[ParametricProbabilisticModel, PyroFlowModel]): Model
        task (Task): Task
        xs (Tensor): Points on which we should eval the predictive
        type (str, optional): Plotting type. Defaults to "marginalplot".
        n_samples (int, optional): Number of samples. Defaults to 1000.
        device (str, optional): Device. Defaults to "cpu".

    Returns:
        fig,axes: Figure and axes
    """
    xs = xs.reshape(-1, xs.shape[-1])
    model = model.to(device)
    simulator = task.get_simulator(device=device)
    q = model(xs)
    predictives = simulator(q.sample((n_samples,))).transpose(0, 1)
    predictives = [samples for samples in predictives]
    points = [x for x in xs]
    if type == "marginalplot":
        return custom_marginal_plot(predictives, points, **kwargs)
    elif type == "pairplot":
        return custom_pairplot(predictives, points, **kwargs)


def plot_lv_predictives(
    model: Union[ParametricProbabilisticModel, PyroFlowModel],
    task: Task,
    xs: Tensor,
    time_points_predictive=500,
    n_samples=5000,
    markersize=None,
    lw=None,
    device="cpu",
    colors=None,
    figsize=(10, 3),
):

    xs = xs.reshape(-1, xs.shape[-1])

    model = model.to(device)
    t_max = task.t_max

    task.time_points_observed = time_points_predictive
    simulator = task.get_simulator(device=device)
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    for i, x_obs in enumerate(xs):
        q = model(x_obs)
        samples = q.sample((n_samples,))
        x_predictives = simulator(samples)

        color1 = colors[i]
        color2 = colors[i] #[max(c - 0.3, 0) for c in to_rgb(colors[i])]

        time_points_obs = x_obs.shape[-1] // 2
        time_points_predicted = time_points_predictive

        t_obs = torch.linspace(0, t_max, time_points_obs)
        t_predictive = torch.linspace(0, t_max, time_points_predicted)

        x_obs = x_obs.reshape(2, time_points_obs)
        x_predictives = x_predictives.reshape(-1, 2, time_points_predicted)

        x_predictive_q01 = savgol_filter(x_predictives.quantile(0.01, axis=0), 40, 4)
        x_predictive_q05 = savgol_filter(x_predictives.quantile(0.05, axis=0), 40, 4)
        x_predictive_q17 = savgol_filter(x_predictives.quantile(0.175, axis=0), 40, 4)
        x_predictive_q82 = savgol_filter(x_predictives.quantile(0.825, axis=0), 40, 4)
        x_predictive_q95 = savgol_filter(x_predictives.quantile(0.95, axis=0), 40, 4)
        x_predictive_q99 = savgol_filter(x_predictives.quantile(0.99, axis=0), 40, 4)

        axes[0].plot(t_obs, x_obs[0, :], color=color1, markersize=markersize, lw=lw)
        axes[1].plot(t_obs, x_obs[1, :], color=color2, markersize = markersize, lw=lw)
        axes[0].fill_between(
            t_predictive,
            x_predictive_q01[0, :],
            x_predictive_q99[0, :],
            alpha=0.1,
            color=color1,
        )
        axes[0].fill_between(
            t_predictive,
            x_predictive_q05[0, :],
            x_predictive_q95[0, :],
            alpha=0.2,
            color=color1,
        )
        axes[0].fill_between(
            t_predictive,
            x_predictive_q17[0, :],
            x_predictive_q82[0, :],
            alpha=0.3,
            color=color1,
        )

        axes[1].fill_between(
            t_predictive,
            x_predictive_q01[1, :],
            x_predictive_q99[1, :],
            alpha=0.1,
            color=color2,
        )
        axes[1].fill_between(
            t_predictive,
            x_predictive_q05[1, :],
            x_predictive_q95[1, :],
            alpha=0.2,
            color=color2,
        )
        axes[1].fill_between(
            t_predictive,
            x_predictive_q17[1, :],
            x_predictive_q82[1, :],
            alpha=0.3,
            color=color2,
        )

    axes[0].set_ylabel("Population density")
    axes[0].set_xlabel("Time")
    axes[1].set_xlabel("Time")
    axes[0].set_title("Prey")
    axes[1].set_title("Predator")
    plt.setp(axes[1].get_yticklabels(), visible=False)

    fig.tight_layout()

    return fig, axes


def plot_sir_predictives(
    model,
    task,
    xs,
    time_points_predictive=500,
    n_samples=5000,
    device="cpu",
    colors=None,
    all_states=True,
    figsize=(10, 3),
):

    xs = xs.reshape(-1, xs.shape[-1])

    model = model.to(device)
    t_max = task.t_max

    task.time_points_observed = time_points_predictive
    simulator = task.ode_sol

    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    if all_states:
        fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)

        for i, x_obs in enumerate(xs):
            q = model(x_obs)
            samples = q.sample((n_samples,))
            S, I, R = simulator(samples)

            S = S.transpose(0, 1).squeeze(-1)
            I = I.transpose(0, 1).squeeze(-1)
            I = torch.distributions.LogNormal(
                torch.log(I + 1e-4), task.observation_noise
            ).sample()
            R = R.transpose(0, 1).squeeze(-1)

            color1 = colors[i]
            color2 = colors[i]#[max(c - 0.3, 0) for c in to_rgb(colors[i])]
            color3 = colors[i]#[min(c + 0.3, 1) for c in to_rgb(colors[i])]

            time_points_obs = x_obs.shape[-1]
            time_points_predicted = time_points_predictive

            t_obs = torch.linspace(0, t_max, time_points_obs)
            t_predictive = torch.linspace(0, t_max, time_points_predicted)

            S_predictive_q01 = savgol_filter(S.quantile(0.01, axis=0), 50, 4)
            S_predictive_q05 = savgol_filter(S.quantile(0.05, axis=0), 50, 4)
            S_predictive_q17 = savgol_filter(S.quantile(0.175, axis=0), 50, 4)
            S_predictive_q82 = savgol_filter(S.quantile(0.825, axis=0), 50, 4)
            S_predictive_q95 = savgol_filter(S.quantile(0.95, axis=0), 50, 4)
            S_predictive_q99 = savgol_filter(S.quantile(0.99, axis=0), 50, 4)

            I_predictive_q01 = savgol_filter(I.quantile(0.01, axis=0), 50, 4)
            I_predictive_q05 = savgol_filter(I.quantile(0.05, axis=0), 50, 4)
            I_predictive_q17 = savgol_filter(I.quantile(0.175, axis=0), 50, 4)
            I_predictive_q82 = savgol_filter(I.quantile(0.825, axis=0), 50, 4)
            I_predictive_q95 = savgol_filter(I.quantile(0.95, axis=0), 50, 4)
            I_predictive_q99 = savgol_filter(I.quantile(0.99, axis=0), 50, 4)

            R_predictive_q01 = savgol_filter(R.quantile(0.01, axis=0), 50, 4)
            R_predictive_q05 = savgol_filter(R.quantile(0.05, axis=0), 50, 4)
            R_predictive_q17 = savgol_filter(R.quantile(0.175, axis=0), 50, 4)
            R_predictive_q82 = savgol_filter(R.quantile(0.825, axis=0), 50, 4)
            R_predictive_q95 = savgol_filter(R.quantile(0.95, axis=0), 50, 4)
            R_predictive_q99 = savgol_filter(R.quantile(0.99, axis=0), 50, 4)

            axes[1].plot(t_obs, x_obs, lw=2, color=color1)
            axes[1].fill_between(
                t_predictive,
                I_predictive_q01,
                I_predictive_q99,
                alpha=0.1,
                color=color1,
            )
            axes[1].fill_between(
                t_predictive,
                I_predictive_q05,
                I_predictive_q95,
                alpha=0.2,
                color=color1,
            )
            axes[1].fill_between(
                t_predictive,
                I_predictive_q17,
                I_predictive_q82,
                alpha=0.3,
                color=color1,
            )

            axes[0].fill_between(
                t_predictive,
                S_predictive_q01,
                S_predictive_q99,
                alpha=0.1,
                color=color2,
            )
            axes[0].fill_between(
                t_predictive,
                S_predictive_q05,
                S_predictive_q95,
                alpha=0.2,
                color=color2,
            )
            axes[0].fill_between(
                t_predictive,
                S_predictive_q17,
                S_predictive_q82,
                alpha=0.3,
                color=color2,
            )

            axes[2].fill_between(
                t_predictive,
                R_predictive_q01,
                R_predictive_q99,
                alpha=0.1,
                color=color3,
            )
            axes[2].fill_between(
                t_predictive,
                R_predictive_q05,
                R_predictive_q95,
                alpha=0.2,
                color=color3,
            )
            axes[2].fill_between(
                t_predictive,
                R_predictive_q17,
                R_predictive_q82,
                alpha=0.3,
                color=color3,
            )

        axes[0].set_ylabel("Population density")
        axes[0].set_xlabel("Time")
        axes[1].set_xlabel("Time")
        axes[2].set_xlabel("Time")
        axes[0].set_title("Susceptible")
        axes[1].set_title("Infected")
        axes[2].set_title("Recovered")
        plt.setp(axes[1].get_yticklabels(), visible=False)
        plt.setp(axes[2].get_yticklabels(), visible=False)

        return fig, axes
    else:
        fig = plt.figure(figsize=(figsize[1],figsize[1]))
        axes = plt.gca()
        for i, x_obs in enumerate(xs):
            q = model(x_obs)
            samples = q.sample((n_samples,))
            S, I, R = simulator(samples)

            I = I.transpose(0, 1).squeeze(-1)
            I = torch.distributions.LogNormal(
                torch.log(I + 1e-4), task.observation_noise
            ).sample()
           

           
            color1 = colors[i]#[max(c - 0.3, 0) for c in to_rgb(colors[i])]
           

            time_points_obs = x_obs.shape[-1]
            time_points_predicted = time_points_predictive

            t_obs = torch.linspace(0, t_max, time_points_obs)
            t_predictive = torch.linspace(0, t_max, time_points_predicted)

            I_predictive_q01 = savgol_filter(I.quantile(0.01, axis=0), 50, 4)
            I_predictive_q05 = savgol_filter(I.quantile(0.05, axis=0), 50, 4)
            I_predictive_q17 = savgol_filter(I.quantile(0.175, axis=0), 50, 4)
            I_predictive_q82 = savgol_filter(I.quantile(0.825, axis=0), 50, 4)
            I_predictive_q95 = savgol_filter(I.quantile(0.95, axis=0), 50, 4)
            I_predictive_q99 = savgol_filter(I.quantile(0.99, axis=0), 50, 4)

            
            axes.plot(t_obs, x_obs, lw=2, color=color1)
            axes.fill_between(
                t_predictive,
                I_predictive_q01,
                I_predictive_q99,
                alpha=0.1,
                color=color1,
            )
            axes.fill_between(
                t_predictive,
                I_predictive_q05,
                I_predictive_q95,
                alpha=0.2,
                color=color1,
            )
            axes.fill_between(
                t_predictive,
                I_predictive_q17,
                I_predictive_q82,
                alpha=0.3,
                color=color1,
            )
            axes.set_title("Infections")
            axes.set_ylabel("Population density")
            axes.set_xlabel("Time")
            #axes.set_xlim(0., t_max)

        return fig, axes

def plot_hh_predictives(
    model,
    task,
    xs,
    figsize=(10,3),
    ax=None,
    colors=None,
    time_points_predictive=1000,
    samples_to_plot=3,
    lw=None,
    n_samples=5000,
    device="cpu",
):
    xs = xs.reshape(-1, xs.shape[-1])

    model = model.to(device)
    t_max = task.t_max

    task.time_points_observed = time_points_predictive
    simulator = task.get_simulator(device=device)

    t_predictive = torch.linspace(0, t_max, time_points_predictive)
    t_obs = torch.linspace(0, t_max, xs.shape[-1])

    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    
    if figsize is None:
        fig = plt.figure()
        ax = plt.gca()
    else:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()


    for i, x_obs in enumerate(xs):
        color = colors[i]
        q = model(x_obs)
        samples = q.sample((n_samples,))
        x_predictives = simulator(samples)

        x_predictive_q01 = savgol_filter(x_predictives.quantile(0.01, axis=0), 10, 2)
        x_predictive_q05 = savgol_filter(x_predictives.quantile(0.05, axis=0), 10, 2)
        x_predictive_q17 = savgol_filter(x_predictives.quantile(0.175, axis=0), 10, 2)
        x_predictive_q82 = savgol_filter(x_predictives.quantile(0.825, axis=0), 10, 2)
        x_predictive_q95 = savgol_filter(x_predictives.quantile(0.95, axis=0), 10, 2)
        x_predictive_q99 = savgol_filter(x_predictives.quantile(0.99, axis=0), 10, 2)

        ax.fill_between(
            t_predictive, x_predictive_q01, x_predictive_q99, alpha=0.1, color=color
        )
        ax.fill_between(
            t_predictive, x_predictive_q05, x_predictive_q95, alpha=0.2, color=color
        )
        ax.fill_between(
            t_predictive, x_predictive_q17, x_predictive_q82, alpha=0.3, color=color
        )

        ax.plot(t_obs, x_obs.squeeze(), lw=lw, color=color)

        for i in range(samples_to_plot):
            ax.plot(t_predictive, x_predictives[i], ":", color=color, lw=lw)

    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Voltage [mV]")
    ax.set_xlim(0,t_max)

    return fig, ax

def plot_glr_predictives(
    model,
    task,
    xs,
    figsize=(10,3),
    ax=None,
    n_samples=10000,
    device="cpu",
):
    xs = xs.reshape(-1, xs.shape[-1])

    model = model.to(device)
    simulator = task.sample_functions

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    
    if figsize is None:
        fig = plt.figure()
        ax = plt.gca()
    else:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()


    for i, x_obs in enumerate(xs):
        color = colors[i]
        q = model(x_obs)
        samples = q.sample((n_samples,))
        l, x_predictives = simulator(samples.reshape(-1, samples.shape[-1]))
        l = l[0]

        x_predictive_q01 = savgol_filter(x_predictives.quantile(0.01, axis=0), 50, 2)
        x_predictive_q05 = savgol_filter(x_predictives.quantile(0.05, axis=0), 50, 2)
        x_predictive_q17 = savgol_filter(x_predictives.quantile(0.175, axis=0), 50, 2)
        x_predictive_q82 = savgol_filter(x_predictives.quantile(0.825, axis=0), 50, 2)
        x_predictive_q95 = savgol_filter(x_predictives.quantile(0.95, axis=0), 50, 2)
        x_predictive_q99 = savgol_filter(x_predictives.quantile(0.99, axis=0), 50, 2)

        ax.fill_between(
            l, x_predictive_q01, x_predictive_q99, alpha=0.1, color=color
        )
        ax.fill_between(
            l, x_predictive_q05, x_predictive_q95, alpha=0.2, color=color
        )
        ax.fill_between(
            l, x_predictive_q17, x_predictive_q82, alpha=0.3, color=color
        )

        x,y = x_obs.split(task.N, -1)
        idx = x.argsort()
        ax.plot(x[idx].squeeze(), y[idx].squeeze(), "o", color=color)

       
    return fig, ax


def plot_vae_predictives(
    model,
    task,
    xs,
    ax=None,
    figsize=None,
    n_samples=16,
    grid_padding=10,
    grid_pad_value=1.,
    nrow=8,
    device="cpu",
    grid_spec_width_ratios= (0.95,4),
    titles=[
        ["Observation", "Posterior predictive"],
        ["Adversarial example", "Adversarial posterior predictive"],
    ],
):

    xs = xs.reshape(-1, xs.shape[-1])
    model = model.to(device)
    simulator = task.get_simulator(device=device)
    q = model(xs)
    predictives = simulator(q.sample((n_samples,))).transpose(0, 1)
    predictives = [samples for samples in predictives]
    points = [x for x in xs]

    if ax is None and figsize is not None:
        fig, axes = plt.subplots(
            len(points), 2, gridspec_kw={"width_ratios": grid_spec_width_ratios}, figsize=figsize
        )
    elif ax is None and figsize is None:
        fig, axes = plt.subplots(
            len(points), 2, gridspec_kw={"width_ratios": grid_spec_width_ratios}, figsize=(10, 5)
        )
    else:
        fig = None

    if len(points) == 1:
        axes = np.array([axes])

    for i, (x_obs, x_predictives) in enumerate(zip(points, predictives)):

        dim = int(sqrt(xs.shape[-1]))
        x_img = x_predictives.reshape(-1, 1, dim, dim)
        x_obs_img = x_obs.reshape(dim, dim)
        axes[i, 0].imshow(x_obs_img, cmap="binary_r", vmin=0, vmax=1.)
        axes[i, 1].imshow(
            make_grid(x_img, padding=grid_padding, pad_value=grid_pad_value, nrow=nrow).mean(0), cmap="binary_r", vmin=0, vmax=1.
        )
        axes[i, 0].set_title(titles[i][0])
        axes[i, 1].set_title(titles[i][1])

        axes[i, 0].axes.set_axis_off()
        axes[i, 1].axes.set_axis_off()

        if fig is not None:
            fig.tight_layout()

    return fig, axes


def plot_rps_predictives(
    model, task, xs, ax=None, figsize=None, n_samples=16, device="cpu"
):

    xs = xs.reshape(-1, xs.shape[-1])
    model = model.to(device)
    simulator = task.get_simulator(device=device)
    q = model(xs)
    predictives = simulator(q.sample((n_samples,))).transpose(0, 1)
    predictives = [samples for samples in predictives]
    points = [x for x in xs]

    if ax is None and figsize is not None:
        fig, axes = plt.subplots(
            len(points), 2, gridspec_kw={"width_ratios": (0.95, 4)}, figsize=figsize
        )
    elif ax is None and figsize is None:
        fig, axes = plt.subplots(
            len(points), 2, gridspec_kw={"width_ratios": (0.95, 4)}, figsize=(10, 5)
        )
    else:
        fig = None

    if len(points) == 1:
        axes = np.array([axes])

    CMAP = ListedColormap(["white", "C0", "C1", "C2"])

    for i, (x_obs, x_predictives) in enumerate(zip(points, predictives)):

        dim = int(sqrt(xs.shape[-1]))
        x_img = x_predictives.reshape(-1, 1, dim, dim)
        x_obs_img = x_obs.reshape(dim, dim)
        axes[i, 0].imshow(x_obs_img, cmap="binary_r")
        axes[i, 1].imshow(
            make_grid(x_img, padding=10, pad_value=1.0).mean(0),
            cmap=CMAP,
            vmin=0.0,
            vmax=3.0,
            filterrad=1.0,
        )
        axes[i, 0].set_title("Observation", pad=15)
        axes[i, 1].set_title("Posterior predictive samples")

        axes[i, 0].axes.set_axis_off()
        axes[i, 1].axes.set_axis_off()

        if fig is not None:
            fig.tight_layout()

    return fig, axes


def plot_spatial_sir_predictives(
    model, task, xs, ax=None, figsize=None, n_samples=16, nrow=8, grid_padding=5, grid_pad_value=0., grid_spec_width_ratios= (0.95,4), pad_label=0,device="cpu",titles=[
        ["Observation", "Posterior predictive"],
        ["Adversarial example", "Adversarial posterior predictive"],
    ]
):

    xs = xs.reshape(-1, xs.shape[-1])
    model = model.to(device)
    simulator = task.get_simulator(device=device)
    q = model(xs)
    predictives = simulator(q.sample((n_samples,))).transpose(0, 1)
    predictives = [samples for samples in predictives]
    points = [x for x in xs]

    if ax is None and figsize is not None:
        fig, axes = plt.subplots(
            len(points), 2, gridspec_kw={"width_ratios": grid_spec_width_ratios}, figsize=figsize
        )
    elif ax is None and figsize is None:
        fig, axes = plt.subplots(
            len(points), 2, gridspec_kw={"width_ratios": grid_spec_width_ratios}, figsize=(10, 5)
        )
    else:
        fig = None

    if len(points) == 1:
        axes = np.array([axes])

    CMAP = "Reds"

    for i, (x_obs, x_predictives) in enumerate(zip(points, predictives)):

        dim = int(sqrt(xs.shape[-1]))
        x_img = x_predictives.reshape(-1, 1, dim, dim)
        x_obs_img = x_obs.reshape(dim, dim)
        axes[i, 0].imshow(x_obs_img, cmap=CMAP, vmin=0, vmax=1.0)
        axes[i, 1].imshow(
            make_grid(x_img, padding=grid_padding, pad_value=grid_pad_value, nrow=nrow).mean(0),
            cmap=CMAP,
            vmin=0.0,
            vmax=1.0,
        )
        axes[i, 0].set_title(titles[i][0], pad=pad_label)
        axes[i, 1].set_title(titles[i][1])

        axes[i, 0].axes.set_axis_off()
        axes[i, 1].axes.set_axis_off()

        if fig is not None:
            fig.tight_layout()

    return fig, axes


def plot_pyloric_predictives(
    model, task, xs, ax=None, figsize=None, n_samples=16, device="cpu", colors=None, titles=["Observation", "Adversarial"]
):
    xs = xs.reshape(-1, xs.shape[-1])
    model = model.to(device)
    simulator = task.get_simulator(device=device)
    q = model(xs)
    predictives = simulator(q.sample((n_samples,))).transpose(0,1)
    predictives = [samples.reshape(n_samples, 3, 800) for samples in predictives]
    print(len(predictives))
    points = [x.reshape(3, 800) for x in xs]
    print(len(points))
    if ax is None and figsize is not None:
        fig, axes = plt.subplots(
            3,len(points), figsize=figsize
        )
    elif ax is None and figsize is None:
        fig, axes = plt.subplots(
            3,len(points), figsize=(7, 3)
        )
    else:
        fig = None

    if len(points) == 1:
        axes = np.array([[ax] for ax in axes])

    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    t = np.arange(0, 2000, 2.5)
    global_min = -80
    global_max = 60
    neuron_labels = ["AB/PD", "LP", "PY"]

    for i, (x_obs, x_predictives) in enumerate(zip(points, predictives)):
      

        for j, ax in enumerate(axes[:,i]):
            ax.plot(t, x_obs[j], color=colors[i])
            for x in x_predictives:
                ax.plot(t, x[j], color=colors[i], alpha=0.1)
            ax.set_ylim([global_min, global_max])
            if i > 0:
                ax.set_yticklabels([])
            if titles is not None:
                if j == 0:
                    ax.set_title(titles[i])

            if j < 2:
                ax.set_xticks([])
            else:
                ax.set_xlabel("Time (ms)")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if i < 1:
                ax.set_ylabel(neuron_labels[j])
            
    
    fig.tight_layout()
    return fig, axes
    
