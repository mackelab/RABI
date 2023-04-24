import torch
from sbi.analysis import pairplot, marginal_plot
import matplotlib
from matplotlib.colors import to_rgb

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

import math
from scipy.stats import gaussian_kde


def add_adversarial_example_and_x_general(fig, x, x_tilde, **kwargs):
    d = int(math.sqrt(len(fig.axes)))
    ax = fig.add_subplot(d, 1, 1)
    ax.set_axis_off()
    ax.set_facecolor((0.1, 0.1, 0.1))

    ax.set_xticks([])
    ax.set_yticks([])

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    ax.plot(x, ".-", color=colors[0], alpha=0.5)
    ax.plot(x_tilde, ".-", color=colors[1], alpha=0.5)

    box = ax.get_position()
    reference = fig.axes[0].get_position()
    length = 1.15 * (reference.y1 - reference.y0)
    box.y0 = box.y0 + length
    box.y1 = box.y1 + length
    ax.set_position(box)


def add_adversarial_example_and_x_difference(
    fig, x, x_tilde, color_positive="tab:green", color_negative="tab:red", **kwargs
):
    d = int(math.sqrt(len(fig.axes)))
    ax = fig.add_subplot(d, 1, 1)

    diff = x - x_tilde
    colors = [color_positive, color_negative]

    ax.bar(range(len(diff)), diff, color=colors, alpha=0.8)

    ax.spines["right"].set_color("none")
    ax.spines["bottom"].set_position("zero")
    ax.spines["top"].set_color("none")
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    ax.set_xticks([])
    ax.set_xlim(-1, len(diff))
    ax.set_ylabel("Difference")

    box = ax.get_position()
    reference = fig.axes[0].get_position()
    length = 1.15 * (reference.y1 - reference.y0)
    box.y0 = box.y0 + length
    box.y1 = box.y1 + length
    ax.set_position(box)


def get_plotting_kwargs_pairplot(kwargs):
    if kwargs is None:
        kwargs = {}
    default = {
        "diag": "kde",
        "upper": "contour",
        "contour_offdiag": {"levels": [0.95], "precentile": True},
        "points_offdiag": {"markersize": 10},
        "kde_diag": {"bins": 30},
        "kde_offdiag": {"bins": 40},
        "figsize": (10, 10),
    }
    for key, val in kwargs.items():
        default[key] = val

    return default


def get_plotting_kwargs_marginal_plot(kwargs):
    if kwargs is None:
        kwargs = {}
    default = {"figsize": (10, 3), "diag": "hist"}
    for key, val in kwargs.items():
        default[key] = val

    return default


def custom_marginal_plot(
    samples, points=None, bins=100, box=False, labels=None, colors=None, limits=None, subset=None,figsize=(10, 2)
):
    if not isinstance(samples, list):
        samples = [samples]

    if not isinstance(points, list):
        points = [points]

    if subset is not None:
        for i in range(len(samples)):
            samples[i] = samples[i][..., subset]
        for i in range(len(points)):
            points[i] = points[i][..., subset]

    all_samples = torch.vstack(samples + [p for p in points if p is not None])

    mins = all_samples.min(0).values
    maxs = all_samples.max(0).values
    lengths = (maxs - mins).abs()

    d = all_samples.shape[-1]

    if limits is None:
        global_min = torch.round(
            mins - 1 / d * torch.minimum(lengths, torch.ones(1) - 0.1), decimals=1
        )
        global_max = torch.round(
            maxs + 1 / d * torch.minimum(lengths, torch.ones(1) + 0.1), decimals=1
        )
    else:
        global_min = torch.as_tensor(limits[0])
        global_max = torch.as_tensor(limits[1])

    fig, axes = plt.subplots(1, d, figsize=figsize)

    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i in range(d):
        if labels is not None:
            axes[i].set_xlabel(labels[i])
        else:
            axes[i].set_xlabel(f"dim {i}")
        for c, sample in enumerate(samples):

            x = torch.linspace(global_min[i], global_max[i], bins)
            color = colors[c]

            f = gaussian_kde(sample[..., i].numpy().T)
            y = f(x)
            axes[i].plot(x, y, color=color)
            axes[i].fill_between(x, torch.zeros_like(x), y, alpha=0.1, color=color)
            axes[i].set_ylim(0)
            axes[i].set_yticks([])
            axes[i].set_xlim([global_min[i], global_max[i]])
            axes[i].set_xticks([global_min[i], global_max[i]])
            axes[i].xaxis.set_ticks_position("bottom")

            if not box:
                axes[i].spines["top"].set_visible(False)
                axes[i].spines["right"].set_visible(False)
                axes[i].spines["left"].set_visible(False)

        for c, point in enumerate(points):
            color = colors[c]
            if point is not None:
                axes[i].vlines(
                    point[..., i], 0.0, axes[i].get_ylim()[-1], color=color, lw=3
                )
    fig.tight_layout()
    return fig, axes


def custom_2d_joint_plot(
    samples,
    dim1=0,
    dim2=1,
    points=None,
    bins=100,
    labels=None,
    figsize=(5, 5),
    colors=None,
    ratio=5,
):
    if not isinstance(samples, list):
        samples = [samples]

    if not isinstance(points, list):
        points = [points]


    all_samples = torch.vstack(samples + [p for p in points if p is not None])
    points = [p[..., [dim1, dim2]] if p is not None else None for p in points]
    samples = [s[..., [dim1, dim2]] for s in samples]
    all_samples = all_samples[..., [dim1, dim2]]


    mins = all_samples.quantile(0.01, dim=0)
    maxs = all_samples.quantile(0.99, dim=0)
    lengths = (maxs - mins).abs()

    d = all_samples.shape[-1]

    global_min = torch.round(
        mins - 1 / d * torch.minimum(lengths, torch.ones(1) - 0.1), decimals=1
    )
    global_max = torch.round(
        maxs + 1 / d * torch.minimum(lengths, torch.ones(1) + 0.1), decimals=1
    )

    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(ratio + 1, ratio + 1)

    ax_joint = fig.add_subplot(gs[1:, :-1])
    ax_marg_x = fig.add_subplot(gs[0, :-1], sharex=ax_joint)
    ax_marg_y = fig.add_subplot(gs[1:, -1], sharey=ax_joint)

    ax_joint.set_ylim(float(global_min[1]),float(global_max[1]))
    ax_joint.set_ylim(float(global_min[0]),float(global_max[0]))
    ax_marg_y.set_ylim(float(global_min[1]),float(global_max[1]))
    ax_marg_x.set_ylim(float(global_min[0]),float(global_max[0]))

    # Turn off tick visibility for the measure axis on the marginal plots
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)
    plt.setp(ax_marg_x.get_xticklabels(minor=True), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(minor=True), visible=False)
    plt.setp(ax_marg_x.get_yticklabels(), visible=False)
    plt.setp(ax_marg_y.get_xticklabels(), visible=False)

    if labels is None:
        ax_joint.set_xlabel(f"dim {0}")
        ax_joint.set_ylabel(f"dim {1}")
    else:
        ax_joint.set_xlabel(labels[0])
        ax_joint.set_ylabel(labels[1])



    x = torch.linspace(float(global_min[0]), float(global_max[0]), bins)
    y = torch.linspace(float(global_min[1]), float(global_max[1]), bins)
    xx, yy = torch.meshgrid(x, y)
    pos = torch.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])

    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for c, sample in enumerate(samples):

        p_x = gaussian_kde(sample[:, 0].numpy().T)
        p_y = gaussian_kde(sample[:, 1].numpy().T)
        p_xy = gaussian_kde(sample.numpy().T)

        f_x = p_x(x.numpy())
        f_y = p_y(y.numpy())
        f_xy = p_xy(pos.numpy().T).reshape(xx.shape)
        color = colors[c]
        ax_marg_x.plot(x, f_x, color=color)
        ax_marg_x.fill_between(x, torch.zeros_like(x), f_x, alpha=0.1, color=color)
        ax_marg_x.set_ylim(0, f_x.max() + 0.3)
        ax_marg_y.plot(f_y, y, color=color)
        ax_marg_y.fill_between(f_y, f_y, y, alpha=0.1, color=color)
        ax_marg_y.set_xlim(0, f_y.max() + 0.3)
    
        cmap = ListedColormap([to_rgb(color) + (i,) for i in torch.linspace(0, 1, 10)])
        ax_joint.contour(xx, yy, f_xy, levels=5, cmap=cmap)

    for c,point in enumerate(points):
        if point is not None:
            ax_joint.scatter(point[...,0], point[...,1], color=colors[c])
    


    #fig.tight_layout()
    return fig, [ax_joint, ax_marg_x, ax_marg_y]


def custom_pairplot(
    samples,
    points=None,
    bins=50,
    levels=5,
    box=False,
    labels=None,
    limits=None,
    subset=None,
    jitter=1e-5,
    figsize=(10, 10),
    colors=None,
):

    
    if not isinstance(samples, list):
        samples = [samples]

    if not isinstance(points, list):
        points = [points]

    if subset is not None:
        for i in range(len(samples)):
            samples[i] = samples[i][..., subset]
        for i in range(len(points)):
            points[i] = points[i][..., subset]

    all_samples = torch.vstack(samples + [p for p in points if p is not None])

    mins = all_samples.quantile(0.005, dim=0)
    maxs = all_samples.quantile(0.995, dim=0)
    lengths = (maxs - mins).abs()

    d = all_samples.shape[-1]

    # Bins based on dimension, lower for larger dimension
    bins += int(1 / d * 50)

    if limits is None:
        global_min = torch.round(
            mins - 1 / d * torch.minimum(lengths, torch.ones(1)) - 0.1, decimals=1
        )
        global_max = torch.round(
            maxs + 1 / d * torch.minimum(lengths, torch.ones(1)) + 0.1, decimals=1
        )
    else:
        global_min = torch.as_tensor(limits[0])
        global_max = torch.as_tensor(limits[1])


    fig, axes = plt.subplots(d, d, figsize=figsize)
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    

    for i in range(d):

        for j in range(d):
            if j < i:
                # Unused
                axes[i, j].set_axis_off()
            else:
                for c, sample in enumerate(samples):
                    sample = sample + jitter*(torch.rand_like(sample) - 0.5) # TO make kde non singular in extreme cases...
                    mins = sample.quantile(0.0001, dim=0)
                    maxs = sample.quantile(0.9999, dim=0)
                    lengths = (maxs - mins).abs()
                    min_ax_limit = torch.round(mins - 0.3 * lengths, decimals=1)
                    max_ax_limit = torch.round(maxs + 0.3 * lengths, decimals=1)
                    x = torch.linspace(min_ax_limit[i], max_ax_limit[i], bins)
                    color = colors[c]
                    axes[i, j].set_xlim(global_min[i], global_max[i])
                    axes[i, j].set_yticks([])
                    axes[i, j].set_xticks([])
                    cmap = ListedColormap(
                        [to_rgb(color) + (i,) for i in torch.linspace(0, 1, 10)]
                    )
                    if j > i:

                        y = torch.linspace(min_ax_limit[j], max_ax_limit[j], bins)
                        xx, yy = torch.meshgrid(x, y)
                        pos = torch.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
                        f = gaussian_kde(sample[..., [i, j]].numpy().T)
                        Z = f(pos.numpy().T).reshape(xx.shape)
                        axes[i, j].contour(yy, xx, Z, levels=levels, cmap=cmap)

                        axes[i, j].set_ylim(global_min[i], global_max[i])
                        axes[i, j].set_xlim(global_min[j], global_max[j])
                        axes[i, j].set_yticks([])
                        axes[i, j].set_xticks([])
                        if not box:
                            axes[i, j].spines["top"].set_visible(False)
                            axes[i, j].spines["right"].set_visible(False)
                            axes[i, j].spines["bottom"].set_visible(False)
                            axes[i, j].spines["left"].set_visible(False)
                    elif i == j:
                        x = torch.linspace(global_min[i], global_max[i], bins * 2)
                        if labels is not None:
                            axes[i,0].set_xlabel(labels[i])
                        f = gaussian_kde(sample[..., i].numpy().T)
                        y = f(x)
                        axes[i, j].plot(x, y, color=color)
                        axes[i, j].fill_between(x, torch.zeros_like(x), y, alpha=0.1, color=color)
                        axes[i, j].set_ylim(0)
                        axes[i, j].set_xticks([global_min[i], global_max[j]])
                        axes[i, j].xaxis.set_ticks_position("bottom")

                        if labels is None:
                            axes[i, j].set_xlabel(f"dim {i}")
                        else:
                            axes[i, j].set_xlabel(labels[i])

                        if not box:
                            axes[i, j].spines["top"].set_visible(False)
                            axes[i, j].spines["right"].set_visible(False)
                            axes[i, j].spines["left"].set_visible(False)
                    else:
                        pass

                for c, point in enumerate(points):
                    color = colors[c]
                    if point is not None:
                        if j > i:
                            axes[i, j].scatter(
                                point[..., j],
                                point[..., i],
                                color=color,
                                edgecolors="black",
                                s=10.0,
                            )
                        if j == i:
                            axes[i, j].vlines(
                                point[..., i],
                                0.0,
                                axes[i, j].get_ylim()[-1],
                                color=color,
                                lw=3,
                            )

    return fig, axes
