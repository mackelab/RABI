import torch  # type: ignore
from torch import Tensor  # type: ignore
from sbi.analysis import pairplot, marginal_plot  # type: ignore
import matplotlib
from matplotlib.colors import to_rgb
from typing import Optional
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
    samples: list[Tensor], points: Optional[list[Tensor]]=None, bins: int =100, box: bool =False, labels: Optional[list[str]]=None, colors:Optional[list]=None, limits: Optional[list]=None, subset: Optional[list]=None,figsize: Optional[tuple] =None
):
    """Plots a distribution by plotting each 1d marginal distirbution based on iid samples from this distribtuions.

    Args:
        samples (list[Tensor]): Samples used to compute marginals, list of samples is treated as multiple distirbutions to plot
        points (Optional[list[Tensor]], optional): Points that are also plotted explicitly. Defaults to None.
        bins (int, optional): Bins to plot. Defaults to 100.
        box (bool, optional): If there should be a box. Defaults to False.
        labels (Optional[list[str]], optional): Axis labels. Defaults to None.
        colors (Optional[list], optional): Colors for different distributions. Defaults to None.
        limits (Optional[list], optional): Limits per dimension. Defaults to None.
        subset (Optional[list], optional): Subset of dimensions to plot. Defaults to None.
        figsize (Optional[tuple], optional): Size of the figure. Defaults to None.

    Returns:
        tuple: figure and axes
    """
    # Wrapp in list
    if not isinstance(samples, list):
        samples = [samples]

    if not isinstance(points, list):
        points = [points]

    # Choose subset by index
    if subset is not None:
        for i in range(len(samples)):
            samples[i] = samples[i][..., subset]

        for i in range(len(points)):
            p = points[i]
            if p is not None:
                points[i] = p[..., subset]

    # Get all samples
    all_samples = torch.vstack(samples + [p for p in points if p is not None])

    # Compute bounds which contain all of the points
    mins = all_samples.min(0).values
    maxs = all_samples.max(0).values
    lengths = (maxs - mins).abs()

    # Dimension
    d = all_samples.shape[-1]

    # Use limits based on global bounds if not given.
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

    # Figsize based on dimension
    if figsize is None:
        figsize = (d, 3)


    fig, axes = plt.subplots(1, d, figsize=figsize)

    # Get colors if not specified
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


    for i in range(d):

        # If label is not given use "dim i"
        if labels is not None:
            axes[i].set_xlabel(labels[i])
        else:
            axes[i].set_xlabel(f"dim {i}")

        # Plot all the samples in right colors
        for c, sample in enumerate(samples):

            x = torch.linspace(global_min[i], global_max[i], bins)
            color = colors[c]  # type: ignore

            # KDE for density estimate
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

        # Plot all points in right color
        for c, point in enumerate(points):
            color = colors[c]  # type: ignore
            if point is not None:
                axes[i].vlines(
                    point[..., i], 0.0, axes[i].get_ylim()[-1], color=color, lw=3
                )
    # Tight layout
    fig.tight_layout()

    return fig, axes


def custom_2d_joint_plot(
    samples: list[Tensor],
    dim1: int=0,
    dim2: int=1,
    points: Optional[Tensor]=None,
    bins: int =100,
    labels: Optional[tuple[str,str]]=None,
    figsize: tuple[int, int]=(5, 5),
    colors: Optional[list[str]]=None,
    ratio: int =5,
    levels: int = 5,
):
    """Plots the 2d and corresponding 1d marginals in one plot.

    Args:
        samples (list[Tensor]): Samples to be used to plot
        dim1 (int, optional): First dimension to use (x axis). Defaults to 0.
        dim2 (int, optional): Second dimension to use (y axis). Defaults to 1.
        points (Optional[Tensor], optional): Points to plot -> must be 2d . Defaults to None.
        bins (int, optional): Bins to use. Defaults to 100.
        labels (Optional[tuple[str,str]], optional): Labels to use. Defaults to None.
        figsize (tuple[int, int], optional): Figsize. Defaults to (5, 5).
        colors (Optional[list[str]], optional): Colors to use. Defaults to None.
        ratio (int, optional): How much larger the 2d marginal should be compared to the 1d marginals. Defaults to 5.
        levels (int, optional): How much contour lines are ploted.

    Returns:
        _type_: _description_
    """
    
    # Wrapps samples in list of not given as one
    if not isinstance(samples, list):
        samples = [samples]

    if not isinstance(points, list):
        points = [points]

    # Stack all samples together
    all_samples = torch.vstack(samples + [p for p in points if p is not None])

    # Select specified subset of dim1 and dim2 to obtain 2d samples
    points = [p[..., [dim1, dim2]] if p is not None else None for p in points]
    samples = [s[..., [dim1, dim2]] for s in samples]
    all_samples = all_samples[..., [dim1, dim2]]

    # Calculate global mins and maxs excluding outliers
    mins = all_samples.quantile(0.01, dim=0)
    maxs = all_samples.quantile(0.99, dim=0)
    lengths = (maxs - mins).abs()

    # Event shape
    d = all_samples.shape[-1]

    # Global min and max calculated
    global_min = torch.round(
        mins - 1 / d * torch.minimum(lengths, torch.ones(1) - 0.1), decimals=1
    )
    global_max = torch.round(
        maxs + 1 / d * torch.minimum(lengths, torch.ones(1) + 0.1), decimals=1
    )

    # Generate figure with custom grid spec -> small 1d marginal large 2d.
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(ratio + 1, ratio + 1)   # type: ignore

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

    # Label if specified other dim 0/1 -> x axis is determined by dim choosen
    if labels is None:
        ax_joint.set_xlabel(f"dim {dim1}")
        ax_joint.set_ylabel(f"dim {dim2}")
    else:
        ax_joint.set_xlabel(labels[0])
        ax_joint.set_ylabel(labels[1])


    # Get linespace and meshgrid
    x = torch.linspace(float(global_min[0]), float(global_max[0]), bins)
    y = torch.linspace(float(global_min[1]), float(global_max[1]), bins)
    xx, yy = torch.meshgrid(x, y)
    pos = torch.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])

    # Get colors if not provided
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Plot samples in correct color
    for c, sample in enumerate(samples):

        # Get all kdes
        p_x = gaussian_kde(sample[:, 0].numpy().T)
        p_y = gaussian_kde(sample[:, 1].numpy().T)
        p_xy = gaussian_kde(sample.numpy().T)

        # Get 1d marginals and 2d marginal densities
        f_x = p_x(x.numpy())
        f_y = p_y(y.numpy())
        f_xy = p_xy(pos.numpy().T).reshape(xx.shape)
        color = colors[c]   # type: ignore

        # Plot x axis 1d marginal
        ax_marg_x.plot(x, f_x, color=color)
        ax_marg_x.fill_between(x, torch.zeros_like(x), f_x, alpha=0.1, color=color)
        ax_marg_x.set_ylim(0, f_x.max() + 0.3)

        # Plot y axis 1d marginal -> Here x and y axis are interchanged
        ax_marg_y.plot(f_y, y, color=color)
        ax_marg_y.fill_between(f_y, f_y, y, alpha=0.1, color=color)
        ax_marg_y.set_xlim(0, f_y.max() + 0.3)
    
        # 2d marginal plot
        cmap = ListedColormap([to_rgb(color) + (i,) for i in torch.linspace(0, 1, 10)]) # type: ignore
        ax_joint.contour(xx, yy, f_xy, levels=levels, cmap=cmap)

    # Plot points into in right color...
    for c,point in enumerate(points):
        if point is not None:
            ax_joint.scatter(point[...,0], point[...,1], color=colors[c])    # type: ignore
    

    return fig, [ax_joint, ax_marg_x, ax_marg_y]


def custom_pairplot(
    samples: list[Tensor],
    points: Optional[list[Tensor]]=None,
    bins:int=50,
    levels:int=5,
    box: bool =False,
    labels: Optional[list[str]]=None,
    limits: Optional[list[float]]=None,
    subset: Optional[list[int]]=None,
    colors: Optional[list[str]]=None,
    jitter: float =1e-5,
    figsize: tuple[int,int]=(10, 10),
) -> tuple :
    """General pairplot function.

    Args:
        samples (list[Tensor]): Sampels to plot.
        points (Optional[list[Tensor]], optional): Points to plot, should be of same size then samples. Defaults to None.
        bins (int, optional): Bins to use -> The actual value is slighly larger than this for lower dimension and converges to this for large dim. Defaults to 50.
        levels (int, optional): Contour levels. Defaults to 5.
        box (bool, optional): Boxes around 2d or 1d marginals ?. Defaults to False.
        labels (Optional[list[str]], optional): Labels. Defaults to None.
        limits (Optional[list[float]], optional): Limits. Defaults to None.
        subset (Optional[list[int]], optional): Subsets of dimension. Defaults to None.
        colors (Optional[list[str]], optional): Colors. Defaults to None.
        jitter (float, optional): Jitter on samples to avoud singular KDE. Defaults to 1e-5.
        figsize (tuple[int,int], optional): Figsize. Defaults to (10, 10).

    Returns:
        tuple: Figure and axes.
    """

    
    # Wrapping as lsit if not
    if not isinstance(samples, list):
        samples = [samples]

    if not isinstance(points, list):
        points = [points]

    # Select subset of dimension
    if subset is not None:
        for i in range(len(samples)):
            samples[i] = samples[i][..., subset]
        for i in range(len(points)):
            p = points[i]
            if p is not None:
                points[i] = p[..., subset]

    all_samples = torch.vstack(samples + [p for p in points if p is not None])

    # Compute mins and max excluding outliers.
    mins = all_samples.quantile(0.005, dim=0)
    maxs = all_samples.quantile(0.995, dim=0)
    lengths = (maxs - mins).abs()

    # Event dimension
    d = all_samples.shape[-1]

    # Bins based on dimension, lower for larger dimension -> Plots also get smaller
    bins += int(1 / d * bins)

    # Use given limits or compute some.
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


    # Plot size  TODO Maybe make figsize default dependent on dimension...
    fig, axes = plt.subplots(d, d, figsize=figsize)


    # Get colors if not given.
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    

    for i in range(d):

        for j in range(d):

            # Disable lower corner...
            if j < i:
                # Unused
                axes[i, j].set_axis_off()
            else:
                # Plot all the samples
                for c, sample in enumerate(samples):
                    sample = sample + jitter*(torch.rand_like(sample) - 0.5) # TO make kde non singular in extreme cases...
                    
                    # Get bounds without outliers
                    mins = sample.quantile(0.0001, dim=0)
                    maxs = sample.quantile(0.9999, dim=0)
                    lengths = (maxs - mins).abs()
                    min_ax_limit = torch.round(mins - 0.3 * lengths, decimals=1)
                    max_ax_limit = torch.round(maxs + 0.3 * lengths, decimals=1)

                    # Get x axis
                    x = torch.linspace(min_ax_limit[i], max_ax_limit[i], bins)
                    color = colors[c] # type: ignore
                    axes[i, j].set_xlim(global_min[i], global_max[i])
                    axes[i, j].set_yticks([])
                    axes[i, j].set_xticks([])
                    cmap = ListedColormap(
                        [to_rgb(color) + (i,) for i in torch.linspace(0, 1, 10)]
                    ) # type: ignore

                    if j > i:
                        # This branch requires a 2d marginal plot between variable i and j
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
                        # Here we plot the 1d marginal -> Diagonal
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
                
                # Plotting the points
                for c, point in enumerate(points):
                    color = colors[c] # type: ignore
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
