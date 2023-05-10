import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, ListedColormap
import matplotlib as mpl
import seaborn as sns
import math
import random

from typing import Optional, Tuple, Callable

import pandas as pd
import numpy as np

from torch import Value

from rbibm.utils.utils_data import query


def get_ylims_for_metric(name: Optional[str], df=None):
    if name == "C2STBayesOptimal2GroundTruthMetric":
        return (0.5, 1.0)
    elif name == "ReverseKL2GroundTruthMetric":
        if df is not None:
            ylim = (
                df.groupby(["N_train", "eps"])["main_value_approx_tilde"]
                .quantile(0.8)
                .quantile(0.8)
            )
        else:
            ylim = None
        return (0.0, ylim)
    elif name == "ForwardKL2GroundTruthMetric":
        if df is not None:
            ylim = (
                df.groupby(["N_train", "eps"])["main_value_approx_tilde"]
                .quantile(0.8)
                .quantile(0.8)
            )
        else:
            ylim = None
        return (0.0, ylim)
    elif (
        name == "ReverseKLRobMetric"
        or name == "ForwardKLRobMetric"
        or name == "metric_rob"
    ):
        if df is not None:
            ylim = (
                df.groupby(["N_train", "eps"])["main_value_rob"]
                .quantile(0.9)
                .quantile(0.8)
            )
        else:
            ylim = None
        return (0.0, ylim)
    elif name == "MedianL2DistanceToObsMetric":
        if df is not None:
            ylim = (
                df.groupby(["N_train", "eps"])["main_value_approx_tilde"]
                .quantile(0.9)
                .quantile(0.8)
            )
        else:
            ylim = None
        return (0.0, ylim)

    elif name == "NegativeLogLikelihoodMetric":
        return (None, None)
    elif name == "ExpectedCoverageMetric":
        return (0.0, 0.5)
    elif name == "R2LinearFit2Potential":
        return (0.0, 1.0)
    else:
        return (None, None)


def get_plot_name_fn(name: Optional[str]):
    """Better naming for labels ..."""
    if name == "task":
        return task_plot_name
    if name == "defense":
        return defense_to_plot_name
    elif name == "loss":
        return loss_to_plot_name
    elif name == "attack":
        return attack_to_plot_name
    elif name == "metric_rob":
        return metric_rob_name_to_plot_name
    elif name == "model_name":
        return model_name_to_plot_name
    elif isinstance(name, str) and "metric_approx" in name:
        return approx_metric_name_to_plot_name
    else:
        return lambda x: x


def use_all_plot_name_fn(name: Optional[str]):
    if name == "eps":
        return "Tolerance $\\epsilon$"
    else:
        return model_name_to_plot_name(
            task_plot_name(
                loss_to_plot_name(
                    attack_to_plot_name(
                        metric_rob_name_to_plot_name(approx_metric_name_to_plot_name(name))
                    )
                )
            )
        )


def rename_cols(name: Optional[str]):
    if name == "task":
        return "Task"
    elif name == "loss":
        return "Loss"
    elif name == "test_loss":
        return "Test loss"
    elif name == "N_train":
        return "Training datapoints"
    elif name == "N_test":
        return "Testing datapoints"
    elif name == "N_val":
        return "Validation datapoints"
    elif name == "train_loss":
        return "Train loss"
    elif name == "val_loss":
        return "Validation loss"
    elif name == "train_time":
        return "Runtime (train)"
    elif name == "model_name":
        return "Model"
    elif name == "defense":
        return "Defense"
    elif name == "model_name":
        return "Model"
    elif name == "metric_rob":
        return "Robustness metric"
    elif name == "metric_approx_clean":
        return "Approximation metric"
    elif name == "metric_approx_tilde":
        return "Approximation metric "
    else:
        return name


def float_to_power_of_ten(val: float):
    exp = math.log10(val)
    exp = int(exp)
    return rf"$10^{exp}$"


def defense_to_plot_name(defense: Optional[str]):
    if defense == math.nan or defense == "nan":
        return "None"
    elif defense == "FIMTraceRegularizer":
        return "FIM regularizer"
    elif defense == "L2PGDrKLTrades":
        return "Trades"
    elif defense == "L2PGDTargetedAdversarialTraining":
        return "Adv. training"
    else:
        return defense


def task_plot_name(task: Optional[str]):
    if task == "gaussian_linear":
        return "Linear Gaussian"
    elif task == "rbf_regression":
        return "GLR"
    elif task == "lotka_volterra":
        return "Lotka Volterra"
    elif task == "sir":
        return "SIR"
    elif task == "vae_task":
        return "VAE"
    elif task == "hudgkin_huxley":
        return "Hodgkin Huxley"
    elif task == "rps_task":
        return "Rock paper sicior"
    elif task == "spatial_sir":
        return "Spatial SIR"
    else:
        return task


def loss_to_plot_name(loss: Optional[str]):
    if loss == "NLLLoss":
        return "Negative loglikelihood"
    elif loss == "NegativeElboLoss":
        return "Negative ELBO"
    elif loss is None:
        return ""
    else:
        return loss


def attack_to_plot_name(attack: Optional[str]):
    if attack == "L2PGDAttack":
        return "L2PGD"
    elif attack == "LinfPGDAdamAttack":
        return "LinfPGD"
    elif attack == "L2UniformNoiseAttack":
        return "L2Noise"
    elif attack == "LinfUniformNoiseAttack":
        return "LinfNoise"
    else:
        return attack


def metric_rob_name_to_plot_name(metric: Optional[str]):
    if metric == "ReverseKLRobMetric":
        return r"Reverse KL divergence"
    elif metric == "ForwardKLRobMetric":
        return r"Forward KL divergence"
    elif metric is None:
        return "Robustness metric"
    else:
        return metric


def approx_metric_name_to_plot_name(metric):
    if metric == "MedianL2DistanceToObsMetric":
        return r"Median $||x - x_o||_2$"
    elif metric == "MedianL1DistanceToObsMetric":
        return r"Median $||x - x_o||_1$"
    elif metric == "MedianLinfDistanceToObsMetric":
        return r"Median $||x - x_o||_\infty$"
    elif metric == "SimulationBasedCalibrationMetric":
        return "Simulation-based calibration"
    elif metric == "NegativeLogLikelihoodMetric":
        return "Negative loglikelihood"
    elif metric == "ExpectedCoverageMetric":
        return "Expected coverage"
    elif metric == "C2STBayesOptimal2GroundTruthMetric":
        return "C2ST"
    elif metric == "VarianceLogPotential2LogQ":
        return r"$Var( \log p(\theta|x) - \log q(\theta|x))$"
    elif metric == "R2LinearFit2Potential":
        return r"$R^2$ to potential"
    elif metric is None:
        return ""
    else:
        return metric


def model_name_to_plot_name(model_name):
    if model_name == "gaussian":
        return "Diagonal Gaussian"
    elif model_name == "multivariate_gaussian":
        return "Multivariate Gaussian"
    elif model_name == "mixture_gaussian":
        return "Mixture of Gaussian's"
    elif model_name == "maf" or model_name == "iaf":
        return "MAF"
    elif model_name == "nsf":
        return "NSF"
    elif model_name is None:
        return ""
    else:
        return model_name


def order_defense(key: str):
    key = key.replace("None", "0")
    return key


def model_order(key: str):
    key = key.replace("Gaussian Diagonal", "0")
    key = key.replace("Multivariate Gaussian", "1")
    key = key.replace("Mixture of Gaussian's", "2")
    key = key.replace("MAF", "3")
    key = key.replace("NSF", "4")
    key = key.replace("gaussian", "0")
    key = key.replace("multivariate_gaussian", "1")
    key = key.replace("mixture_gaussian", "2")
    key = key.replace("maf", "3")
    key = key.replace("nsf", "4")
    return key


def plot_by_num_simulations(
    name: str,
    y: Optional[str],
    y_label: Optional[str] = None,
    y_lim: Optional[Tuple] = None,
    yscale: str = "linear",
    hue: Optional[str] = None,
    hue_order_fn: Optional[Callable] = None,
    figsize: Optional[Tuple] = None,
    ax: Optional[plt.Axes] = None,
    verbose: bool = False,
    color_map: Optional[dict] = None,
    alpha=1.,
    **kwargs,
):
    if ax is None:

        if figsize is None:
            fig = None
        else:
            fig = plt.figure(figsize=figsize)

        ax = plt.gca()
    else:
        fig = None

    df_s = query(name, **kwargs)
    if verbose:
        print("Following data is used:")
        print(f"Id: {df_s.id.unique()}, Col: {df_s[y].unique()}")
        if hue is not None:
            print(f"Hue: {df_s[hue].unique()}")

    if len(df_s) == 0:
        raise ValueError("No data for these keywords....")

    # Sort models
    df_s = df_s.sort_values("model_name", key=model_order)

    # Sorted by N_train ...
    df_s = df_s.sort_values("N_train", kind="stable")
    df_s = df_s.dropna(axis=0, subset=[y])
    df_s = df_s.drop_duplicates(subset=[y])

    if hue is not None:
        if hue_order_fn is None:
            df_s = df_s.sort_values(hue)
        else:
            df_s = df_s.sort_values(hue, key=hue_order_fn)

    df_s.columns = [rename_cols(c) for c in df_s.columns]
    y = rename_cols(y)


    if hue is not None:
        plot_name_fn = get_plot_name_fn(hue)
        df_s[rename_cols(hue)] = df_s[rename_cols(hue)].apply(plot_name_fn)
        g = sns.pointplot(
            x=rename_cols("N_train"),
            y=y,
            data=df_s,
            ax=ax,
            hue=rename_cols(hue),
            capsize=0.1,
            scale=1.5,
            markers=".",
            palette=color_map,
            dodge=False,
        )
    else:
        g = sns.pointplot(
            x=rename_cols("N_train"),
            y=y,
            data=df_s,
            ax=ax,
            capsize=0.075,
            scale=1.5,
            palette=color_map,
            markers=".",
            dodge=False,
        )
    plt.setp(g.collections, alpha=alpha) #for the markers
    plt.setp(g.lines, alpha=alpha)       
    #ax.grid(axis="x")

    ax.set_xlabel("Number of simulations")
    if y_label is not None:
        ax.set_ylabel(y_label)
    else:
        ax.set_ylabel(y)

    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.set(yscale=yscale)
    try:
        ax.set_xticklabels(
            [float_to_power_of_ten(float(a._text)) for a in ax.get_xticklabels()]
        )
    except:
        pass
    return fig, ax


def plot_rob_tolerance_plot(
    name: str,
    task: Optional[str] = None,
    model_name: Optional[str] = None,
    metric_rob: Optional[str] = None,
    attack: Optional[str] = None,
    attack_loss_fn: Optional[str] = None,
    N_train: Optional[str] = None,
    defense: Optional[str] = None,
    loss: Optional[str] = None,
    ax=None,
    ylim=None,
    hue: Optional[str] = None,
    hue_order_fn: Optional[Callable] = None,
    uncertainty_lower: str = "q15",
    uncertainty_upper: str = "q85",
    main_value_adjust: Optional[str] = "q50",
    yscale: str = "log",
    color_map: Optional[dict] = None,
    markersize: float = 2.5,
    min_clip=1e-7,
    jitter=False,
    alpha=1.,
    adjust_df=None,
    **kwargs,
):
    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
    else:
        fig = None

    df_s = query(
        name,
        task=task,
        model_name=model_name,
        N_train=N_train,
        metric_rob=metric_rob,
        attack=attack,
        attack_loss_fn=attack_loss_fn,
        loss=loss,
        defense=defense,
        **kwargs,
    )
    if adjust_df is not None:
        df_s = adjust_df(df_s)

    if len(df_s) == 0:
        raise ValueError("No data for these keywords....")

    # Sort models
    df_s = df_s.sort_values("model_name", key=model_order)

    df_s = df_s.sort_values("eps", kind="stable")
    df_s = df_s.fillna("None")
    df_s = df_s.drop_duplicates(subset=["main_value_rob", "additional_value_rob"])

    if hue is not None:
        if hue_order_fn is None:
            df_s = df_s.sort_values(hue)
        else:
            df_s = df_s.sort_values(hue, key=hue_order_fn)

    try:

        quantile05 = df_s.additional_value_rob.apply(
            lambda x: eval(x)[uncertainty_lower]
        )
        quantile95 = df_s.additional_value_rob.apply(
            lambda x: eval(x)[uncertainty_upper]
        )

        if main_value_adjust is None:
            main = df_s.main_value_rob
        else:
            main = df_s.additional_value_rob.apply(lambda x: eval(x)[main_value_adjust])

        df_s["q05"] = quantile05
        df_s["q95"] = quantile95
        df_s["main"] = main
    except:
        pass

    epsilon = df_s.eps.unique()
    epsilon.sort()
    if hue is not None:
        df_grouped = df_s.groupby([hue, "eps"]).mean()
        y_hue = df_s.sort_values(hue, key=model_order)[hue].unique()

    else:
        df_grouped = df_s.groupby(["eps"]).mean()
        y_hue = [None]

    lw = mpl.rcParams["lines.linewidth"] * 1.8
    markersize = math.pi * markersize


    for i, y in enumerate(y_hue):
        if hue is not None:
            vals = df_grouped.loc[y]
        else:
            vals = df_grouped
        if color_map is not None:
            color = color_map[y]
        else:
            color = None

        quantile05 = np.clip(vals.q05, min_clip, np.inf)
        quantile95 = np.clip(vals.q95, min_clip, np.inf)
        mean = np.clip(vals.main, min_clip, np.inf)

        if uncertainty_lower != "std":
            q05 = mean - quantile05
        else:
            q05 = quantile05

        if uncertainty_upper != "std":
            q95 = quantile95 - mean
        else:
            q95 = quantile95

        if not jitter:
            plot_at = [list(epsilon).index(e) for e in vals.index]
        else:
            plot_at = [list(epsilon).index(e) + random.random()*0.25 - 0.125 for e in vals.index]

        markers, caps, bar = ax.errorbar(
            plot_at,
            mean,
            (q05, q95),
            fmt="o-",
            label=y,
            capsize=3.5,
            capthick=3.0,
            alpha=alpha,
            linewidth=lw,
            markersize=markersize,
            color=color,
        )

        caps[0].set_alpha(0.8*alpha)
        caps[1].set_alpha(0.8*alpha)
        bar[0].set_alpha(0.8*alpha)

        # get handles
    handles, labels = ax.get_legend_handles_labels()
    # remove the errorbars
    handles = [h[0] for h in handles]
    plot_name_fn = get_plot_name_fn(hue)
    ax.legend(
        handles,
        [plot_name_fn(l) for l in labels],
        title=rf"{rename_cols(hue)}",
        numpoints=1,
        handlelength=0,
    )

    ax.set_xlabel("Tolerance $\\epsilon$")
    ax.set_ylabel(metric_rob_name_to_plot_name(metric_rob))
    ax.set_xlim(-0.5, len(epsilon) - 0.5)
    ax.set_xticks(range(len(epsilon)))
    ax.set_xticklabels(epsilon)
    ax.set_yscale(yscale)
    if ylim is not None:
        ax.set_ylim(ylim)

    return fig, ax


def plot_approximation_metric(
    name: str,
    metric_approx_clean: str,
    target: str = "x_tilde",
    task: Optional[str] = None,
    model_name: Optional[str] = None,
    defense: Optional[str] = None,
    loss: Optional[str] = None,
    color_start: str = "black",
    color_end: str = "C3",
    uncertainty_lower: str = "q05",
    uncertainty_upper: str = "q95",
    ax=None,
    ylim=None,
    yscale="linear",
    color_map: Optional[dict] = None,
    **kwargs,
):
    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
    else:
        fig = None

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

    df_s = df_s.sort_values(["N_train", "eps"])
    df_s = df_s.fillna("None")
    df_s = df_s.drop_duplicates(
        subset=[
            "main_value_approx_tilde",
            "additional_value_approx_tilde",
            "main_value_approx_clean",
            "additional_value_approx_clean",
            "main_value_tilde_to_x",
            "additional_value_tilde_to_x",
        ]
    )

    if target == "x_tilde":
        suffix = "_approx_tilde"
    elif target == "x":
        suffix = "_tilde_to_x"
    else:
        raise ValueError()

    id_main_value_tilde = "main_value" + suffix
    id_additional_value_tilde = "additional_value" + suffix

    # Extract quantiles
    try:
        quantile05_tilde = df_s[id_additional_value_tilde].apply(
            lambda x: eval(x)[uncertainty_lower]
        )
        quantile95_tilde = df_s[id_additional_value_tilde].apply(
            lambda x: eval(x)[uncertainty_upper]
        )

        df_s["q05_tilde"] = quantile05_tilde
        df_s["q95_tilde"] = quantile95_tilde
    except:
        pass

    df_clean = df_s[
        ["id", "N_train", "main_value_approx_clean", "additional_value_approx_clean"]
    ].drop_duplicates()

    try:
        quantile05 = df_clean.additional_value_approx_clean.apply(
            lambda x: eval(x)[uncertainty_lower]
        )
        quantile95 = df_clean.additional_value_approx_clean.apply(
            lambda x: eval(x)[uncertainty_upper]
        )

        df_clean["q05"] = quantile05
        df_clean["q95"] = quantile95
    except:
        pass

    epsilon = df_s.eps.unique()
    epsilon.sort()
    N_trains = df_s.N_train.unique()
    N_trains.sort()
    df_grouped_clean = df_clean.groupby("N_train").mean()
    df_grouped = df_s.groupby(["eps", "N_train"]).mean()

    num_eps = len(epsilon)

    start = np.clip(np.array(to_rgb(color_start)), 0, 1)
    end = np.array(to_rgb(color_end))

    colors = [start + i * (end - start) for i in np.linspace(0.3, 1, num_eps)]
    df_s.columns = [rename_cols(c) for c in df_s.columns]

    lw = mpl.rcParams["lines.linewidth"] * 1.8
    markersize = math.pi * 2.5

    for i, e in enumerate(epsilon):
        if i == 0:
            mean = df_grouped_clean["main_value_approx_clean"]
            if "q05" in df_grouped_clean and "q95" in df_grouped_clean:
                if uncertainty_lower != "std":
                    q05 = mean - df_grouped_clean["q05"]
                    q95 = df_grouped_clean["q95"] - mean
                else:
                    q05 = df_grouped_clean["q05"]
                    q95 = df_grouped_clean["q95"]
                error = (q05, q95)
            else:
                error = None

            markers, caps, bar = ax.errorbar(
                range(len(N_trains)),
                mean,
                error,
                fmt="o-",
                label="0.0",
                capsize=3.5,
                capthick=3.0,
                linewidth=lw,
                color=color_start,
                markersize=markersize,
                alpha=0.8,
            )
        vals = df_grouped.loc[e]
        mean_tilde = vals[id_main_value_tilde]
        if "q95_tilde" in df_grouped and "q05_tilde" in df_grouped:
            if uncertainty_lower != "std":
                q05_tilde = mean_tilde - vals["q05_tilde"]
                q95_tilde = vals["q95_tilde"] - mean_tilde
            else:
                q05_tilde = vals["q05_tilde"]
                q95_tilde = vals["q95_tilde"]
            error_tilde = (q05_tilde, q95_tilde)
        else:
            error_tilde = None

        markers, caps, bar = ax.errorbar(
            [list(N_trains).index(i) for i in vals.index],
            mean_tilde,
            error_tilde,
            fmt="o-",
            label=str(e),
            capsize=3.5,
            capthick=3.0,
            linewidth=lw,
            color=colors[i],
            markersize=markersize,
            alpha=0.8,
        )

        # get handles
    handles, labels = ax.get_legend_handles_labels()

    # remove the errorbars
    handle = [h[0] for h in handles]

    ax.legend(
        handles=handle, labels=labels, title=r"$\epsilon$", numpoints=1, handlelength=0
    )

    # sns.pointplot(
    #     x="N_train",
    #     y="main_value_approx_clean",
    #     data=df_s,
    #     ax=ax,
    #     color=color_start,
    #     capsize=0.05,
    # )
    # sns.pointplot(
    #     x="N_train",
    #     y="main_value_approx_tilde",
    #     data=df_s,
    #     ax=ax,
    #     palette=colors,
    #     hue=rename_cols("eps"),
    #     capsize=0.05,
    # )

    ax.set_xlabel("Number of simulations")
    ax.set_ylabel(approx_metric_name_to_plot_name(metric_approx_clean))
    ax.set_yscale(yscale)

    # model = model_name_to_plot_name(model_name)
    # title = task_plot_name(task)
    # if model is not None or model == "":
    #     title += f"(Model: {model})"
    # ax.set_title(title)
    ax.set_xlim(-0.5, len(N_trains) - 0.5)
    ax.set_xticks(range(len(N_trains)))
    ax.set_xticklabels([float_to_power_of_ten(float(a)) for a in N_trains])
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(get_ylims_for_metric(metric_approx_clean, df_s))
    # leg = ax.get_legend()
    # leg.set_title(r"$\epsilon$")

    return fig, ax

def plot_something(x_value, y_value, labels, y_label=None, y_error=None, colors=None, markersize=2.5, ax=None, alpha=1., title=None, legend=False):

    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
    else:
        fig = None

    lw = mpl.rcParams["lines.linewidth"] * 1.8
    markersize = math.pi * markersize

    plot_at = [x_value.index(e) for e in x_value]
    for i, y in enumerate(y_value):
        if colors is not None:
            color = colors[i]
        else:
            color = None
        if y_error is None:
            ax.plot(plot_at, y, "o-", alpha=alpha, linewidth=lw, markersize=markersize, color=color, label=labels[i])
        else:
            q05, q95 = y_error[i]
            markers, caps, bar = ax.errorbar(
            plot_at,
            y,
            (q05, q95),
            fmt="o-",
            label=labels[i],
            capsize=3.5,
            capthick=3.0,
            alpha=alpha,
            linewidth=lw,
            markersize=markersize,
            color=color,
        )

            caps[0].set_alpha(0.8*alpha)
            caps[1].set_alpha(0.8*alpha)
            bar[0].set_alpha(0.8*alpha)
        

    # get handles
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if y_error is not None:
            handles = [h[0] for h in handles]
        # remove the errorbars
        ax.legend(
            handles,
            labels,
            title=rf"{title}",
            numpoints=1,
            handlelength=0,
        )

    ax.set_xlabel("Tolerance $\\epsilon$")
    ax.set_ylabel(y_label)
    ax.set_xlim(-0.5, len(x_value) - 0.5)
    ax.set_xticks(range(len(x_value)))
    ax.set_xticklabels(x_value)


    return fig, ax


def plot_expected_coverage(
    name: str,
    task: Optional[str] = None,
    model_name: Optional[str] = None,
    defense: Optional[str] = None,
    loss: Optional[str] = None,
    hue: Optional[str] = None,
    with_eps: bool = True,
    verbose: bool = True,
    color_map=None,
    with_grid = True,
    ax=None,
    **kwargs,
):
    metric_approx_clean = "ExpectedCoverageMetric"
    if ax is None:
        fig = plt.figure(figsize=(3, 2.8))
        ax = plt.gca()
    else:
        fig = None

    lw = mpl.rcParams["lines.linewidth"] 

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

    # df_s = df_s.fillna("None")

    if hue is not None:
        df_s = df_s.sort_values(hue)
    else:
        df_s = df_s[:1]

    # df_s = df_s.drop_duplicates(subset= ["additional_value_approx_clean", "additional_value_approx_tilde"])

    if hue is not None:
        df_s = df_s.groupby(hue).first()

    if verbose:
        print("Following data row is used:")
        print(df_s.index)
        print(df_s.id.unique())

    alphas = df_s.additional_value_approx_clean.apply(lambda x: eval(x)["alphas"])
    quantiles_clean = df_s.additional_value_approx_clean.apply(
        lambda x: eval(x)["quantiles"]
    )

    quantiles_tilde = df_s.additional_value_approx_tilde.apply(
        lambda x: eval(x)["quantiles"]
    )

    # Diagonal
    ax.plot(
        alphas.iloc[0],
        alphas.iloc[0],
        color=(0.1, 0.1, 0.1),  # grey
        linestyle="dashdot",
        lw= 0.5*lw,
        alpha=0.8,
    )

    lines = []

    (l,) = ax.plot(
        alphas.iloc[0], quantiles_clean.iloc[0], lw=lw, label="0.0", color=f"black"
    )
    lines += [l]

    if with_eps:
        epsilon = df_s.index.to_list()
        for i, N in enumerate(alphas.index):
            if color_map is None:
                color = f"C{i}"
            else:
                color = color_map[epsilon[i]]
            (l,) = ax.plot(
                alphas[N],
                quantiles_tilde[N],
                lw=lw,
                color=color,
                alpha=0.5,
            )
            lines.append(l)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    if with_grid:
        ax.grid(which="minor", alpha=0.4, linestyle="dotted")
        ax.grid(which="major", alpha=0.6)
    ax.set_xlabel("Expected coverage")
    ax.set_ylabel("Observed coverage")

    ax.legend(handles=lines, labels=[0] + df_s.index.to_list(), title=rename_cols(hue))

    # if with_eps:
    #     leg = ax.get_legend()
    #     leg.set_title(r"$\epsilon$")

    return fig, ax
