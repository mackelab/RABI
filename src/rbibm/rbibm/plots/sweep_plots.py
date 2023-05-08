import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch

from rbibm.utils.utils_data import get_sweep_dataset
import pandas as pd
import numpy as np


def plot_all_single_objectives(name: str, figsize=(7, 6)):

    df = get_sweep_dataset(name)

    values = df["best_value"].apply(pd.to_numeric, errors="coerce").dropna()
    idx = values.index

    df = df.loc[idx]

    search_space = df["search_space"].apply(lambda x: eval(x))

    search_spaces_strings = []
    for space in search_space:
        search_string = ""
        for s in space:
            param, space = s.split("=")
            search_string += param.split(".")[-1]
            search_string += "=" + space + "\n"
        search_spaces_strings.append(search_string)

    bar_labels = [
        f"Objective: {o}\nSweeper: {s}\nDirection:{d}"
        for o, s, d in zip(df.objective, df.sweeper, df.direction)
    ]

    num_bars = len(df)

    fig = plt.figure(figsize=(5+num_bars, 4))
    ax = plt.bar(search_spaces_strings, values, width=0.5)
    plt.bar_label(ax, labels=bar_labels)
    plt.tight_layout()
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig, ax


def plot_mulitobjective_paretto_front(name: str, idx: int, figsize=(7, 6), axes=None, alpha=1.0):

    df = get_sweep_dataset(name)



    df = df.loc[idx]

    search_space = [
        e.split("=")[0].split(".")[-1]
        for e in eval(df["search_space"])
        if "sweeper" not in e
    ]
    objectives = eval(df.objective)
    best_params = torch.tensor([list(a.values()) for a in eval(df.best_params)])
    values = torch.tensor(eval(df.best_value))

    num_objectives = len(objectives)
    num_params = best_params.shape[-1]

    print(f"Sweeper: {df.sweeper}")
    print(f"Overrides: {df.overrides}")

    if axes is None:
        fig, axes = plt.subplots(num_objectives, num_params, figsize=figsize)
    else:
        fig = None

    if num_objectives == 1 :
        axes = np.array([axes])

    if  num_params == 1:
        axes = np.array([[a,] for a in axes])
    

    for i in range(num_objectives):
        axes[i, 0].set_ylabel(objectives[i])
        for j in range(num_params):
            axes[-1, j].set_xlabel(search_space[j])
            axes[i, j].plot(best_params[:, j], values[:, i], "o", alpha=alpha)

            if j > 0:
                axes[i, j].set_yticklabels([])

            if i < num_objectives - 1:
                axes[i, j].set_xticklabels([])

    if fig is not None:
        fig.tight_layout()
    return fig, axes


def plot_biobjective_singleparameter(name: str, idx: int, figsize=(7, 6), axes=None, cmap=None, vmax=None, vmin=None, color_bar=True, color_label=None, alpha=0.9):

    df = get_sweep_dataset(name)



    df = df.loc[idx]

    search_space = [
        e.split("=")[0].split(".")[-1]
        for e in eval(df["search_space"])
        if "sweeper" not in e
    ]
    objectives = eval(df.objective)
    best_params = torch.tensor([list(a.values()) for a in eval(df.best_params)])
    values = torch.tensor(eval(df.best_value))

    num_objectives = len(objectives)
    assert num_objectives == 2, "Only supports 2"
    num_params = best_params.shape[-1]
    assert num_params == 1, "Only supports 1"

    print(f"Sweeper: {df.sweeper}")
    print(f"Overrides: {df.overrides}")

    if axes is None:
        fig = plt.figure(figsize=figsize)
        axes = plt.gca()
    else:
        fig = None

    axes.set_ylabel(objectives[1])
    axes.set_xlabel(objectives[0])
    cl = axes.scatter(values[:,0], values[:,1], c=best_params, cmap=cmap, norm=colors.LogNorm(vmin=vmin, vmax=vmax), alpha=alpha)
    if color_bar:
        cbar = plt.colorbar(cl, label = color_label)

    if fig is not None:
        fig.tight_layout()
    return fig, axes