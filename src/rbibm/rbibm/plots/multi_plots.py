from rbibm.utils.utils_data import query
import matplotlib.pyplot as plt
import numpy as np

import traceback

from rbibm.plots.metric_plots import (
    plot_by_num_simulations,
    plot_approximation_metric,
    plot_rob_tolerance_plot,
    get_plot_name_fn,
    rename_cols,
    use_all_plot_name_fn,
)
from rbibm.plots.metric_plots import get_plot_name_fn


def get_sorting_key_fn(name):
    if name == "task":

        def key_fn(task):
            vals = np.ones(len(task)) * 100
            mask_gl = task == "gaussian_linear"
            mask_sir = task == "sir"
            mask_lv = task == "lotka_volterra"
            mask_hh = task == "hudgkin_huxley"
            mask_ssir = task == "spatial_sir"
            mask_vae = task == "vae_task"
            vals[mask_gl] = 0.0
            vals[mask_sir] = 4.0
            vals[mask_lv] = 1.0
            vals[mask_hh] = 3.0
            vals[mask_ssir] = 5.0
            vals[mask_vae] = 2.0
            return vals

        return key_fn
    elif name == "model_name":

        def key_fn(task):
            vals = np.ones(len(task)) * 100
            mask_gl = task == "gaussian"
            mask_sir = task == "multivariate_gaussian"
            mask_lv = task == "mixture_gaussian"
            mask_hh = task == "maf"
            mask_ssir = task == "nsf"
            vals[mask_gl] = 0.0
            vals[mask_sir] = 1.0
            vals[mask_lv] = 2.0
            vals[mask_hh] = 3.0
            vals[mask_ssir] = 4.0

            return vals

        return key_fn

    elif name == "defense":

        def key_fn(task):
            vals = np.ones(len(task)) * 100
            mask_gl = task == "None"
            mask_noise = task == "L2UniformNoiseTraining"
            mask_adv= task == "L2PGDTargetedAdversarialTraining"
            mask_trades = task == "L2PGDrKLTrades"
            mask_fim = task == "FIMTraceRegularizer"
            vals[mask_gl] = "0"
            vals[mask_noise] = "1"
            vals[mask_adv] = "2"
            vals[mask_trades] = "3"
            vals[mask_fim] = "4"
            return vals

        return key_fn

    else:
        return lambda x: x


def multi_plot(
    name,
    cols,
    rows,
    plot_fn,
    fig_title=None,
    y_label_by_row=True,
    y_labels=None,
    scilimit=3,
    x_labels=None,
    y_lims=None,
    fontsize_title=None,
    figsize_per_row=2,
    figsize_per_col=2.3,
    legend_bbox_to_anchor=[0.5, -0.1],
    legend_title=False,
    legend_ncol=10,
    legend_kwargs={},
    fig_legend=True,
    **kwargs,
):
    df = query(name, **kwargs)

    df = df.sort_values(cols, na_position="first", key=get_sorting_key_fn(cols))
    cols_vals = df[cols].dropna().unique()

    df = df.sort_values(rows, na_position="first", key=get_sorting_key_fn(rows))
    rows_vals = df[rows].dropna().unique()

    # Creating a color map if hue is specified:
    if "hue" in kwargs and "color_map" not in kwargs:
        hue_col = kwargs["hue"]
        df = df.sort_values(
            hue_col, na_position="first", key=get_sorting_key_fn(hue_col)
        )
        unique_vals = df[hue_col].unique()
        unique_vals.sort()
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        color_map = {}
        for i in range(len(unique_vals)):
            color_map[unique_vals[i]] = colors[min(i, len(colors) - 1)]
    else:
        if "color_map" not in kwargs:
            color_map = None
        else:
            color_map = kwargs.pop("color_map")

    df.columns = [rename_cols(c) for c in df.columns]

    n_cols = len(cols_vals)
    n_rows = len(rows_vals)

    if n_cols == 0:
        raise ValueError(f"No columns found in the dataset with label {cols}")

    if n_rows == 0:
        raise ValueError(f"No rows found in the dataset with label {rows}")

    figsize = (n_cols * figsize_per_col, n_rows * figsize_per_row)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    else:
        if n_cols == 1:
            axes = np.array([[ax] for ax in axes])

        if n_rows == 1:
            axes = np.array([axes])

    max_legend_elements = 0

    for i in range(n_rows):
        for j in range(n_cols):

            axes[i, j].ticklabel_format(axis="y", scilimits=[-scilimit, scilimit])
            if y_labels is not None:
                y_label = y_labels[i]
            else:
                if y_label_by_row:
                    name_fn = get_plot_name_fn(rows)
                    y_label = name_fn(rows_vals[i])
                else:
                    y_label = None

            if x_labels is not None:
                x_label = x_labels[i]
            else:
                x_label = None

            if y_lims is not None:
                if isinstance(y_lims, tuple):
                    y_lim = y_lims
                else:
                    if isinstance(y_lims[0], tuple):
                        y_lim = y_lims[i]
                    else:
                        if isinstance(y_lims[0, 0], tuple):
                            y_lim = y_lims[i, j]
                        else:
                            raise ValueError()
            else:
                y_lim = None

            plot_dict = {cols: cols_vals[j], rows: rows_vals[i]}
            plot_kwargs = {**kwargs, **plot_dict}
            print(plot_kwargs)
            # plot_fn(name, ax=axes[i,j], **plot_kwargs)
            try:
                plot_fn(name, ax=axes[i, j], color_map=color_map, **plot_kwargs)
            except Exception as e:
                print(str(e))

            if y_label is not None:
                axes[i, j].set_ylabel(y_label)
                axes[i, j].yaxis.set_label_coords(-0.3, 0.5)
            else:
                fn = get_plot_name_fn(cols)
                y_label = axes[i, j].get_ylabel()
                axes[i, j].set_ylabel(fn(y_label))
                axes[i, j].yaxis.set_label_coords(-0.3, 0.5)

            if x_label is not None:
                axes[i, j].set_xlabel(x_label)
            else:
                fn = get_plot_name_fn(rows)
                x_label = axes[i, j].get_xlabel()
                axes[i, j].set_xlabel(fn(x_label))
            if i == 0:
                name_fn = get_plot_name_fn(cols)
                axes[i, j].set_title(name_fn(cols_vals[j]))

            if i < n_rows - 1:
                axes[i, j].set_xlabel(None)
                axes[i, j].set_xticklabels([])

            if j > 0:
                axes[i, j].set_ylabel(None)

            if y_lim is not None:
                axes[i, j].set_ylim(y_lim)

            if i > 0:
                axes[i, j].set_title(None)

            if axes[i, j].get_legend() is not None:
                legend = axes[i, j].get_legend()
                if len(legend.get_texts()) > max_legend_elements:
                    max_legend_elements = len(legend.get_texts())
                    legend_text = [t._text for t in legend.get_texts()]
                    if legend_title:
                        legend_title = legend.get_title()._text
                    else:
                        legend_title = ""
                    legend_handles = legend.legendHandles
                legend.remove()

    for i in range(n_rows):
        for j in range(n_cols):
            if len(axes[i, j].lines) == 0 and len(axes[i, j].collections) == 0:
                axes[i, j].text(
                    0.5,
                    0.5,
                    "No data",
                    bbox={
                        "facecolor": "white",
                        "alpha": 1,
                        "edgecolor": "none",
                        "pad": 1,
                    },
                    ha="center",
                    va="center",
                )

    if fig_legend and "legend_text" in locals() and len(legend_text) > 0:
        text = [use_all_plot_name_fn(t) for t in list(dict.fromkeys(legend_text))]
        handles = list(dict.fromkeys(legend_handles))
        fig.legend(
            labels=text,
            handles=handles,
            title=use_all_plot_name_fn(str(legend_title)),
            ncol=legend_ncol,
            loc="lower center",
            bbox_to_anchor=legend_bbox_to_anchor,
            **legend_kwargs,
        )

    fig.tight_layout()
    if fig_title is not None:
        fig.suptitle(fig_title)
    return fig, axes