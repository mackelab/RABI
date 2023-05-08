import matplotlib.pyplot as plt
from rbibm.utils.utils_data import query
from rbibm.plots.utils import (
    maybe_get_model_by_id,
    maybe_get_x_tilde_from_id,
    maybe_get_task_by_id,
)
from rbibm.plots.predictives_per_task import get_predicitve_plotting_function
from rbibm.plots.custom_distribution_plots import *

import torch  # type: ignore
from sbi.analysis import pairplot, marginal_plot  # type: ignore
from matplotlib.colors import to_hex



from rbibm.utils.utils_data import load_posterior_samples_by_id

from typing import Optional, Any


def set_normal_adversarial_colors(color_normal, color_adversarial):
    plt.rcParams["axes.prop_cycle"]._left[0] = {"color": to_hex(color_normal)}
    plt.rcParams["axes.prop_cycle"]._left[1] = {"color": to_hex(color_adversarial)}
    plt.rcParams["axes.prop_cycle"]._left[2] = {"color": to_hex("black")}

def plot_posterior(
    name: str,
    task: Optional[str]=None,
    model_name: Optional[str]=None,
    metric_approx_clean: Optional[str]=None,
    metric_approx_tilde: Optional[str]=None,
    defense: Optional[str]=None,
    loss: Optional[str]=None,
    type:str="pairplot",
    verbose: bool=True,
    model: Optional[Any]=None,
    n_samples: int =15000,
    x_o: Optional[Tensor]=None,
    theta_o: Optional[Tensor]=None,
    device: str="cpu",
    idx_adv_example: Optional[int] = None,
    plot_true: Optional[str] = None,
    plotting_kwargs: dict ={},
    **kwargs,
):
    """ This function plots the approximate posterior learned for a certain obervation x_o. 

    Args:
        name (str): Dataset name
        task (Optional[str], optional): Task solved by model. Defaults to None.
        model_name (Optional[str], optional): Type of the model . Defaults to None.
        metric_approx_clean (Optional[str], optional): Approximation metric name . Defaults to None.
        metric_approx_tilde (Optional[str], optional): Approximaiton metric for adversarial examples. Defaults to None.
        defense (Optional[str], optional): Defense applied on the model . Defaults to None.
        loss (Optional[str], optional): Loss function used to train. Defaults to None.
        type (str, optional): Type of the plot i.e. "pariplot", "marginal_plot", "pairplot_sbi"... . Defaults to "pairplot".
        verbose (bool, optional): If to plot which data is used to obtain the plot . Defaults to True.
        model (Optional[Any], optional): This is fetched from the dataset if not given . Defaults to None.
        n_samples (int, optional): Number of samples to use to create the plot. Defaults to 15000.
        x_o (Optional[Tensor], optional): Observation to condition on. Defaults to None.
        theta_o (Optional[Tensor], optional): Parameter which created the observation. Defaults to None.
        device (str, optional): Compute device . Defaults to "cpu".
        plotting_kwargs (dict, optional): Plots passed to the plotting function. Defaults to {}.
        kwargs: Other keywords for the search in the dataset...

    Raises:
        ValueError: No data found matching the keywords
        ValueError: No common task and no x_o provided...
        ValueError: _description_

    Returns:
        tuple: fig and axes
    """
    # Queries arguments
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

    # No data in the dataset
    if len(df_s) == 0:
        raise ValueError("No data for these keywords....")

    # Get data
    id = df_s.iloc[0].id
    if verbose:
        print("Following data row is used:")
        print(
            f"Id: {df_s.iloc[0].id}, Task: {df_s.iloc[0].task}, Model: {df_s.iloc[0].model_name}, Defense: {df_s.iloc[0].defense}, Loss: {df_s.iloc[0].loss}, N_train: {df_s.iloc[0].N_train}"
        )

    # Get task and model
    task = maybe_get_task_by_id(name, id, task)
    model = maybe_get_model_by_id(name, model_id=id, model=model).to(device)

    if task is not None:
        prior = task.get_prior(device=device)
        simulator = task.get_simulator(device=device)

    # Get random x_o and theta_o if not provided
    if idx_adv_example is not None:
        id_adversarial = df_s.id_adversarial.iloc[0]
        x, theta, x_tilde = maybe_get_x_tilde_from_id(
        name, id_adversarial, None, None, None
        )
        x_o = x[idx_adv_example].to(device)
        theta_o = theta[idx_adv_example].to(device)
    elif x_o is None and task is not None:
        theta_o = prior.sample()
        x_o = simulator(theta_o)

    if idx_adv_example is not None and plot_true is not None:
        id_adversarial = df_s.id_adversarial.iloc[0]
        full_dict  = load_posterior_samples_by_id(name, id_adversarial)
        if plot_true == "x_tilde":
            dict1 = full_dict["xs_tilde_top"]
            dict2 = full_dict["xs_tilde_rand"]
        elif plot_true == "x":
            dict1 = full_dict["xs_top"]
            dict2 = full_dict["xs_rand"]
        else:
            dict1 = {}
            dict2 = {}

        if idx_adv_example in dict1:
            true_thetas = dict1[idx_adv_example]
        elif idx_adv_example in dict2:
            true_thetas = dict2[idx_adv_example]
        else:
            true_thetas = None
    else:
        true_thetas = None

    # If there is nothing provided and we have a unknown task, raise error
    if x_o is None and task is None:
        raise ValueError("Either give me the task you trained on or an obsevation...")

    # Collect the samples
    samples = model(x_o).sample((n_samples,))
    samples = [samples.reshape(n_samples, -1).cpu()]

    if true_thetas is not None:
        samples += [true_thetas.cpu()]

    # TODO Plotting ground truth samples

    # Get ploting function
    if type == "pairplot":
        fig, axes = custom_pairplot(
            samples,
            points=[theta_o.reshape(1, -1).cpu()],
            **plotting_kwargs,
        )
    elif type == "marginalplot":
        fig, axes = custom_marginal_plot(
            samples,
            points=[theta_o.reshape(1, -1).cpu()],
            **plotting_kwargs,
        )
    elif type == "pairplot_sbi":
        fig, axes = pairplot(
            samples,
            points=[theta_o.reshape(1, -1).cpu()],
            **get_plotting_kwargs_pairplot(plotting_kwargs),
        )
    elif type == "marginalplot_sbi":
        fig, axes = marginal_plot(
            samples,
            points=[theta_o.reshape(1, -1).cpu()],
            **get_plotting_kwargs_marginal_plot(plotting_kwargs),
        )
    else:
        raise ValueError("Unknown posterior")
    return fig, axes


def plot_adversarial_posterior(
    name: str,
    task: Optional[str]=None,
    model_name: Optional[str]=None,
    metric_approx_clean: Optional[str]=None,
    defense: Optional[str]=None,
    loss: Optional[str]=None,
    type: str ="pairplot",
    verbose: bool =True,
    model: Optional[str]=None,
    idx_adv_example: int=0,
    n_samples: int =15000,
    x: Optional[Tensor]=None,
    theta: Optional[Tensor]=None,
    x_tilde: Optional[Tensor]=None,
    device: str="cpu",
    plotting_kwargs: dict ={},
    plot_adv_example: Optional[str]=None,
    plot_true: Optional[str] = None,
    **kwargs,
):
    """ This function plots the approximate posterior learned for a certain obervation x_o. 

    Args:
        name (str): Dataset name
        task (Optional[str], optional): Task solved by model. Defaults to None.
        model_name (Optional[str], optional): Type of the model . Defaults to None.
        metric_approx_clean (Optional[str], optional): Approximation metric name . Defaults to None.
        metric_approx_tilde (Optional[str], optional): Approximaiton metric for adversarial examples. Defaults to None.
        defense (Optional[str], optional): Defense applied on the model . Defaults to None.
        loss (Optional[str], optional): Loss function used to train. Defaults to None.
        type (str, optional): Type of the plot i.e. "pariplot", "marginal_plot", "pairplot_sbi"... . Defaults to "pairplot".
        verbose (bool, optional): If to plot which data is used to obtain the plot . Defaults to True.
        model (Optional[Any], optional): This is fetched from the dataset if not given . Defaults to None.
        n_samples (int, optional): Number of samples to use to create the plot. Defaults to 15000.
        x (Optional[Tensor], optional): Observation to condition on. Defaults to None.
        theta (Optional[Tensor], optional): Parameter which created the observation. Defaults to None.
        x_tilde (Optional[Tensor], optional): Adversarial example.
        device (str, optional): Compute device . Defaults to "cpu".
        plotting_kwargs (dict, optional): Plots passed to the plotting function. Defaults to {}.
        plot_adv_example: Plot some features of the adversarial example.
        plot_true: Also plot true posterior.
        kwargs: Other keywords for the search in the dataset...

    Raises:
        ValueError: No data found matching the keywords
        ValueError: No common task and no x_o provided...
        ValueError: _description_

    Returns:
        tuple: fig and axes
    """

    # Fetch some data
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

    # Fetch model and data
    model = maybe_get_model_by_id(name, model_id=id, model=model).to(device)
    x, theta, x_tilde = maybe_get_x_tilde_from_id(
        name, id_adversarial, x, theta, x_tilde
    )

    x = x[idx_adv_example].to(device)
    if theta is not None:
        theta = theta[idx_adv_example].to(device)
    x_tilde = x_tilde[idx_adv_example].to(device)

    # Get both approximations
    q = model(x)
    q_tilde = model(x_tilde)

    samples = q.sample((n_samples,))
    samples_tilde = q_tilde.sample((n_samples,))

    # Also plot true samples if available and wanted.
    available_true_samples = False
    if plot_true is not None:
        full_dict  = load_posterior_samples_by_id(name, id_adversarial)

        if not isinstance(plot_true, list):
            plot_true = [plot_true]

        true_samples = []
        for pt in plot_true:
            if pt == "x_tilde":
                dict1 = full_dict["xs_tilde_top"]
                dict2 = full_dict["xs_tilde_rand"]
            elif pt == "x":
                dict1 = full_dict["xs_top"]
                dict2 = full_dict["xs_rand"]

            if idx_adv_example in dict1:
                true_samples.append(dict1[idx_adv_example])
                available_true_samples = True
            elif idx_adv_example in dict2:
                true_samples.append(dict2[idx_adv_example])
                available_true_samples = True 
            else:
                available_true_samples = False
    

    all_samples = [samples.reshape(n_samples, -1).cpu(),
                samples_tilde.reshape(n_samples, -1).cpu()]    
    
    if available_true_samples:
        all_samples.extend(true_samples)
        
    # Plot specified plot
    if type == "pairplot":
        fig, axes = custom_pairplot(
            all_samples,
            points=[theta.reshape(1, -1).cpu()],
            **plotting_kwargs,
        )
    elif type == "marginalplot":
        fig, axes = custom_marginal_plot(
            all_samples,
            points=[theta.reshape(1, -1).cpu()],
            **plotting_kwargs,
        )
    elif type == "2djointplot":
        fig, axes = custom_2d_joint_plot(
            all_samples,
            points=[theta.reshape(1, -1).cpu()],
            **plotting_kwargs,
        )
    elif type == "pairplot_sbi":
        fig, axes = pairplot(
            all_samples,
            points=[theta.reshape(1, -1).cpu()],
            **get_plotting_kwargs_pairplot(plotting_kwargs),
        )
    elif type == "marginalplot_sbi":
        fig, axes = marginal_plot(
            all_samples,
            points=[theta.reshape(1, -1).cpu()],
            **get_plotting_kwargs_marginal_plot(plotting_kwargs),
        )
    else:
        raise ValueError("Unknown posterior")

    # Add additional info
    if plot_adv_example is not None:
        plot_adversarial_example_on_top(fig, plot_adv_example, task, x, x_tilde)

    return fig, axes


def plot_posterior_predictive(
    name: str,
    task: Optional[str]=None,
    model_name: Optional[str]=None,
    metric_approx_clean: Optional[str]=None,
    defense: Optional[str]=None,
    loss: Optional[str]=None,
    verbose: bool =True,
    model: Optional[Any]=None,
    x_o: Optional[Tensor]=None,
    theta_o: Optional[Tensor]=None,
    device: str="cpu",
    idx_adv_example: Optional[int] = None,
    plot_true: Optional[str] = None,
    plotting_kwargs: dict ={},
    **kwargs,
):
    """Plot the posterior predictive.

    Args:
        name (str): Name of the dataset 
        task (Optional[str], optional): Task . Defaults to None.
        model_name (Optional[str], optional): Type of the model. Defaults to None.
        metric_approx_clean (Optional[str], optional): Approxmiation metric. Defaults to None.
        defense (Optional[str], optional): Defense used. Defaults to None.
        loss (Optional[str], optional): Loss used to train. Defaults to None.
        verbose (bool, optional): If to plot data . Defaults to True.
        model (Optional[Any], optional): Model to use. Defaults to None.
        x_o (Optional[Tensor], optional): Observation . Defaults to None.
        theta_o (Optional[Tensor], optional): True parameter . Defaults to None.
        device (str, optional): Compute device. Defaults to "cpu".
        plotting_kwargs (dict, optional): Plotting kwargs . Defaults to {}.
        kwargs: Additional query keywords

    Raises:
        ValueError: No data

    Returns:
        tuple: Figure and axes
    """

    # Get data
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

    # Get plotting function appropriate for task.
    plot_fn = get_predicitve_plotting_function(task)

    # Get data
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

    # Get x_o if not specified
    if task is not None:
        prior = task.get_prior(device=device)
        simulator = task.get_simulator(device=device)

    if idx_adv_example is not None:
        id_adversarial = df_s.id_adversarial.iloc[0]
        x, theta, x_tilde = maybe_get_x_tilde_from_id(
        name, id_adversarial, None, None, None
        )
        x_o = x[idx_adv_example].to(device)
        theta_o = theta[idx_adv_example].to(device)
    if x_o is None and task is not None:
        theta_o = prior.sample()
        x_o = simulator(theta_o)

    if idx_adv_example is not None and plot_true is not None:
        id_adversarial = df_s.id_adversarial.iloc[0]
        full_dict  = load_posterior_samples_by_id(name, id_adversarial)
        if plot_true == "x_tilde":
            dict1 = full_dict["xs_tilde_top"]
            dict2 = full_dict["xs_tilde_rand"]
        elif plot_true == "x":
            dict1 = full_dict["xs_top"]
            dict2 = full_dict["xs_rand"]
        else:
            dict1 = {}
            dict2 = {}

        if idx_adv_example in dict1:
            true_thetas = dict1[idx_adv_example]
        elif idx_adv_example in dict2:
            true_thetas = dict2[idx_adv_example]
        else:
            true_thetas = None
    else:
        true_thetas = None

    return plot_fn(model, task, x_o, true_thetas=true_thetas,**plotting_kwargs)


def plot_adversarial_posterior_predictive(
    name: str,
    task: Optional[str]=None,
    model_name: Optional[str]=None,
    metric_approx_clean: Optional[str]=None,
    defense: Optional[str]=None,
    loss: Optional[str]=None,
    verbose: bool =True,
    model: Optional[Any]=None,
    idx_adv_example: int =0,
    x: Optional[Tensor]=None,
    theta: Optional[Tensor]=None,
    x_tilde: Optional[Tensor]=None,
    device: str="cpu",
    plot_true: Optional[str] = None,
    plotting_kwargs: dict={},
    **kwargs,
):
    """Plot the posterior predictive together with adversarial posterior predictive...

    Args:
        name (str): Name of the dataset 
        task (Optional[str], optional): Task . Defaults to None.
        model_name (Optional[str], optional): Type of the model. Defaults to None.
        metric_approx_clean (Optional[str], optional): Approxmiation metric. Defaults to None.
        defense (Optional[str], optional): Defense used. Defaults to None.
        loss (Optional[str], optional): Loss used to train. Defaults to None.
        verbose (bool, optional): If to plot data . Defaults to True.
        model (Optional[Any], optional): Model to use. Defaults to None.
        x_o (Optional[Tensor], optional): Observation . Defaults to None.
        theta_o (Optional[Tensor], optional): True parameter . Defaults to None.
        device (str, optional): Compute device. Defaults to "cpu".
        plotting_kwargs (dict, optional): Plotting kwargs . Defaults to {}.
        kwargs: Additional query keywords

    Raises:
        ValueError: No data

    Returns:
        tuple: Figure and axes
    """

    # Get data
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

    # Get plotting function
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

    samples = [x[idx_adv_example], x_tilde[idx_adv_example]]

    if plot_true is not None:
        full_dict  = load_posterior_samples_by_id(name, id_adversarial)
        if plot_true == "x_tilde":
            dict1 = full_dict["xs_tilde_top"]
            dict2 = full_dict["xs_tilde_rand"]
        elif plot_true == "x":
            dict1 = full_dict["xs_top"]
            dict2 = full_dict["xs_rand"]
        else:
            dict1 = {}
            dict2 = {}

        if idx_adv_example in dict1:
            true_thetas = dict1[idx_adv_example]
        elif idx_adv_example in dict2:
            true_thetas = dict2[idx_adv_example]
        else:
            true_thetas = None
    else:
        true_thetas = None


    return plot_fn(
        model,
        task,
        torch.stack(samples),
        true_thetas = true_thetas,
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
