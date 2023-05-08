
from rbi.models.base import ParametricProbabilisticModel, PyroFlowModel

from rbibm.tasks.base import Task, InferenceTask
import torch


from typing import Optional, Union, List, Dict, Type
from rbibm.metric.base import ApproximationMetric, PredictiveMetric
from rbibm.utils.batched_processing import get_x_theta_from_loader
from rbibm.utils.get_kernel_mmd import get_kernel
import time

from logging import Logger


def run_approx_eval_clean(
    task: InferenceTask,
    model: Union[ParametricProbabilisticModel, PyroFlowModel],
    metric_classes: List[Type[Union[ApproximationMetric, PredictiveMetric]]],
    test_loader: Optional[object] = None,
    metric_params: Optional[List[Dict]] = None,
    device: str = "cpu",
    verbose: bool = True,
    logger: Optional[Logger] = None,
    **kwargs,
):
    """Evaluate approximation metric for clean data.

    Args:
        task (Task): Task
        model (Union[ParametricProbabilisticModel, PyroFlowModel]): Trained model
        metric_classes (List[Union[ApproximationMetric, PredictiveMetric]]): Metric class
        test_loader (Optional[object], optional): Test dataset. Defaults to None.
        metric_params (Optional[List[dict]], optional): Metric parameters. Defaults to None.
        device (str, optional): Device. Defaults to "cpu".
        verbose (bool, optional): If logging a detail. Defaults to True.
        logger (Optional[Logger], optional): Logging . Defaults to None.
        batch_size (Optional[int], optional): Batch size. Defaults to None.

    Returns:
        _type_: _description_
    """
    model = model.to(device).eval()  # type: ignore

    # Get data
    if test_loader is not None:
        xs, thetas = get_x_theta_from_loader(test_loader)
    else:
        prior = task.get_prior(device=device)  # type: ignore
        simulator = task.get_simulator(device=device)

        thetas = prior.sample((10000,))  # type: ignore
        xs = simulator(thetas)

    # TODO THIS WE REMOVE FROM HERE...
    # Ground truth only applicable if implemented
    eval_ground_truth_metrics = True

    try:
        ground_truth = task.get_true_posterior(device=device)
        # Only for gaussin linear with FAST ground truth
        if "Gaussian" not in task.__name__:
            ground_truth = None
    except:
        ground_truth = None
        eval_ground_truth_metrics = False

    # Potential fn based metrics only applicable if potential tractable.
    eval_with_potential_fn = True
    try:
        potential_fn = task.get_potential_fn(device=device)
    except:
        potential_fn = None
        eval_with_potential_fn = False

    # Some metrics need a simulaotr.
    simulator = task.get_simulator(device=device)

    metric_names = []
    main_values = []
    additional_values = []
    runtimes = []

    if metric_params is None:
        metric_params = [{} for i in range(len(metric_classes))]

    for metric_class, metric_params in zip(metric_classes, metric_params):  # type: ignore
        metric_name = metric_class.__name__
        metric_params = dict(metric_params)

        if "kernel" in metric_params:
            k = get_kernel(
                **metric_params["kernel"],
                model=model,
                xs=torch.concat(
                    [xs[:500], xs[:500] + (torch.rand_like(xs[:500]) - 0.5)]
                ),
            )
            metric_params["kernel"] = k

        start_time = time.time()

        # Skip if no ground truth is there
        if "GroundTruth" in metric_name and not eval_ground_truth_metrics:
            continue

        # Initialize metric
        metric = metric_class(
            model,
            simulator=simulator,
            ground_truth=ground_truth,
            potential_fn=potential_fn,
            device=device,
            **metric_params,  # type: ignore
        )

        # Skip if metric requires potential but cant eval potential.
        if metric.requires_potential and eval_with_potential_fn == False:
            continue

        # If metric requires thetas or not
        if not metric.requires_thetas:
            val = metric.eval(xs)
        else:
            val = metric.eval(xs, thetas)
        if isinstance(val, tuple):
            main_value, additional_value = val
        else:
            main_value = val
            additional_value = None

        if logger is None:
            if verbose:
                print(f"Approx. Metric: {metric_name}")
                print(f"Value: {main_value:.2f}")
        else:
            if verbose:
                logger.info(f"Approx. Metric: {metric_name}")
                logger.info(f"Value: {main_value:.2f}")

        end_time = time.time()
        eval_time = end_time - start_time

        metric_names.append(metric_name)
        main_values.append(main_value)
        additional_values.append(additional_value)
        runtimes.append(eval_time)

    return metric_names, main_values, additional_values, runtimes


def run_approx_eval_tilde(
    task: Task,
    model: Union[ParametricProbabilisticModel, PyroFlowModel],
    metric_classes: List[Type[Union[ApproximationMetric, PredictiveMetric]]],
    x_unsecure: torch.Tensor,
    theta_unsecure: torch.Tensor,
    x_adversarial: torch.Tensor,
    metric_params: Optional[List[Dict]] = None,
    device: str = "cpu",
    verbose: bool = True,
    logger: Optional[Logger] = None,
    **kwargs,
):
    """Run approximation metric on perturbed data.

    Args:
        task (Task): The task.
        model (Union[ParametricProbabilisticModel, PyroFlowModel]): The trained model.
        metric_classes (List[Union[ApproximationMetric, PredictiveMetric]]): The metric class.
        x_unsecure (torch.Tensor): The x with successful attacks on.
        theta_unsecure (torch.Tensor): True thetas for that x.
        x_adversarial (torch.Tensor): Perturbed x.
        metric_params (Optional[List[dict]], optional): Metric parameters. Defaults to None.
        best_k (int, optional): Best k metric evals. Defaults to 10.
        device (str, optional): Devices. Defaults to "cpu".
        verbose (bool, optional): How much we log. Defaults to True.
        logger (Optional[Logger], optional): Logger. Defaults to None.
        batch_size (Optional[int], optional): Batch size. Defaults to None.

    Returns:
        _type_: _description_
    """
    model = model.to(device).eval()  # type: ignore

    # Ground truth metrics only applicable if ground truth exists.
    eval_ground_truth_metrics = True
    try:
        ground_truth = task.get_true_posterior(device=device)  # type: ignore
        if "Gaussian" not in task.__name__:
            ground_truth = None
    except:
        ground_truth = None
        eval_ground_truth_metrics = False

    # Potential based metrics only applicable if potential exits
    eval_with_potential_fn = True
    try:
        potential_fn = task.get_potential_fn(device=device)  # type: ignore
    except:
        potential_fn = None
        eval_with_potential_fn = False

    # Simulator for some metrics
    simulator = task.get_simulator(device=device)

    metric_names = []
    values_adversarial = []
    values_unsecure = []
    values_adversarial2x = []
    additional_values_adversarial = []
    additional_values_unsecure = []
    additional_values_adversarial2x = []
    runtimes = []

    if metric_params is None:
        metric_params = [{} for i in range(len(metric_classes))]

    for metric_class, metric_params in zip(metric_classes, metric_params):  # type: ignore
        metric_name = metric_class.__name__
        metric_params = dict(metric_params)

        if "kernel" in metric_params:
            k = get_kernel(
                **metric_params["kernel"],
                model=model,
                xs=torch.concat([x_unsecure, x_adversarial], dim=0),
            )
            metric_params["kernel"] = k

        start_time = time.time()

        # Skip if no ground truth is there
        if "GroundTruth" in metric_name and not eval_ground_truth_metrics:
            continue

        # Init metric
        metric = metric_class(
            model,
            simulator=simulator,
            ground_truth=ground_truth,
            potential_fn=potential_fn,
            device=device,
            **metric_params,  # type: ignore
        )

        # Skip metrics that cannot be computed.
        if metric.requires_potential and eval_with_potential_fn == False:
            continue

        if not metric.requires_thetas:
            val_tilde = metric.eval(x_adversarial)
            val_unsecure = metric.eval(x_unsecure)
        else:
            val_tilde = metric.eval(x_adversarial, theta_unsecure)
            val_unsecure = metric.eval(x_unsecure, theta_unsecure)

        if metric.can_eval_x_xtilde:
            val_tilde_x = metric.eval(x_adversarial, x_unsecure)
            if isinstance(val_tilde_x, tuple):
                main_val_tilde_x, additional_val_tilde_x = val_tilde_x
            else:
                main_val_tilde_x = val_tilde_x
                additional_val_tilde_x = None
        else:
            main_val_tilde_x = None
            additional_val_tilde_x = None

        if isinstance(val_tilde, tuple):
            main_val_tilde, additional_val_tilde = val_tilde
            main_val_unsec, additional_val_unsec = val_unsecure
        else:
            main_val_tilde = val_tilde
            main_val_unsec = val_unsecure
            additional_val_tilde = None
            additional_val_unsec = None

        if logger is None:
            if verbose:
                print(f"Approx. Metric: {metric_name}")
                print(f"Value adv: {main_val_tilde:.2f}")
                print(f"Value x unsecure: {main_val_unsec:.2f}")
                if main_val_tilde_x is not None:
                    print(f"Value tilde to x: {main_val_tilde_x:.2f}")
        else:
            if verbose:
                logger.info(f"Approx. Metric: {metric_name}")
                logger.info(f"Value adv: {main_val_tilde:.2f}")
                logger.info(f"Value x unsecure: {main_val_unsec:.2f}")
                if main_val_tilde_x:
                    logger.info(f"Value tilde to x: {main_val_tilde_x:.2f}")

        end_time = time.time()
        eval_time = end_time - start_time

        metric_names.append(metric_name)
        values_adversarial.append(main_val_tilde)
        values_unsecure.append(main_val_unsec)
        values_adversarial2x.append(main_val_tilde_x)
        additional_values_adversarial.append(additional_val_tilde)
        additional_values_unsecure.append(additional_val_unsec)
        additional_values_adversarial2x.append(additional_val_tilde_x)
        runtimes.append(eval_time)

    return (
        metric_names,
        values_adversarial,
        additional_values_adversarial,
        values_unsecure,
        additional_values_unsecure,
        values_adversarial2x,
        additional_values_adversarial2x,
        runtimes,
    )
