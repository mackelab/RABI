from logging import Logger
from typing import Callable, List, Optional, Tuple, Type, Union

import torch
from rbi.attacks.custom_attacks import Attack
from rbi.loss.base import EvalLoss
from rbi.models.base import ParametricProbabilisticModel, PyroFlowModel
from torch.utils.data import DataLoader

from sbi.utils.metrics import c2st

from rbibm.metric.robustness_metric import (
    ForwardKLRobMetric,
    NLLRobMetric,
    ReverseKLRobMetric,
    MMDsquaredRobMetric,
)
from rbibm.tasks.base import CDETask, InferenceTask, Task
from rbibm.utils.batched_processing import get_x_theta_from_loader
from rbibm.utils.get_kernel_mmd import get_kernel


def run_rob_eval(
    task: InferenceTask,
    model: Union[ParametricProbabilisticModel, PyroFlowModel],
    metric_class: Type[
        Union[ForwardKLRobMetric, NLLRobMetric, ReverseKLRobMetric, MMDsquaredRobMetric]
    ],
    attack_class: Type[Attack],
    test_loader: Optional[DataLoader] = None,
    attack_attemps: int = 10,
    targeted: bool = False,
    target_strategy: str = "perm_x",
    attack_loss_fn: Optional[EvalLoss] = None,
    attack_mc_budget: int = 4,
    eval_mc_budget: int = 100,
    eps: float = 0.5,
    attack_hyper_parameters: dict = {},
    metric_hyper_parameters: dict = {},
    num_adversarial_examples: int = 1000,
    reduction: Optional[str] = "mean_summary",
    device: str = "cpu",
    verbose: bool = True,
    eps_relative_to_std: bool = True,
    clip_min_max_automatic: bool = True,
    batch_size: Optional[int] = None,
    logger: Optional[Logger] = None,
    **kwargs,
):
    """Evaluates the robustness/smoothness of the amortized posterior approximation.

    Args:
        task (Task): The task.
        model (Union[ParametricProbabilisticModel, PyroFlowModel]): The trained model.
        metric_classes (List[RobustnessMetric]): The metric class.
        test_loader (Optional[object], optional): The test loader. Defaults to None.
        attack_class (Optional[Attack], optional): The attack class. Defaults to None.
        attack_attemps (int, optional): Attack attemps. Defaults to 10.
        targeted (bool, optional): If a targeted attack should be performed. Defaults to False.
        target_strategy (str, optional): Targeting strategy. Defaults to "perm_x".
        eval_targets (bool, optional): Eval distance to targets. Defaults to False.
        eval_for_ground_truth_if_applicable (bool, optional): Eval ground truth if applicable. Defaults to True.
        attack_loss_fn (Optional[EvalLoss], optional): Attack loss_fn. Defaults to None.
        attack_mc_budget (int, optional): Attack mc budget. Defaults to 16.
        eval_mc_budget (int, optional): Eval mc budget. Defaults to 512.
        eps (Optional[float], optional): Attack tolerance. Defaults to None.
        attack_hyper_parameters (dict, optional): Attack hyperparameter. Defaults to {}.
        num_adversarial_examples (int, optional): Number of adversaria examples. Defaults to 500.
        device (str, optional): Device to compute on. Defaults to "cpu".
        verbose (bool, optional): If we should log in detail. Defaults to True.
        batch_size (Optional[int], optional): Batch size. Defaults to None.
        logger (Optional[Logger], optional): Logger. Defaults to None.

    Raises:
        ValueError: Something went wrong.

    Returns:
        _type_: _description_
    """

    # Get data
    if test_loader is not None:
        xs, thetas = get_x_theta_from_loader(test_loader)
    else:
        prior = task.get_prior(device=device)  # type: ignore
        simulator = task.get_simulator(device=device)

        thetas = prior.sample((10000,))  # type: ignore
        xs = simulator(thetas)

    metric_params = dict(metric_hyper_parameters)
    if "mask" in metric_params:
        if metric_params["mask"] is not None:
            mask = torch.as_tensor(metric_params["mask"]).bool()
            metric_params["mask"] = mask
        else:
            mask = None
    else:
        mask = None

    print(mask)

    # Relative eps if neccesary
    if eps_relative_to_std:
        if mask is None:
            std = xs.std(1).mean()
        else:
            std = xs[..., mask].std(1).mean()
        eps_abs = float(eps * std)
    else:
        eps_abs = float(eps)

    # Move to right device

    model = model.to(device).eval()  # type: ignore

    # Create loss function the may is needed by attack

    if "kernel" in metric_params:
        k = get_kernel(
            **metric_params["kernel"],
            model=model,
            xs=torch.concat(
                [xs[:500], xs[:500] + eps_abs * (torch.rand_like(xs[:500]) - 0.5)]
            ),
        )
        metric_params["kernel"] = k

    if attack_loss_fn is not None:
        if "MMD" in attack_loss_fn.__name__ or "Kernel" in attack_loss_fn.__name__:
            loss_fn_a = attack_loss_fn(mc_samples=attack_mc_budget, kernel=k)
        else:
            loss_fn_a = attack_loss_fn(mc_samples=attack_mc_budget)
    else:
        loss_fn_a = None

    attack_hyper_parameters = dict(attack_hyper_parameters)
    attack_hyper_parameters["targeted"] = targeted

    if "eps_iter" in attack_hyper_parameters:
        if attack_hyper_parameters["eps_iter"] == "auto":
            nb_iters = attack_hyper_parameters.get("nb_iter", 1)
            eps_iter = min((eps_abs / nb_iters) * 20, eps_abs)
            attack_hyper_parameters["eps_iter"] = eps_iter

    if clip_min_max_automatic:
        mini = xs.min()
        maxi = xs.max()
        attack_hyper_parameters["clip_min"] = float(mini)
        attack_hyper_parameters["clip_max"] = float(maxi)

    if loss_fn_a is not None:
        attack = attack_class(model, loss_fn_a, eps=eps_abs, **attack_hyper_parameters)
    else:
        attack = attack_class(model, eps=eps_abs, **attack_hyper_parameters)

    if verbose:
        if logger is None:
            print(f"Following attack is used: {type(attack).__name__}")
            print(f"Following loss_fn is used by the attack: {loss_fn_a}")
            print(f"Following parameters used {attack_hyper_parameters}")
        else:
            logger.info(f"Following attack is used: {type(attack).__name__}")
            logger.info(f"Following loss_fn is used by the attack: {loss_fn_a}")
            logger.info(f"Following parameters used {attack_hyper_parameters}")

    # Gather targets if neccessary.
    if targeted:
        with torch.no_grad():
            targets, target_wrapper = get_targets(
                model, task, xs, thetas, target_strategy
            )
    else:
        targets = None
        target_wrapper = lambda x: x

    # Init metric
    metric = metric_class(
        model,
        attack=attack,
        mc_samples=eval_mc_budget,
        attack_attemps=attack_attemps,
        targeted=targeted,
        target_wrapper=target_wrapper,
        device=device,
        batch_size=batch_size,
        reduction=reduction,
        **metric_params,
    )

    metric_name = metric_class.__name__

    # Eval metric
    if not targeted:
        out = metric.eval(xs)
    else:
        out = metric.eval(xs, targets)

    if isinstance(out, Tuple):
        value, additional_values = out
    else:
        value = out
        additional_values = None

    # Report metric
    if verbose:
        if logger is None:
            print(f"Rob. Metric: {metric_name}")
            print(f"Value: {float(value.cpu()):.2f})")
        else:
            logger.info(f"Rob. Metric: {metric_name[-1]}")
            logger.info(f"Value: {float(value.cpu()):.2f})")

    # Corresponding adversarial examples
    x_idx, x_adversarial = metric.generate_adversarial_examples(
        xs, num_examples=num_adversarial_examples
    )

    # C2ST between x adversarial and normal x
    x_idx = x_idx.cpu()
    x_adversarial = x_adversarial.cpu()
    xs = xs.cpu()
    try:
        c2st_x = c2st(x_adversarial, xs[: x_adversarial.shape[0]])
    except:
        c2st_x = 1.

    x_unsecure = xs[x_idx]
    theta_unsecure = thetas[x_idx]

    return (
        metric_name,
        value,
        additional_values,
        x_unsecure,
        theta_unsecure,
        x_adversarial,
        c2st_x,
        eps_abs,
    )


def get_targets(
    model: Union[ParametricProbabilisticModel, PyroFlowModel],
    task: InferenceTask,
    x: torch.Tensor,
    theta: torch.Tensor,
    strategy: str = "perm_x",
) -> Tuple[torch.Tensor, Callable]:
    """Generates automatic targets.

    Args:
        model (Union[ParametricProbabilisticModel, PyroFlowModel]): Model to test
        task (InferenceTask): Task to test
        x (torch.Tensor): Data to eval.
        theta (torch.Tensor): Params to eval
        strategy (str, optional): Strategy on which targets are created.. Defaults to "perm_x".

    Raises:
        NotImplementedError: Strategy not implemented

    Returns:
        Tensor, Callable: Targets, Targets Wrapper
    """
    if strategy == "perm_x":
        perm = torch.randperm(x.shape[0], device=x.device)
        x_perm = x.clone()[perm]

        def wrapper(x):
            with torch.no_grad():
                return model(x)

        return x_perm, wrapper
    elif strategy == "theta_bad":
        q = model(x)
        thetas = task.get_prior(device=theta.device).sample((100, x.shape[0]))
        logq_thetas = q.log_prob(thetas)
        idx = logq_thetas.argmin(0)
        theta = torch.vstack([thetas[i, j] for j, i in enumerate(idx)])
        return theta, lambda x: x

    elif strategy == "rand_dist":
        prior = task.get_prior(x.device)
        theta = prior.sample((x.shape[0],))
        return theta, lambda x: torch.distributions.Independent(
            torch.distributions.Normal(x, 0.1), 1
        )
    elif strategy == "theta":
        return theta, lambda x: x
    else:
        raise NotImplementedError()
