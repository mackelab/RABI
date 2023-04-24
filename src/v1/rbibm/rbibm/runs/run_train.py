from arviz import InferenceData
from numpy import isin
from rbi.models.base import ParametricProbabilisticModel, PyroFlowModel
from rbi.loss.base import TrainLoss
from rbi.loss import NLLLoss, NegativeElboLoss
from rbibm.tasks.base import Task
from rbi.defenses.base import Defense
import time
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from copy import deepcopy

from typing import Any, Optional, Union, Tuple, List, Type
from rbi.models.module import (
    ZScoreLayer,
)
from rbi.defenses.base import PostHocDefense
from rbi.defenses import SIRPostHocAdjustment

from logging import Logger


def run_train(
    task: Task,
    model: Union[ParametricProbabilisticModel, PyroFlowModel],
    loss_fn: Type[TrainLoss],
    train_loader: DataLoader,
    validation_loader: Optional[DataLoader] = None,
    test_loader: Optional[DataLoader] = None,
    defense: Optional[Type[Defense]] = None,
    early_stopping: bool = True,
    patience: int = 5,
    min_epochs: int = 5,
    max_epochs: int = 100,
    optimizer: Any = torch.optim.Adam,
    lr_scheduler=None,
    lr_scheduler_kwargs={},
    lr: float = 1e-4,
    grad_clip_value: Optional[float] = None,
    defense_hyper_parameters: dict = {},
    loss_fn_hyper_parameters: dict = {},
    device: str = "cpu",
    z_score_x: bool = True,
    initialize_as_prior: bool = False,
    initialize_as_prior_rounds: int = 2,
    logger: Optional[Logger] = None,
    verbose: bool = True,
    **kwargs,
) -> Tuple[
    Union[ParametricProbabilisticModel, PyroFlowModel],
    Defense,
    List[Tensor],
    Optional[List[Tensor]],
    Optional[Union[float, Tensor]],
]:
    """Function that trains a conditional density estimator to perform amortized Bayesian inference.

    Args:
        task (Task): The task to complete.
        model (Union[ParametricProbabilisticModel, PyroFlowModel]): A model.
        loss_fn (TrainLoss): A loss function to use during training.
        train_loader (DataLoader): Training dataset
        validation_loader (Optional[DataLoader], optional): Validation dataset, used to e.g. evaluate convergence. Defaults to None.
        test_loader (Optional[DataLoader], optional): Test dataset, used during evaluation. Defaults to None.
        defense (Optional[Defense], optional): Defense. Defaults to None.
        early_stopping (bool, optional): If early stopping should be used. Defaults to True.
        patience (float, optional): Patience of early stopping. Defaults to 5.
        min_epochs (int, optional): Minimal number of epochs. Defaults to 5.
        max_epochs (int, optional): Maximal number of epochs. Defaults to 100.
        optimizer (Any, optional): Optimizer to use. Defaults to torch.optim.Adam.
        lr_scheduler (_type_, optional): Scheduler to use, if any. Defaults to None.
        lr (float, optional): Initial learning rate. Defaults to 1e-3.
        grad_clip_value (Optional[float], optional): Gradient clipping. Defaults to None.
        defense_hyper_parameters (dict, optional): Defense hyperparameter . Defaults to {}.
        loss_fn_hyper_parameters (dict, optional): Loss hyperparameter. Defaults to {}.
        device (str, optional): Device to train on. Defaults to "cpu".
        z_score_x (bool, optional): If a ZScoring layer should be introduced. Defaults to True.
        initialize_as_prior (bool, optional): Initalize model as prior. Can be more stable. Defaults to False.
        initialize_as_prior_rounds (int, optional): The larger the number the better is the prior approximation.
        logger (Optional[Logger], optional): A logger to use if any. Defaults to None.
        verbose (bool, optional): If we should log/print what is going on. Defaults to True.

    Returns:
        Tuple[Union[ParametricProbabilisticModel, PyroFlowModel], Tensor, Tensor, Tensor]: Trained model as well as training, validation and test loss.
    """

    if defense_hyper_parameters is None:
        defense_hyper_parameters = {}

    if loss_fn_hyper_parameters is None:
        loss_fn_hyper_parameters = {}

    # Initial model, optimizers and Loss
    optim = optimizer(model.parameters(), lr=lr)
    if lr_scheduler is not None:
        scheduler = lr_scheduler(optim, **lr_scheduler_kwargs)
    else:
        scheduler = None
    loss_function = construct_loss(
        model, task, loss_fn, device=device, **loss_fn_hyper_parameters
    )

    # Zscoreing layer if necessary
    if z_score_x:
        X = torch.vstack([xy[0] for xy in train_loader.dataset])
        mean = X.mean(0)
        std = X.std(0)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
        std = torch.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0).clamp(min=0.05)
        z_score = ZScoreLayer(mean=mean, std=std)
        model.embedding_net = nn.Sequential(z_score, model.embedding_net)

    # Train mode
    loss_function.train()
    model.train()

    # To right device
    loss_function.to(device)
    model.to(device)

    # Early stopping if necessary.
    if early_stopping:
        convergence_checker = EarlyStopping(
            model, patience=patience, min_epochs=min_epochs
        )

    # Initialize as prior if neccessary
    if initialize_as_prior:
        init_as_prior(
            model=model,
            train_loader=train_loader,
            task=task,
            device=device,
            warm_up_rounds=initialize_as_prior_rounds,
        )

    # Initial defense
    if defense is not None:
        if "scale_eps_beta_by_std" in defense_hyper_parameters:
            if defense_hyper_parameters.pop("scale_eps_beta_by_std"):
                X = torch.vstack([xy[0] for xy in train_loader.dataset])
                std = float(X.std(1).mean())
                # if "beta" in defense_hyper_parameters:
                #     defense_hyper_parameters["beta"] = defense_hyper_parameters["beta"]*std
                if "eps" in defense_hyper_parameters:
                    defense_hyper_parameters["eps"] = (
                        defense_hyper_parameters["eps"] * std
                    )

                if "eps_iter" in defense_hyper_parameters:
                    defense_hyper_parameters["eps_iter"] = (
                        defense_hyper_parameters["eps_iter"] * std
                    )

        if verbose:
            msg = (
                f"Defense: {defense.__name__}\n Parameters: {defense_hyper_parameters}"
            )
            if logger is None:
                print(msg)
            else:
                logger.info(msg)
        if issubclass(defense, PostHocDefense):
            defense_model = defense(model=model, task=task, **defense_hyper_parameters)
        else:
            defense_model = defense(
                model=model, loss_fn=loss_function, **defense_hyper_parameters
            )
            defense_model.activate()
    else:
        defense_model = None

    # Train loop
    train_loss = []
    validation_loss = []

    validation_exists = validation_loader is not None
    test_exists = test_loader is not None

    for i in range(max_epochs):
        epoch_loss = 0
        n = 0
        loss_function.train()
        for x, theta in train_loader:
            x = x.to(device)
            theta = theta.to(device)
            optim.zero_grad()
            loss = loss_function(x, theta)
            loss.backward()
            if grad_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)  # type: ignore
            optim.step()
            if scheduler is not None:
                scheduler.step()
            n += 1
            epoch_loss += loss.detach().cpu()

        if verbose:
            msg = f"Epoch {i}: Train  Loss: {float(epoch_loss/n):.2f}"
            if logger is None:
                print(msg)
            else:
                logger.info(msg)

        with torch.no_grad():
            loss_function.eval()
            train_loss.append(epoch_loss / n)
            if validation_exists:
                n = 0
                loss_val = torch.zeros(1, device=device)
                for x_val, theta_val in validation_loader:  # type: ignore
                    x_val = x_val.to(device)
                    theta_val = theta_val.to(device)
                    loss_val += loss_function(x_val, theta_val)
                    n += 1
                validation_loss.append(loss_val.cpu() / n)
                if verbose:
                    msg = f"Epoch {i}: Valid. Loss: {float(loss_val/n):.2f}"

                    if logger is None:
                        print(msg)
                    else:
                        logger.info(msg)

        if early_stopping:
            if validation_exists:
                converged = convergence_checker(i, validation_loss[i])  # type: ignore
            else:
                converged = convergence_checker(i, train_loss[i])  # type: ignore

            # Stop if converged
            if converged:
                break

    # Compute test_loss
    if test_exists:
        with torch.no_grad():
            test_loss = 0.0
            n = 0.0
            loss_function.eval()
            for x, theta in test_loader:
                x = x.to(device)
                theta = theta.to(device)
                test_loss += loss_function(x, theta).cpu()
                n += 1
            test_loss /= n
    else:
        test_loss = None

    if defense is not None:
        if isinstance(defense_model, PostHocDefense):
            defense_model.activate()

    if verbose:
        msg = f"----------------------------\nTest loss: {test_loss:.2f}"
        if logger is None:
            print(msg)
        else:
            logger.info(msg)

    if not validation_exists:
        validation_loss = None

    # Return results
    return model, defense_model, train_loss, validation_loss, test_loss


def construct_loss(
    model: Union[ParametricProbabilisticModel, PyroFlowModel],
    task: Task,
    loss_fn: Type[TrainLoss],
    device: str = "cpu",
    **kwargs,
) -> TrainLoss:
    """Builds the train loss function.

    Args:
        model (Union[ParametricProbabilisticModel, PyroFlowModel]): Model.
        task (Task): Task to train.
        loss_fn (TrainLoss): Loss fn class.

    Raises:
        NotImplementedError: Train loss fn not implemented.

    Returns:
        TrainLoss: Train loss function
    """
    if loss_fn == NLLLoss:
        return NLLLoss(model, **kwargs)
    elif loss_fn == NegativeElboLoss:
        return NegativeElboLoss(
            model,
            potential_fn=task.get_potential_fn(device=device),
            loglikelihood_fn=task.get_loglikelihood_fn(device=device),
            prior=task.get_prior(device=device),
            **kwargs,
        )
    else:
        raise NotImplementedError("Not implemented yet...")


def init_as_prior(
    model: nn.Module,
    train_loader: DataLoader,
    task: Task,
    device: str,
    warm_up_rounds: int = 2,
):
    """Initializes the model near the prior for numerical stability

    Args:
        model (nn.Module): Model
        train_loader (DataLoader): Train loader
        task (Task): Task
        device (str): Device
        warm_up_rounds (int, optional): Warmup rounds i.e. epochs to fit to prior. Defaults to 2.
    """
    prior = task.get_prior(device=device)
    optim = torch.optim.Adam(model.parameters())
    i = 0
    while i < warm_up_rounds:
        for x, theta in train_loader:
            x = x.to(device)
            theta = theta.to(device)
            optim.zero_grad()
            l = -model(x).log_prob(prior.sample(theta.shape[:-1])).mean()
            l.backward()
            optim.step()
            i += 1


class EarlyStopping:
    """Helper class that check for convergence."""

    def __init__(
        self,
        model: Union[ParametricProbabilisticModel, PyroFlowModel],
        patience: int,
        min_epochs: int,
        max_resets: int = 10,
    ) -> None:
        """Class that check for convergence in a early stopping manner.

        Args:
            model (Union[ParametricProbabilisticModel, PyroFlowModel]): Model which is trained
            patience (int): If a loss does not improved for *patience* number of iterations, we will stop.
            min_epochs (int): Don't stop before these epochs.
        """
        self.model = model
        self.patience = patience
        self.min_epochs = min_epochs
        self.current_patience = 0.0
        self.current_best_state = model.state_dict()
        self.current_best_loss = torch.inf
        self.resets = 0
        self.max_resets = max_resets

    def __call__(self, epoch: int, loss: Tensor) -> bool:
        """Returns if current run converged or not.

        Args:
            epoch (int): Current epoch
            loss (Tensor): Current loss.

        Returns:
            bool: Converged or not.
        """

        if loss < self.current_best_loss:
            self.current_best_loss = loss
            self.current_best_state = deepcopy(self.model.state_dict())
            self.current_patience = max(self.current_patience - 1, 0)
        else:
            self.current_patience += 1

        if torch.isnan(loss):
            self.model.load_state_dict(self.current_best_state)
            self.resets += 1
            return self.resets > self.max_resets
        elif epoch < self.min_epochs:
            return False
        elif self.current_patience > self.patience:
            self.model.load_state_dict(self.current_best_state)
            return True
        else:
            return False
