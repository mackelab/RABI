#!/usr/bin/env python

import importlib
import logging
import os
import random
import socket
import time
from logging import Logger
from typing import Iterable, Optional, Tuple, Union

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.distributions.constraints import _IndependentConstraint, _Real

from rbi.models.base import ParametricProbabilisticModel, PyroFlowModel

from rbibm.utils.torchutils import tensors_to_floats
from rbibm.runs.run_approx_eval import run_approx_eval_clean, run_approx_eval_tilde
from rbibm.runs.run_rob_eval import run_rob_eval
from rbibm.runs.run_train import run_train
from rbibm.tasks.base import Task
from rbibm.utils.embedding_nets import get_embedding_net
from rbibm.utils.output_transform import get_output_transform
from rbibm.utils.get_task import get_task
from rbibm.utils.utils_data import (
    add_new_approx_metric_clean_entry,
    add_new_approx_metric_tilde_entry,
    add_new_entry,
    add_new_rob_metric_entry,
    check_id_exists,
    check_if_already_simulated,
    check_if_approximation_metric_already_computed,
    get_full_model_dataset,
    get_model_by_id,
    get_simulations_by_id,
    init_datarepo,
    save_simulations_by_id,
    query_rob_metric,
    get_adversarial_examples_by_id,
)
from rbi.defenses.base import PostHocDefense

# from rbibm.runs.evaluate import store_results, evaluate_metric, do_not_evaluate_metric
# from sbivibm.utils import get_tasks


# TODO Start modularize this shit...


@hydra.main(version_base=None, config_path="../../config", config_name="config.yaml")
def rbibm(cfg: DictConfig) -> Optional[float]:
    """Main script with all the functionality

    Args:
        cfg (DictConfig): Config file provided by hydra

    Raises:
        ValueError: If something goes wrong...

    Returns:
        Optional[float]: Returns a certain objective for sweeping...
    """

    # Logging
    log = logging.getLogger(__name__)
    log.info(OmegaConf.to_yaml(cfg))
    # log.info(f"rbibm version: {rbibm.__version__}")
    log.info("Working directory : {}".format(os.getcwd()))
    log.info(f"Hostname: {socket.gethostname()}")

    # Seeding
    seed = cfg.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    log.info(f"Random seed: {seed}")
    log.info(f"Device: {cfg.device}")

    # Setting up datastructure
    name = cfg.name
    try:
        init_datarepo(name, root_path=cfg.data_path)
    except:
        log.warn("Init of repo may failed. You typically can ignore this...")

    sweeper = cfg.sweeper
    objective = cfg.sweeper.objective

    # Does model already exist ?
    id = cfg.id
    idx = cfg.idx

    (
        id,
        idx,
        task_name,
        task_parameters,
        train_parameters,
        seed,
    ) = updated_task_train_params_if_idx_or_id_specified(name, idx, id, cfg, log)

    # Setting up tasks and training dataset.
    task_fn = get_task(task_name)
    task = task_fn(**task_parameters)
    input_dim = task.input_dim
    output_dim = task.output_dim
    log.info(f"The task {task_name} was selected to run.")
    log.info(f"The input_dim is {input_dim} and hte output_dim is {output_dim}.")
    log.info(f"The following parameters were used: {task_parameters}")

    sim_ident = dict(task_parameters)
    sim_ident["N_train"] = train_parameters["N_train"]
    sim_ident["N_test"] = train_parameters["N_test"]
    sim_ident["N_val"] = train_parameters["N_val"]
    sim_ident["seed"] = seed

    (
        train_loader,
        test_loader,
        validation_loader,
        simulation_time,
    ) = simulate_if_necessary(
        name, task, task_name, task_parameters, train_parameters, seed, cfg, log
    )

    # Train

    model, defense, id, idx, train_loss, val_loss, test_loss = load_or_train_model(
        name,
        id,
        idx,
        seed,
        task,
        task_name,
        simulation_time,
        input_dim,
        output_dim,
        train_loader,
        test_loader=test_loader,
        validation_loader=validation_loader,
        cfg=cfg,
        log=log,
    )

    # Evaluation part
    if cfg.run_eval:
        # Requires reseeding for some reason... probably loading vs training reverse seed
        set_seed(seed)

        if cfg.run_eval_approx:
            (
                approx_val,
                approx_val_additional,
                runtime_approx,
                approx_metrics,
                approx_metrics_params,
            ) = compute_clean_approximation_metric_if_necessary(
                name, id, task, model, test_loader, cfg, log
            )

        if cfg.run_eval_rob:
            log.info("\nStarting robustness evaluation")
            tolerance_levels = cfg.eval_rob.eps


            log.info(f"\nFollowing eps were selected {tolerance_levels}")
            if isinstance(tolerance_levels,float):
                tolerance_levels = [tolerance_levels]
            else:
                tolerance_levels = list(tolerance_levels)

            for eps in tolerance_levels:
                log.info(f"\nStarting evaluation for tolerance level eps={eps}")
                if isinstance(defense, PostHocDefense):
                # Disable for attack (This may should depend on which ...)
                    defense.deactivate()
                try:
                    (
                        id_adversarial,
                        rob_value,
                        x_unsecure,
                        theta_unsecure,
                        x_adversarial,
                        runtime_rob,
                    ) = eval_robustness_metric(name, model, id, task, test_loader, eps, cfg, log)

                    if isinstance(defense, PostHocDefense):
                    # Disable for attack (This may should depend on which ...)
                        defense.activate()

                    if cfg.run_eval_approx:
                        log.info("\nCompute approximaiton metrics for adversarial examples")
                        (
                            values_adversarial,
                            additional_values_adversarial,
                            values_unsecure,
                            additional_values_unsecure,
                            values_adversarial2x,
                            additional_values_adversarial2x,
                            runtime_approx_tilde,
                        ) = compute_adversarial_approximation_metric(
                            name,
                            task,
                            model,
                            approx_metrics,
                            x_unsecure,
                            theta_unsecure,
                            x_adversarial,
                            approx_metrics_params,
                            id_adversarial,
                            cfg,
                            log,
                        )
                        log.info("Finished computing evaluation metrics")
                except: 
                    rob_value = 100000000
                    values_adversarial = 1000000000000
        else:
            df_rob = query_rob_metric(name, id=id)
            id_adversarials = df_rob.id_adversarial.unique().tolist()

            for id_adversarial in id_adversarials:   
                x_unsecure, theta_unsecure, x_adversarial = get_adversarial_examples_by_id(name, id_adversarial)    
                log.info("\nCompute approximaiton metrics for adversarial examples")
                (
                    values_adversarial,
                    additional_values_adversarial,
                    values_unsecure,
                    additional_values_unsecure,
                    values_adversarial2x,
                    additional_values_adversarial2x,
                    runtime_approx_tilde,
                ) = compute_adversarial_approximation_metric(
                    name,
                    task,
                    model,
                    approx_metrics,
                    x_unsecure,
                    theta_unsecure,
                    x_adversarial,
                    approx_metrics_params,
                    id_adversarial,
                    cfg,
                    log,
                )
                log.info("Finished computing evaluation metrics")

    # The function outputs a objective for sweeping mode.
    objective = construct_objective(objective, locals())

    return objective


def construct_objective(objective, variables):
    if objective is None:
        return None
    elif isinstance(objective, str):
        return get_single_objective(objective, variables)
    elif isinstance(objective, Iterable):
        return_vals = []
        for obj in objective:
            return_vals.append(get_single_objective(obj, variables))
        return tuple(return_vals)
    else:
        raise ValueError("Unknown objectives")


def get_single_objective(objective, variables):
    if objective == "metric_rob":
        metric_rob_vals = variables["rob_value"]
        return float(metric_rob_vals)
    elif objective == "metric_approx_clean":
        metrix_approx_vals = variables["approx_val"]
        return 1 / len(metrix_approx_vals) * sum(metrix_approx_vals)
    elif objective == "metric_approx_tilde":
        metric_approx_tilde_vals = variables["values_adversarial"]
        return 1 / len(metric_approx_tilde_vals) * sum(metric_approx_tilde_vals)
    else:
        return float(variables[objective])


def set_seed(seed: int):
    """This methods just sets the seed."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def updated_task_train_params_if_idx_or_id_specified(
    name: str, idx: Optional[int], id: Optional[str], cfg: DictConfig, log: Logger
) -> Tuple[Optional[str], Optional[int], str, dict, dict, int]:
    """If an idx or id is specified and it exists in the dataset this means we don't have to train or simulate because this was already done!

    Hence in this case we update all train and task params to that of the id/idx.

    Args:
        name (str): Dateset name
        idx (Optional[int]): Index
        id (Optional[str]): Id
        cfg (DictConfig): Config
        log (Logger): Logger

    Raises:
        ValueError: If idx or id is invalid

    Returns:
        _type_: _description_
    """
    if not cfg.run_train or idx is not None:
        df = get_full_model_dataset(name)
        # Neagative index support

        # if isinstance(idx, int) and idx < 0:
        #     idx = len(df) -1 - idx
        
        # Check if exists in data
        if isinstance(idx, int):
            id = df.iloc[idx]["id"]
            log.info(f"Found model at idx with id {id}, using this instead")

            # Task parameters update
            task_all_parameters = eval(df.iloc[idx]["params_task"])
            task_name = df.iloc[idx]["task"]
            train_parameters = eval(df.iloc[idx]["params_train"])
            train_parameters["N_test"] = cfg.train.N_test    # This may be updated
            task_parameters = task_all_parameters["params"]

            # Seed update
            seed = df.iloc[idx]["seed"]
            set_seed(seed)
            log.info(f"Updated seed to {seed}")
            log.info(f"Recovered train params: {train_parameters}")
            log.info(f"Recovered task params: {task_parameters}")

        elif isinstance(id, str) and id in df.id.tolist():
            log.info(f"Found model with id {id}, using this instead")
            idx = int(df[df.id == id].index.tolist()[0])
            # Task parameters update
            task_all_parameters = eval(df.iloc[idx]["params_task"])
            task_name = df.iloc[idx]["task"]
            train_parameters = eval(df.iloc[idx]["params_train"])
            train_parameters["N_test"] = cfg.train.N_test    # This may be updated
            task_parameters = task_all_parameters["params"]

            # Seed update
            seed = df.iloc[idx]["seed"]
            set_seed(seed)
            log.info(f"Updated seed to {seed}")
            log.info(f"Recovered train params: {train_parameters}")
            log.info(f"Recovered task params: {task_parameters}")
        else:
            raise ValueError(
                "Idx or id not in dataset or wrong datatype, must be int or str!"
            )
    else:
        # Else use task specified in config
        seed = cfg.seed
        task_name = cfg.task.name
        task_parameters = cfg.task.params
        train_parameters = cfg.train

    return id, idx, task_name, task_parameters, train_parameters, seed


def simulate_if_necessary(
    name: str,
    task: Task,
    task_name: str,
    task_parameters: DictConfig,
    train_parameters: DictConfig,
    seed: int,
    cfg: DictConfig,
    log: Logger,
):
    """Simulates a dataset if necessary. It may already be on disk, then it will fetch it.

    Args:
        name (str): Name of dataset
        task (Task): Task
        task_name (str): Name of task
        task_parameters (DictConfig): Prams of task
        train_parameters (DictConfig): Params of train
        seed (int): Seed
        cfg (DictConfig): Config
        log (Logger): Logger

    Returns:
        _type_: _description_
    """
    sim_ident = dict(task_parameters)
    N_train = train_parameters["N_train"]
    N_test = train_parameters["N_test"]
    N_val = train_parameters["N_val"]
    sim_ident["N_train"] = N_train
    sim_ident["N_test"] = N_test
    sim_ident["N_val"] = N_val
    sim_ident["seed"] = seed
    # Simulating data
    if check_if_already_simulated(name, cfg.task.name, sim_ident):

        # We found cached simulation, we will use them instead.
        log.info("Detected cached simulation, using them instead")
        try:
            # Try to load
            (
                train_loader,
                test_loader,
                validation_loader,
                simulation_time,
            ) = get_simulations_by_id(name, cfg.task.name, sim_ident)
            train_loader.batch_sampler.batch_size = train_parameters["batch_size"]
            test_loader.batch_sampler.batch_size = train_parameters["batch_size"]
            validation_loader.batch_sampler.batch_size = train_parameters["batch_size"]
            simulation_time = None
        except:
            # If something unexpected happends then simulate...
            log.info("Simulating data")
            start_time = time.time()
            (
                train_loader,
                test_loader,
                validation_loader,
            ) = task.get_train_test_val_dataset(
                N_train,
                N_test,
                N_val,
                batch_size=train_parameters["batch_size"],
                shuffle=train_parameters["batch_size"],
                device=cfg.device,
            )
            simulation_time = time.time() - start_time
            log.info(
                f"Finished making {N_train} + {N_test} + {N_val} simulationgs in {simulation_time:.2f} seconds"
            )
    else:
        log.info("Simulating data")
        start_time = time.time()
        train_loader, test_loader, validation_loader = task.get_train_test_val_dataset(
            N_train,
            N_test,
            N_val,
            batch_size=train_parameters["batch_size"],
            shuffle=train_parameters["shuffle"],
            device=cfg.device,
        )
        simulation_time = time.time() - start_time
        log.info(
            f"Finished making {N_train} + {N_test} + {N_val} simulationgs in {simulation_time:.2f} seconds"
        )

        if cfg.store_simulations:
            save_simulations_by_id(
                train_loader,
                test_loader,
                validation_loader,
                simulation_time,
                name,
                task_name,
                sim_ident,
            )

        # Selecting kernel

    # Ensure seeding after torch.load
    set_seed(seed)

    return train_loader, test_loader, validation_loader, simulation_time


def load_or_train_model(
    name: str,
    id: Optional[str],
    idx: Optional[int],
    seed: int,
    task: Task,
    task_name: str,
    simulation_time: Optional[float],
    input_dim: int,
    output_dim: int,
    train_loader,
    validation_loader,
    test_loader,
    cfg: DictConfig,
    log: Logger,
):
    """This either loads an existing model or trains a new one.

    Args:
        name (str): Name of dataset
        id (Optional[str]): Id of model
        idx (Optional[int]): Index of model
        seed (int): Seed
        task (Task): Task to solve.
        task_name (str): Name of task.
        simulation_time (Optional[float]): Simulation time.
        input_dim (int): Input dimension
        output_dim (int): Output dimension
        train_loader (_type_): Train Loader
        validation_loader (_type_): Validation loader
        test_loader (_type_): Test loader
        cfg (DictConfig): Configuration.
        log (Logger): Logger

    Returns:
        Tuple: Train model, id, train-test-val loss.
    """
    if check_id_exists(name, id) or not cfg.run_train:
        # If we found a model previously, this is just loaded...
        log.info("A trained model already exists, loading and skipping training step")
        model = get_model_by_id(name, id)
        tr_loss, val_loss, test_loss, defense = None, None, None,None
        log.info(f"Loaded model: \n {model}")
    else:
        # Generating model
        log.info(f"The model {cfg.model.name} was selected to run.")
        module = importlib.import_module(cfg.model.module_path)
        model_class = getattr(module, cfg.model.class_name)
        hyper_parameters = dict(cfg.model.params)
        # Embedding nets
        embedding_cfg = cfg.model.embedding_net
        log.info(f"Embedding net seleected: {embedding_cfg}")
        embedding_net = get_embedding_net(embedding_cfg, input_dim)
        hyper_parameters["embedding_net"] = embedding_net

        # Mathching support if necessary
        support = task.get_prior().support
        hyper_parameters["output_transform"] = get_output_transform(
            hyper_parameters["output_transform"], support=support
        )

        # Get used nonlinearity
        if "nonlinearity" in hyper_parameters:
            nonlinearity = getattr(nn, hyper_parameters["nonlinearity"])
            hyper_parameters["nonlinearity"] = nonlinearity

        # Initialize model
        model = model_class(input_dim, output_dim, **hyper_parameters)
        log.info(f"Initialized model: {model}")

        # Run train
        log.info("Starting training:")
        loss_module = importlib.import_module(cfg.train.loss_module)
        defense_module = importlib.import_module(cfg.defense.defense_module)
        defense_name = cfg.defense.defense
        loss_class = getattr(loss_module, cfg.train.class_name)
        hyper_parameters_train = dict(cfg.train.params)
        hyper_parameters_defense = dict(cfg.defense.params)
        log.info(f"Selected loss: {loss_class.__name__}")
        log.info(f"Selected defense: {defense_name}")
        # Defense
        if cfg.defense.defense is not None:
            defense = getattr(defense_module, defense_name)
        else:
            defense = None
        # Optimizer
        hyper_parameters_train["optimizer"] = getattr(
            optim, hyper_parameters_train["optimizer"]
        )
        # Scheduler
        if hyper_parameters_train["lr_scheduler"] is not None:
            hyper_parameters_train["lr_scheduler"] = getattr(
                optim.lr_scheduler, hyper_parameters_train["lr_scheduler"]
            )

        # Start training
        start_time = time.time()
        model, defense, train_loss, validation_loss, test_loss = run_train(
            task,
            model,
            loss_class,
            train_loader,
            validation_loader=validation_loader,
            test_loader=test_loader,
            verbose=cfg.verbose,
            defense=defense,
            defense_hyper_parameters=hyper_parameters_defense,
            device=cfg.device,
            logger=log,
            **hyper_parameters_train,
        )
        end_time = time.time()
        train_time = end_time - start_time

        tr_loss = float(train_loss[-1])
        if validation_loss is not None:
            val_loss = float(validation_loss[-1])
        else:
            val_loss = None

        if test_loss is not None:
            test_loss = float(test_loss)
        else:
            test_loss = None
        # Storing train results
        if cfg.store_model:

            loss = cfg.train.class_name
            model_name = cfg.model.name

            id = add_new_entry(
                name,
                model,
                id=id,
                loss=loss,
                model_name=model_name,
                defense=defense_name,
                N_train=cfg.train.N_train,
                N_val=cfg.train.N_val,
                N_test=cfg.train.N_test,
                train_loss=tr_loss,
                validation_loss=val_loss,
                test_loss=test_loss,
                task=task_name,
                train_time=train_time,
                sim_time=simulation_time,
                seed=seed,
                params_model=dict(cfg.model),
                params_train=dict(cfg.train),
                params_task=dict(cfg.task),
                params_defense=dict(cfg.defense),
            )
    return model, defense, id, idx, tr_loss, val_loss, test_loss


def eval_robustness_metric(
    name: str,
    model: Union[ParametricProbabilisticModel, PyroFlowModel],
    id: Optional[str],
    task: Task,
    test_loader: DataLoader,
    eps: float,
    cfg: DictConfig,
    log: Logger,
):
    log.info("\nStarting robustness evaluation")

    # Get metrics and attack
    metric_kwargs = cfg.eval_rob.metric
    attack_kwargs = cfg.eval_rob.attack

    loss_module_attack = importlib.import_module(attack_kwargs.loss_module)
    attack_module = importlib.import_module(attack_kwargs.attack_module)
    metric_module = importlib.import_module(metric_kwargs.metric_module)

    attack = getattr(attack_module, attack_kwargs.attack_class)
    if attack_kwargs.attack_loss_fn is not None:
        attack_loss_fn = getattr(loss_module_attack, attack_kwargs.attack_loss_fn)
    else:
        attack_loss_fn = None

    # Get metric to evaluate
    rob_metrics = getattr(metric_module, metric_kwargs.metric_class)

    start_time = time.time()
    (
        metric_name,
        rob_value,
        rob_additional_values,
        x_unsecure,
        theta_unsecure,
        x_adversarial,
        c2st_x,
        eps_abs,
    ) = run_rob_eval(
        task=task,
        model=model,
        metric_class=rob_metrics,
        attack_class=attack,
        test_loader=test_loader,
        attack_loss_fn=attack_loss_fn,
        attack_attemps=cfg.eval_rob.attack_attemps,
        targeted=cfg.eval_rob.targeted,
        target_strategy=cfg.eval_rob.target_strategy,
        attack_mc_budget=attack_kwargs.attack_mc_budget,
        eval_mc_budget=metric_kwargs.eval_mc_budget,
        eps=eps,
        attack_hyper_parameters=attack_kwargs.params,
        metric_hyper_parameters=dict(cfg.eval_rob.metric.params),
        num_adversarial_examples=metric_kwargs.num_adversarial_examples,
        verbose=cfg.verbose,
        batch_size=cfg.eval_rob.batch_size,
        eps_relative_to_std=cfg.eval_rob.eps_relative_to_std,
        clip_min_max_automatic=cfg.eval_rob.clip_min_max_automatic,
        device=cfg.device,
    )
    end_time = time.time()
    eval_rob_time = end_time - start_time

    log.info(f"\nFinished robustness evaluation in {eval_rob_time} seconds")


    if cfg.store_metrics:
        id_adversarial = add_new_rob_metric_entry(
            name,
            id=id,
            metric_rob=metric_name,
            attack=attack.__name__,
            attack_loss_fn=attack_loss_fn.__name__ if attack_loss_fn is not None else None,
            attack_attemps=cfg.eval_rob.attack_attemps,
            eps=float(eps),
            eps_abs=float(eps_abs),
            main_value_rob=tensors_to_floats(rob_value),
            additional_value_rob=tensors_to_floats(rob_additional_values),
            target_strategy=cfg.eval_rob.target_strategy,
            x_unsecure=x_unsecure,
            theta_unsecure=theta_unsecure,
            x_adversarial=x_adversarial,
            params=dict(cfg.eval_rob),
            c2st_x_xtilde=float(c2st_x),
            eval_time=eval_rob_time,
        )
    else:
        id_adversarial = None

    return (
        id_adversarial,
        rob_value,
        x_unsecure,
        theta_unsecure,
        x_adversarial,
        eval_rob_time,
    )


def compute_clean_approximation_metric_if_necessary(
    name: str,
    id: str,
    task: Task,
    model: Union[ParametricProbabilisticModel, PyroFlowModel],
    test_loader: DataLoader,
    cfg: DictConfig,
    log: Logger,
):
    """Computes a approximation metric on clean data.

    Args:
        name (str): Name
        id (str): Id
        task (Task): task
        model (Union[ParametricProbabilisticModel, PyroFlowModel]): Model
        test_loader (DataLoader): Testloader
        cfg (DictConfig): Config
        log (Logger): Loger

    Returns:
        _type_: _description_
    """
    log.info("\nStarting approximaiton error evaluation")
    approx_metrics = [
        getattr(importlib.import_module(m.metric_module), m.metric_class)
        for m in cfg.eval_approx.metric.values()
    ]
    approx_metrics_params = [m.params for m in cfg.eval_approx.metric.values()]

    if not check_if_approximation_metric_already_computed(name, id):
        log.info("Computing approximation metric")

        (
            approx_metric_names,
            approx_val,
            approx_val_additional,
            runtimes,
        ) = run_approx_eval_clean(
            task,
            model,
            approx_metrics,
            test_loader,
            metric_params=approx_metrics_params,
            verbose=cfg.verbose,
            device=cfg.device,
        )

        if cfg.store_metrics:
            log.info("Saving approximation metrics")
            for approx_m, val, val2, t in zip(
                approx_metric_names, approx_val, approx_val_additional, runtimes
            ):
                add_new_approx_metric_clean_entry(
                    name,
                    id=id,
                    metric_approx_clean=approx_m,
                    main_value_approx_clean=tensors_to_floats(val),
                    additional_value_approx_clean=tensors_to_floats(val2),
                    eval_time=t,
                )
    else:
        approx_val = None
        approx_val_additional = None
        runtimes = None

    return (
        approx_val,
        approx_val_additional,
        runtimes,
        approx_metrics,
        approx_metrics_params,
    )


def compute_adversarial_approximation_metric(
    name: str,
    task: Task,
    model: Union[ParametricProbabilisticModel, PyroFlowModel],
    approx_metrics: list,
    x_unsecure: torch.Tensor,
    theta_unsecure: torch.Tensor,
    x_adversarial: torch.Tensor,
    approx_metrics_params: list,
    id_adversarial: str,
    cfg: DictConfig,
    log: Logger,
):
    """Compute adversarial approximation metric on adversarial examples.

    Args:
        name (str): Name
        task (Task): Task
        model (Union[ParametricProbabilisticModel, PyroFlowModel]): Model
        approx_metrics (list): Metrics
        x_unsecure (torch.Tensor): X on which adversarial examples where found
        theta_unsecure (torch.Tensor): Theta unsecure
        x_adversarial (torch.Tensor): Adversarial examples
        approx_metrics_params (list): Parameters for approximaiton metrics
        id_adversarial (str): Adversarial id.
        cfg (DictConfig): Configs
        log (Logger): Logger
    """
    log.info("\nCompute approximaiton metrics for adversarial examples")

    (
        approx_metric_names,
        values_adversarial,
        additional_values_adversarial,
        values_unsecure,
        additional_values_unsecure,
        values_adversarial2x,
        additional_values_adversarial2x,
        runtimes,
    ) = run_approx_eval_tilde(
        task,
        model,
        approx_metrics,
        x_unsecure=x_unsecure,
        theta_unsecure=theta_unsecure,
        x_adversarial=x_adversarial,
        metric_params=approx_metrics_params,
        verbose=cfg.verbose,
        device=cfg.device,
    )

    if cfg.store_metrics:
        log.info("Saving results")
        for (approx_m, val_a, val_a2, val_u, val_u2, val_ax, val_ax2,t,) in zip(
            approx_metric_names,
            values_adversarial,
            additional_values_adversarial,
            values_unsecure,
            additional_values_unsecure,
            values_adversarial2x,
            additional_values_adversarial2x,
            runtimes,
        ):
            add_new_approx_metric_tilde_entry(
                name,
                id_adversarial=id_adversarial,
                metric_approx_tilde=approx_m,
                main_value_approx_tilde=tensors_to_floats(val_a),
                additional_value_approx_tilde=tensors_to_floats(val_a2),
                main_value_approx_unsecure=tensors_to_floats(val_u),
                additional_value_approx_unsecure=tensors_to_floats(val_u2),
                main_value_tilde_to_x=tensors_to_floats(val_ax),
                additional_value_tilde_to_x=tensors_to_floats(val_ax2),
                eval_time=t,
            )

    return (
        values_adversarial,
        additional_values_adversarial,
        values_unsecure,
        additional_values_unsecure,
        values_adversarial2x,
        additional_values_adversarial2x,
        runtimes,
    )
