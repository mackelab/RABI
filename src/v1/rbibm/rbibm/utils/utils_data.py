import base64
import hashlib
import math
import os
import uuid
from time import sleep
from typing import Any, Optional, Tuple

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

# Naming constants--------------------------------------------------------------------------------------------------------------------------------------


SEP = os.path.sep
ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
DATA_PATH = ROOT_PATH + SEP + "data" + SEP

with open(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    + SEP
    + "config"
    + SEP
    + "config.yaml",
    "r",
) as f:
    config = yaml.safe_load(f)
    if config["data_path"] is not None:
        ROOT_PATH = config["data_path"]
        DATA_PATH = ROOT_PATH + SEP + "data" + SEP

DATASET_NAME = "dataset.csv"
SWEEP_NAME = "sweep.csv"
ROB_METRIC_DATASET_NAME = "metrics_rob.csv"
APPROX_METRIC_DATASET_NAME_CLEAN = "metrics_approx_clean.csv"
APPROX_METRIC_DATASET_NAME_TILDE = "metrics_approx_tilde.csv"

SWEEP_STRUC = {
    "objective": [],
    "sweeper": [],
    "direction": [],
    "search_space": [],
    "overrides": [],
    "best_params": [],
    "best_value": [],
}
DATASET_STRUC = {
    "id": [],
    "task": [],
    "loss": [],
    "defense": [],
    "model_name": [],
    "N_train": [],
    "N_val": [],
    "N_test": [],
    "train_loss": [],
    "validation_loss": [],
    "test_loss": [],
    "train_time": [],
    "sim_time": [],
    "seed": [],
    "params_model": [],
    "params_train": [],
    "params_task": [],
    "params_defense": [],
}
ROB_EVAL_DATASET_STRUC = {
    "id": [],
    "metric_rob": [],
    "attack": [],
    "attack_loss_fn": [],
    "attack_attemps": [],
    "target_strategy": [],
    "eps": [],
    "eps_abs": [],
    "main_value_rob": [],
    "additional_value_rob": [],
    "id_adversarial": [],
    "c2st_x_xtilde": [],
    "eval_time": [],
    "params": [],
}
APPROX_EVAL_DATASET_STRUC_CLEAN = {
    "id": [],
    "metric_approx_clean": [],
    "main_value_approx_clean": [],
    "additional_value_approx_clean": [],
    "eval_time": [],
}
APPROX_EVAL_DATASET_STRUC_TILDE = {
    "id_adversarial": [],
    "metric_approx_tilde": [],
    "main_value_approx_tilde": [],
    "additional_value_approx_tilde": [],
    "main_value_approx_unsecure": [],
    "additional_value_approx_unsecure": [],
    "main_value_tilde_to_x": [],
    "additional_value_tilde_to_x": [],
    "eval_time": [],
}
MODEL_FOLDER = "models"
EVAL_FOLDER = "eval"
SIMULATION_FOLDER = "simulations"
FIGURE_FOLDER = "figures"


# ------------------------------------------------------------------------------------------------------------------------------------------------------


def init_datarepo(name: str, root_path: Optional[str] = None):
    """This functions initializes the data folder and all files within it.

    Args:
        name (str): Name of the benchmark, will be the name of the folder
        root_path (Optional[str], optional): Data folder path. Defaults to None.
    """
    if root_path is not None:
        global ROOT_PATH, DATA_PATH
        ROOT_PATH = root_path
        DATA_PATH = ROOT_PATH + SEP + "data" + SEP
    path = DATA_PATH + name

    # Create data folder if necessary
    if not os.path.exists(DATA_PATH[:-1]):
        os.mkdir(DATA_PATH[:-1])

    # Create dataset
    if not os.path.exists(path):
        os.mkdir(path)

    # Create dataset csv table
    if not os.path.exists(path + SEP + DATASET_NAME):
        df = pd.DataFrame(DATASET_STRUC)
        with open(path + SEP + DATASET_NAME, "w") as f:
            df.to_csv(f, header=f.tell() == 0, index=False)

    # Sweep dataset
    if not os.path.exists(path + SEP + SWEEP_NAME):
        df = pd.DataFrame(SWEEP_STRUC)
        with open(path + SEP + SWEEP_NAME, "w") as f:
            df.to_csv(f, header=f.tell() == 0, index=False)

    # Create model and figure folder
    if not os.path.exists(path + SEP + MODEL_FOLDER):
        os.mkdir(path + SEP + MODEL_FOLDER)

    if not os.path.exists(path + SEP + FIGURE_FOLDER):
        os.mkdir(path + SEP + FIGURE_FOLDER)

    if not os.path.exists(path + SEP + EVAL_FOLDER):
        os.mkdir(path + SEP + EVAL_FOLDER)

    if not os.path.exists(path + SEP + SIMULATION_FOLDER):
        os.mkdir(path + SEP + SIMULATION_FOLDER)

    # Create metric csv table
    if not os.path.exists(path + SEP + EVAL_FOLDER + SEP + ROB_METRIC_DATASET_NAME):
        df = pd.DataFrame(ROB_EVAL_DATASET_STRUC)
        with open(path + SEP + EVAL_FOLDER + SEP + ROB_METRIC_DATASET_NAME, "w") as f:
            df.to_csv(f, header=f.tell() == 0, index=False)

    if not os.path.exists(
        path + SEP + EVAL_FOLDER + SEP + APPROX_METRIC_DATASET_NAME_CLEAN
    ):
        df = pd.DataFrame(APPROX_EVAL_DATASET_STRUC_CLEAN)
        with open(
            path + SEP + EVAL_FOLDER + SEP + APPROX_METRIC_DATASET_NAME_CLEAN, "w"
        ) as f:
            df.to_csv(f, header=f.tell() == 0, index=False)

    if not os.path.exists(
        path + SEP + EVAL_FOLDER + SEP + APPROX_METRIC_DATASET_NAME_TILDE
    ):
        df = pd.DataFrame(APPROX_EVAL_DATASET_STRUC_TILDE)
        with open(
            path + SEP + EVAL_FOLDER + SEP + APPROX_METRIC_DATASET_NAME_TILDE, "w"
        ) as f:
            df.to_csv(f, header=f.tell() == 0, index=False)


def check_id_exists(name: str, id: str) -> bool:
    """Checks if id exists in a benchmark

    Args:
        name (str): Name of the benchmark to check
        id (str): Id to check

    Returns:
        bool: True if the id exists, else false.
    """

    df = get_full_model_dataset(name)
    if id is None:
        return False
    else:
        return id in df["id"].to_list()


def check_if_already_simulated(name: str, task_name: str, params_dict: dict) -> bool:
    """Checks if we already made these simulations.

    Args:
        name (str): Name of the benchmarks
        task_name (str): Name of the task
        params_dict (dict): Parameters of the task.

    Returns:
        bool: True if simulations exists, else false.
    """
    id = generate_task_hash_id(params_dict)
    path = (
        DATA_PATH + name + SEP + SIMULATION_FOLDER + SEP + task_name + "_" + id + ".pkl"
    )
    print(path)
    return os.path.exists(path)


def check_if_approximation_metric_already_computed(name: str, id: str) -> bool:
    """Checks if a certain approximation metric is already computed.

    Args:
        name (str): Name of the benchmark.
        id (str): Id of the model.

    Returns:
        bool: True if approximatmion metric is already there.
    """
    df = get_full_approx_metric_clean_dataset(name)
    print(id)
    return id in df.id.tolist()


def generate_task_hash_id(params: dict) -> str:
    """Generates an unique id based on the parameeters of a task.

    Args:
        params (dict): Parmeters of the task

    Returns:
        str: Id of task
    """
    long_string = str(params)
    shorter = (
        base64.b32encode(hashlib.sha256(long_string.encode()).digest())
        .decode()
        .strip("=")
    )
    return shorter


def generate_unique_id(name: str, id: Optional[str] = None) -> str:
    """Generates a unique, random id.

    Args:
        name (str): Name of the benchmark.
        id (Optional[str], optional): Fixed id used instead. Defaults to None.

    Returns:
        str: Id
    """
    df = get_full_model_dataset(name)
    if id is None or id in df["id"].to_list():
        id = str(uuid.uuid4())
        while id in df["id"]:
            id = str(uuid.uuid4())
        return id
    else:
        return id


def generate_unique_id_adversarial_examples(name: str) -> str:
    """Generates a unique, random id for adversarial examples.

    Args:
        name (str): Name of the benchmark.

    Returns:
        str: Id
    """
    df = get_full_rob_metric_dataset(name)
    id = str(uuid.uuid4())
    while id in df["id_adversarial"]:
        id = str(uuid.uuid4())
    return id


def update_entry(name: str, id: str, **kwargs) -> None:
    """Updates the model dataset.

    Args:
        name (str): Name of the benchmark.
        id (str): Id of the entry that should be updated.
    """

    df = get_full_model_dataset(name)
    entry = df[df["id"] == id]
    for key, val in kwargs.items():
        if key in entry:
            entry[key] = [val]
    df[df["id"] == id] = entry

    path = DATA_PATH + name
    with open(path + SEP + DATASET_NAME, "w") as f:
        df.to_csv(header=f.tell() == 0, index=False)

def remove_entry_by_id(name:str, id: str):
    """Removes the entry and associated files form the dataset.

    Args:
        name (str): Name of the benchmark
        id (str): Id of the model.
    """
    df = query(name)
    ids = df.id.tolist()


    if id in ids:
        df_to_remove = df[df.id == id]
        id_adversarials = df_to_remove.id_adversarial.dropna().unique().tolist()

        df_model = get_full_model_dataset(name)
        df_approx_clean = get_full_approx_metric_clean_dataset(name)
        df_rob = get_full_rob_metric_dataset(name)
        df_aprrox_tilde = get_full_approx_metric_tilde_dataset(name)


        df_model = df_model[df_model.id != id]
        df_approx_clean = df_approx_clean[df_approx_clean.id != id]
        df_rob = df_rob[df_rob.id != id]
        for id_a in id_adversarials:
            df_aprrox_tilde = df_aprrox_tilde[df_aprrox_tilde.id_adversarial != id_a]

        # Updates files
        update_model_dataset(name, df_model)
        delete_model_by_id(name, id)
        update_rob_metric_dataset(name, df_rob)
        for id_a in id_adversarials:
            delete_adversarial_examples_by_id(name,id_a)
        update_approx_metric_clean_dataset(name,df_approx_clean)
        if len(id_adversarials) > 0:
            update_approx_metric_tilde_dataset(name, df_aprrox_tilde)
    else:
        df = query_main(name)
        ids = df.id.tolist()
        if id in ids:
            df_model = get_full_model_dataset(name)
            df_model = df_model[df_model.id != id]
            update_model_dataset(name, df_model)
            delete_model_by_id(name, id)


    


def add_new_entry(name: str, model: Any, id: Optional[str] = None, **kwargs) -> str:
    """Adds new entry to model dataset.

    Args:
        name (str): Name of the benchmark.
        model (Any): Model to save and add an entry.
        id (Optional[str], optional): Id of the model, if None a unique one is generated. Defaults to None.

    Returns:
        str: Return the generated unique id.
    """
    path = DATA_PATH + name
    id = generate_unique_id(name, id)
    entry = DATASET_STRUC.copy()
    for key in entry:
        entry[key] = [pd.NA]

    entry["id"] = id  # type: ignore

    for key, val in kwargs.items():
        if key in entry:
            entry[key] = [val]

    df = pd.DataFrame(entry)

    save_model_by_id(model, name, id)

    for i in range(10):
        try:
            with open(path + SEP + DATASET_NAME, "a") as f:
                df.to_csv(f, mode="a", header=f.tell() == 0, index=False)
                break
        except:
            print("Fauled to save, try again")
            sleep(1)

    return id


def add_new_sweep(name: str, **kwargs) -> None:
    """Adds a new sweep entry.

    Args:
        name (str): Name of the benchmark.
    """
    path = DATA_PATH + name
    entry = SWEEP_STRUC.copy()
    for key in entry:
        entry[key] = [pd.NA]

    for key, val in kwargs.items():
        if key in entry:
            entry[key] = [val]

    df = pd.DataFrame(entry)

    for i in range(10):
        try:
            with open(path + SEP + SWEEP_NAME, "a") as f:
                df.to_csv(f, mode="a", header=f.tell() == 0, index=False)
                break 
        except:
            print("Fauled to save, try again")
            sleep(1)
        


def add_new_rob_metric_entry(name: str, **kwargs) -> str:
    """Adds a new rob. metric entry.

    Args:
        name (str): Name of the benchmark.

    Returns:
        str: Id of adversarial example.
    """
    path = DATA_PATH + name + SEP + EVAL_FOLDER
    entry = ROB_EVAL_DATASET_STRUC.copy()

    for key in entry:
        entry[key] = [pd.NA]

    for key, val in kwargs.items():
        if key in entry:
            entry[key] = [val]

    id_adversarial = generate_unique_id_adversarial_examples(name)
    x_adversarial = kwargs.get("x_adversarial", None)
    x_unsecure = kwargs.get("x_unsecure", None)
    theta_unsecure = kwargs.get("theta_unsecure", None)
    save_adversarial_examples_by_id(
        x_unsecure, theta_unsecure, x_adversarial, name=name, id=id_adversarial
    )
    entry["id_adversarial"] = id_adversarial  # type: ignore

    df = pd.DataFrame(entry)

    for i in range(10):
        try:
            with open(path + SEP + ROB_METRIC_DATASET_NAME, "a") as f:
                df.to_csv(f, mode="a", header=f.tell() == 0, index=False)
                break
        except:
            print("Fauled to save, try again")
            sleep(1)

    return id_adversarial


def add_new_approx_metric_clean_entry(name: str, **kwargs):
    """Adds new approx. metric entry.

    Args:
        name (str): Name of the benchmark.
    """
    path = DATA_PATH + name + SEP + EVAL_FOLDER
    entry = APPROX_EVAL_DATASET_STRUC_CLEAN.copy()

    for key in entry:
        entry[key] = [pd.NA]

    for key, val in kwargs.items():
        if key in entry:
            entry[key] = [val]

    df = pd.DataFrame(entry)
    with open(path + SEP + APPROX_METRIC_DATASET_NAME_CLEAN, "a") as f:
        df.to_csv(f, mode="a", header=f.tell() == 0, index=False)


def add_new_approx_metric_tilde_entry(name: str, **kwargs):
    """Adds new rob. approx. metric entry.

    Args:
        name (str): Name of the benchmark.
    """

    path = DATA_PATH + name + SEP + EVAL_FOLDER
    entry = APPROX_EVAL_DATASET_STRUC_TILDE.copy()

    for key in entry:
        entry[key] = [pd.NA]

    for key, val in kwargs.items():
        if key in entry:
            entry[key] = [val]

    df = pd.DataFrame(entry)

    for i in range(10):
        try:
            with open(path + SEP + APPROX_METRIC_DATASET_NAME_TILDE, "a") as f:
                df.to_csv(f, mode="a", header=f.tell() == 0, index=False)
                break 
        except:
            print("Fauled to save, try again")
            sleep(1)


def save_adversarial_examples_by_id(
    xs: Optional[torch.Tensor],
    thetas: Optional[torch.Tensor],
    xs_tilde: Optional[torch.Tensor],
    name: str,
    id: str,
):
    """Saves adversarial examples by id.

    Args:
        xs (torch.Tensor): Clean data
        thetas (torch.Tensor): True parameters.
        xs_tilde (torch.Tensor): Attacked data.
        name (str): Name of benchmark.
        id (str): Id of attack.
    """
    dictionary = {"xs": xs, "thetas": thetas, "xs_tilde": xs_tilde}
    path = DATA_PATH + name + SEP + EVAL_FOLDER + SEP + id + ".pkl"
    
    for i in range(10):
        try:
            torch.save(dictionary, path)
            break
        except:
            print("Fauled to save, try again")
            sleep(1)

def delete_adversarial_examples_by_id(name:str, id: str):
    path = DATA_PATH + name + SEP + EVAL_FOLDER + SEP + id + ".pkl"
    if os.path.exists(path):
        os.remove(path)


def get_adversarial_examples_by_id(name: str, id: str) -> Tuple:
    """Loads adverarial examples by id.

    Args:
        name (str): Name of the benchmark.
        id (str): Id of the attack.

    Returns:
        Tuple: Clean data, True parameters, Attacked data.
    """
    path = DATA_PATH + name + SEP + EVAL_FOLDER + SEP + id + ".pkl"
    dictionary = torch.load(path, map_location=torch.device("cpu"))
    return dictionary["xs"], dictionary["thetas"], dictionary["xs_tilde"]


def get_model_by_idx(name: str, idx: int):
    """Loads a model by idx.

    Args:
        name (str): Name of the benchmark.
        idx (int): Index of model

    Raises:
        ValueError: Out of index bounds.

    Returns:
        Model: Model with index=idx in the database.
    """
    df = get_full_model_dataset(name)
    if idx in df.index:
        id = df.iloc[idx]["id"]
        return get_model_by_id(name, id)
    else:
        raise ValueError("Invalid index")


def get_model_by_id(name: str, id: str):
    """Loads a model by id.

    Args:
        name (str): Name of the benchmark
        id (str): Id of the model

    Returns:
        Model: Modle with id=id.
    """
    path = DATA_PATH + name + SEP + MODEL_FOLDER + SEP + id + ".pkl"
    model = torch.load(path, map_location=torch.device("cpu"))
    return model


def get_simulations_by_id(name: str, task_name: str, params_dict: dict) -> Tuple:
    """Gets simulations by id.

    Args:
        name (str): Name of the benchmark.
        task_name (str): Name of the task
        params_dict (dict): Parameter dict.

    Returns:
        Tuple: Simulations.
    """
    id = generate_task_hash_id(params_dict)
    path = (
        DATA_PATH + name + SEP + SIMULATION_FOLDER + SEP + task_name + "_" + id + ".pkl"
    )
    dictionary = torch.load(path, map_location=torch.device("cpu"))
    return (
        dictionary["train_loader"],
        dictionary["test_loader"],
        dictionary["validation_loader"],
        dictionary["simulation_time"],
    )


def save_simulations_by_id(
    train_loader: DataLoader,
    test_loader: DataLoader,
    validation_loader: DataLoader,
    simulation_time: float,
    name: str,
    task_name: str,
    params_dict: dict,
):
    """Saves simulation with a certain id.

    Args:
        train_loader (DataLoader): Training dataset.
        test_loader (DataLoader): Testing dataset.
        validation_loader (DataLoader): Validation dataset.
        simulation_time (float): Simulation time.
        name (str): Name of benchmark.
        task_name (str): Name of atsk.
        params_dict (dict): Task parameters.
    """
    id = generate_task_hash_id(params_dict)
    path = (
        DATA_PATH + name + SEP + SIMULATION_FOLDER + SEP + task_name + "_" + id + ".pkl"
    )
    dictionary = {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "validation_loader": validation_loader,
        "simulation_time": simulation_time,
    }

    for i in range(10):
        try:
            torch.save(dictionary, path)
            break 
        except:
            print("Fauled to save, try again")
            sleep(1)


def save_model_by_id(model: Any, name: str, id: str) -> None:
    """Saves the model by id.

    Args:
        model (Any): Saves a model in a file named by it's id.
        name (str): Name of the benchmark.
        id (str): Id of the model.
    """
    path = DATA_PATH + name + SEP + MODEL_FOLDER + SEP + id + ".pkl"

    for i in range(10):
        try:
            torch.save(model.to("cpu"), path)
            break
        except:
            print("Fauled to save, try again")
            sleep(1)
            

def delete_model_by_id(name: str , id: str) -> None:
    """Deletes the model

    Args:
        name (str): Name of the benchmark.
        id (str): Id of the run.
    """
    path = DATA_PATH + name + SEP + MODEL_FOLDER + SEP + id + ".pkl"
    if os.path.exists(path):
        os.remove(path)


def to_query_string(name: str, var: Any) -> str:
    """Translates a variable to string.

    Args:
        name (str): Query argument
        var (str): value

    Returns:
        str: Query == value ?
    """
    if var is None:
        return ""
    elif var is pd.NA or var is torch.nan or var is math.nan or str(var) == "nan":
        return f"{name}!={name}"
    elif isinstance(var, list) or isinstance(var, tuple):
        query = "("
        for v in var:
            if query != "(":
                query += "|"
            if isinstance(v, str):
                query += f"{name}=='{v}'"
            else:
                query += f"{name}=={v}"
        query += ")"
    else:
        if isinstance(var, str):
            query = f"{name}=='{var}'"
        else:
            query = f"{name}=={var}"
    return query


def append_to_query_with_and(query: str, arg: str) -> str:
    """Appends two conditions together with logical AND.

    Args:
        query (str): First condition
        arg (str): Second condition

    Returns:
        str: First and second condition
    """
    if query != "" and arg != "":
        query += "&"

    query += arg
    return query


def query(name: str, *args, **kwargs) -> pd.DataFrame:
    """Queries all datasets.

    Args:
        name (str): Name of benchmark.

    Returns:
        pd.DataFrame: Queried datsets
    """

    df_main = query_main(name, *args, **kwargs)
    df_approx_clean = query_approx_metric_clean(name, *args, **kwargs)
    df_rob = query_rob_metric(name, *args, **kwargs)
    df_approx_rob = query_approx_metric_tilde(name, *args, **kwargs)

    df_all = merge_all(df_main, df_approx_clean, df_rob, df_approx_rob, how="inner")

    # for key in kwargs:
    #     if key in df_all.columns:
    #         if key != "defense":
    #             df_all = df_all[df_all[key].notna()]

    return df_all


def query_main(
    name: str,
    id: Optional[str] = None,
    task: Optional[str] = None,
    defense: Optional[str] = None,
    model_name: Optional[str] = None,
    loss: Optional[str] = None,
    N_train: Optional[int] = None,
    N_val: Optional[int] = None,
    N_test: Optional[int] = None,
    df: Optional[pd.DataFrame] = None,
    *arg,
    **kwargs,
) -> pd.DataFrame:
    """Queries the model dataset.

    Args:
        name (str): Name of the benchmark
        id (Optional[str], optional): Id of the model. Defaults to None.
        task (Optional[str], optional): Name of the task. Defaults to None.
        defense (Optional[str], optional): Class of defense. Defaults to None.
        model_name (Optional[str], optional): Class of model. Defaults to None.
        N_train (Optional[int], optional): Number of training simulations. Defaults to None.
        N_val (Optional[int], optional): Number of validation simulations. Defaults to None.
        N_test (Optional[int], optional): Number of testing simulations. Defaults to None.
        df (Optional[pd.DataFrame], optional): Dataset used, else taken from disk. Defaults to None.

    Returns:
        pd.DataFrame: Queried dataset.
    """
    if df is None:
        df = get_full_model_dataset(name)
        df["defense"] = df["defense"].fillna("None")

    args = locals()
    del args["name"]
    del args["df"]
    del args["kwargs"]
    del args["arg"]

    query = ""
    for name, var in args.items():
        var_q = to_query_string(name, var)
        query = append_to_query_with_and(query, var_q)

    if query != "":
        df_q = df.query(query)
    else:
        df_q = df

   
    if kwargs != {}:
        additional_params = [
            "params_model",
            "params_train",
            "params_task",
            "params_defense",
        ]

        if "expand_params_defense" in kwargs:
            expand_df = kwargs["expand_params_defense"]
            del kwargs["expand_params_defense"]
        else:
            expand_df = False

        for key, val in kwargs.items():
            for additional_param in additional_params:
                if additional_param in key:
                    parameters = [eval(d)["params"] for d in df_q[additional_param].tolist()]
                    param_key = key.split("_")[-1]
                    mask = []
                    for params in parameters:
                            if param_key in params:
                                val_q = params[param_key]
                                mask.append(val == val_q)
                            else:
                                mask.append(False)
                    df_q = df_q[mask]

        if expand_df:
            new_columns = {}
            parameters = [eval(d)["params"] for d in df_q["params_defense"].tolist()]

            for i,param in enumerate(parameters):
                for key, val in param.items():
                    item = new_columns.get("params_defense_" + key, [None]*len(parameters))
                    item[i] = val 
                    new_columns["params_defense_" + key] = item
            pd.options.mode.chained_assignment = None 
    
            for k,v in new_columns.items():
                df_q[k] = v

            
        

    # Drop params columns
    df_q = df_q.drop(
        ["params_model", "params_train", "params_task", "params_defense"], axis=1
    )

    return df_q


def query_rob_metric(
    name: str,
    id: Optional[str] = None,
    metric_rob: Optional[str] = None,
    attack: Optional[str] = None,
    attack_loss_fn: Optional[str] = None,
    target_strategy: Optional[str] = None,
    eps: Optional[float] = None,
    attack_attemps: Optional[int] = None,
    df: Optional[pd.DataFrame] = None,
    *arg,
    **kwargs,
) -> pd.DataFrame:
    """Queries the rob metric dataset.

    Args:
        name (str): Name of the benchmark.
        id (Optional[str], optional): Id of the model. Defaults to None.
        metric_rob (Optional[str], optional): Robustness metric type. Defaults to None.
        attack (Optional[str], optional): Attack. Defaults to None.
        attack_loss_fn (Optional[str], optional): Attack loss_fn. Defaults to None.
        eps (Optional[float], optional): Attack tolerance. Defaults to None.
        attack_attemps (Optional[int], optional): Attack attemps i.e restarts. Defaults to None.
        df (Optional[pd.DataFrame], optional): Dataframe used, else taken from disk. Defaults to None.

    Returns:
        pd.DataFrame: _description_
    """

    if df is None:
        df = get_full_rob_metric_dataset(name)

    args = locals()
    del args["name"]
    del args["df"]
    del args["kwargs"]
    del args["arg"]

    query = ""
    for name, var in args.items():
        var_q = to_query_string(name, var)
        query = append_to_query_with_and(query, var_q)

    if query != "":
        df_q = df.query(query)
    else:
        df_q = df

    if kwargs != {}:
        additional_params = [
            "params_attack",
            "params_metric",
        ]

        for key, val in kwargs.items():
            for additional_param in additional_params:
                if additional_param in key:
                    parameters_params = [eval(d)[additional_param.split("_")[-1]]["params"] for d in df_q["params"].tolist()]
                    parameters = [eval(d)[additional_param.split("_")[-1]] for d in df_q["params"].tolist()]
                    param_key = "_".join(key.split("_")[2:])
                    mask = []
                    for params1, params2 in zip(parameters, parameters_params):
                            if param_key in params1:
                                val_q = params1[param_key]
                                mask.append(val == val_q)
                            elif param_key in params2:
                                val_q = params2[param_key]
                                mask.append(val == val_q)
                            else:
                                mask.append(False)
                    df_q = df_q[mask]

    # Drop params columns
    df_q = df_q.drop("params", axis=1)
    return df_q


def query_approx_metric_clean(
    name: str,
    id: Optional[str] = None,
    metric_approx_clean: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    *arg,
    **kwargs,
) -> pd.DataFrame:
    """Queries a approx metric.

    Args:
        name (str): Benchmark name.
        id_adversarial (Optional[str], optional): Adversarial id. Defaults to None.
        metric_approx_clean (Optional[str], optional): Approx metric type. Defaults to None.
        df (Optional[pd.DataFrame], optional): Dataset, else it is taken from disk. Defaults to None.

    Returns:
        pd.DataFrame: Filtered dataset, according to queries.
    """
    if df is None:
        df = get_full_approx_metric_clean_dataset(name)

    args = locals()
    del args["name"]
    del args["df"]
    del args["kwargs"]
    del args["arg"]

    query = ""
    for name, var in args.items():
        var_q = to_query_string(name, var)
        query = append_to_query_with_and(query, var_q)

    if query != "":
        df_q = df.query(query)
    else:
        df_q = df

    return df_q


def query_approx_metric_tilde(
    name: str,
    id_adversarial: Optional[str] = None,
    metric_approx_tilde: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    *arg,
    **kwargs,
) -> pd.DataFrame:
    """Queries a approx metric.

    Args:
        name (str): Benchmark name.
        id_adversarial (str, optional): Adversarial id. Defaults to None.
        metric_approx_tilde (str, optional): Approx metric type. Defaults to None.
        df (Optional[pd.DataFrame], optional): Dataset, else it is taken from disk. Defaults to None.

    Returns:
        pd.DataFrame: Filtered dataset, according to queries.
    """

    if df is None:
        df = get_full_approx_metric_tilde_dataset(name)

    args = locals()
    del args["name"]
    del args["df"]
    del args["kwargs"]
    del args["arg"]

    query = ""
    for name, var in args.items():
        var_q = to_query_string(name, var)
        query = append_to_query_with_and(query, var_q)

    if query != "":
        df_q = df.query(query)
    else:
        df_q = df

    return df_q


def merge_all(
    df_main: pd.DataFrame,
    df_approx_clean: pd.DataFrame,
    df_rob: pd.DataFrame,
    df_approx_rob: pd.DataFrame,
    how:str="outer"
) -> pd.DataFrame:
    """Merges the three datasets together...

    Args:
        df_main (pd.DataFrame): Model dataset.
        df_approx_clean (pd.DataFrame): Approximation metrics dataset.
        df_rob (pd.DataFrame): Robustness metrics dataset.
        df_approx_rob (pd.DataFrame): Rob. approximaition dataset.

    Returns:
        pd.DataFrame: Merged data.
    """
    df1 = pd.merge(
        df_main,
        df_approx_clean,
        on="id",
        suffixes=["_model", "_clean_approx"],
        how=how,
    )
    df2 = pd.merge(df1, df_rob, on="id", suffixes=["", "_rob"], how=how)
    df_all = pd.merge(
        df2,
        df_approx_rob,
        on="id_adversarial",
        suffixes=["", "_rob_approx"],
        how=how,
    )
    df_all = df_all.drop_duplicates().dropna(axis=0, how="all").dropna(subset=["id"])
    df_smaller_all = df_all.drop("defense", axis=1)

    df_smaller_all.insert(3, "defense", df_all["defense"].tolist())
    return df_smaller_all


def get_full_dataset(name: str) -> pd.DataFrame:
    """Returns the raw csv of all merged datasets, as pandas DataFrame.

    Args:
        name (str): Name of the benchmark.

    Returns:
        pd.DataFrame: Dataset
    """
    df_main = get_full_model_dataset(name)
    df_approx_clean = get_full_approx_metric_clean_dataset(name)
    df_rob = get_full_rob_metric_dataset(name)
    df_approx_rob = get_full_approx_metric_tilde_dataset(name)

    df_all = merge_all(df_main, df_approx_clean, df_rob, df_approx_rob)
    return df_all


def get_full_model_dataset(name: str) -> pd.DataFrame:
    """Returns the raw csv of models, as pandas DataFrame.

    Args:
        name (str): Name of the benchmark.

    Returns:
        pd.DataFrame: Dataset
    """
    path = DATA_PATH + name + SEP
    df = pd.read_csv(path + DATASET_NAME,on_bad_lines="warn")
    return df

def update_model_dataset(name:str, df: pd.DataFrame):
    """Overwrites the model dataset csv.

    Args:
        name (str): Name of the benchmark.
        df (pd.DataFrame): Dataframe.
    """

    assert all(df.columns == pd.Series(DATASET_STRUC.keys())), "Your dataframe has the wrong format"

    path = DATA_PATH + name + SEP
    with open(path + DATASET_NAME, "w") as f:
        df.to_csv(f, index=False)


def get_full_rob_metric_dataset(name: str) -> pd.DataFrame:
    """Returns the raw csv of robustness metrics, as pandas DataFrame.

    Args:
        name (str): Name of the benchmark.

    Returns:
        pd.DataFrame: Dataset
    """
    path = DATA_PATH + name + SEP + EVAL_FOLDER + SEP
    df = pd.read_csv(path + ROB_METRIC_DATASET_NAME,on_bad_lines="warn")
    return df

def update_rob_metric_dataset(name:str, df: pd.DataFrame):
    """Overwrites the rob metric dataset csv.

    Args:
        name (str): Name of the benchmark.
        df (pd.DataFrame): Dataframe.
    """

    assert all(df.columns == pd.Series(ROB_EVAL_DATASET_STRUC.keys())), "Your dataframe has the wrong format"

    path = DATA_PATH + name + SEP + EVAL_FOLDER + SEP
    with open(path + ROB_METRIC_DATASET_NAME, "w") as f:
        df.to_csv(f, index=False)


def get_full_approx_metric_tilde_dataset(name: str) -> pd.DataFrame:
    """Returns the raw csv of perturbed approximation metrics, as pandas DataFrame.

    Args:
        name (str): Name of the benchmark.

    Returns:
        pd.DataFrame: Dataset
    """
    path = DATA_PATH + name + SEP + EVAL_FOLDER + SEP
    df = pd.read_csv(path + APPROX_METRIC_DATASET_NAME_TILDE,on_bad_lines="warn")
    return df

def update_approx_metric_tilde_dataset(name:str, df: pd.DataFrame):
    """Overwrites the approx metric tilde dataset csv.

    Args:
        name (str): Name of the benchmark.
        df (pd.DataFrame): Dataframe.
    """

    assert all(df.columns == pd.Series(APPROX_EVAL_DATASET_STRUC_TILDE.keys())), "Your dataframe has the wrong format"

    path = DATA_PATH + name + SEP + EVAL_FOLDER + SEP
    with open(path + APPROX_METRIC_DATASET_NAME_TILDE, "w") as f:
        df.to_csv(f, index=False)


def get_full_approx_metric_clean_dataset(name: str) -> pd.DataFrame:
    """Returns the raw csv of approximation metrics, as pandas DataFrame.

    Args:
        name (str): Name of the benchmark.

    Returns:
        pd.DataFrame: Dataset
    """
    path = DATA_PATH + name + SEP + EVAL_FOLDER + SEP
    df = pd.read_csv(path + APPROX_METRIC_DATASET_NAME_CLEAN,on_bad_lines="warn")
    return df

def update_approx_metric_clean_dataset(name:str, df: pd.DataFrame):
    """Overwrites the approx metric clean dataset csv.

    Args:
        name (str): Name of the benchmark.
        df (pd.DataFrame): Dataframe.
    """
    assert all(df.columns == pd.Series(APPROX_EVAL_DATASET_STRUC_CLEAN.keys())), "Your dataframe has the wrong format"

    path = DATA_PATH + name + SEP + EVAL_FOLDER + SEP
    with open(path + APPROX_METRIC_DATASET_NAME_CLEAN, "w") as f:
        df.to_csv(f, index=False)

def get_sweep_dataset(name: str) -> pd.DataFrame:
    path = DATA_PATH + name + SEP + SWEEP_NAME
    df = pd.read_csv(path,on_bad_lines="warn")
    return df