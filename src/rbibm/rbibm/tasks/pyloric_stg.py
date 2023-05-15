from rbibm.tasks.base import InferenceTask
import torch

from pyloric import simulate, create_prior
import pandas as pd
import numpy as np

from sbi.utils import BoxUniform
import pickle

import os, sys
from multiprocessing import Pool
import subprocess
import glob
import time
from torch.utils.data import DataLoader

global nan_replace_glob
global summary_glob
global NAMES
global CACHE

summary_glob = "summary_statistics"
nan_replace_glob = -99
p1 = create_prior()
COLUMN_NAMES = p1.sample((1,)).columns
NAMES = [
    "AB/PD_Na",
    "AB/PD_CaT",
    "AB/PD_CaS",
    "AB/PD_A",
    "AB/PDK_Ca",
    "AB/PD_Kd",
    "AB/PD_H",
    "AB/PD_Leak",
    "LP_Na",
    "LP_CaT",
    "LP_CaS",
    "LP_A",
    "LP_KCa",
    "LP_Kd",
    "LP_H",
    "LP_Leak",
    "PY_Na",
    "PY_CaT",
    "PY_CaS",
    "PY_A",
    "PY_KCa",
    "PY_Kd",
    "PY_H",
    "PY_Leak",
    "SynapsesAB-LP",
    "SynapsesPD-LP",
    "SynapsesAB-PY",
    "SynapsesPD-PY",
    "SynapsesLP-PD",
    "SynapsesLP-PY",
    "SynapsesPY-LP",
]
DIR_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "pyloric_scripts")
WITH_DT = False


def my_simulator(params_with_seeds):
    p1 = create_prior()
    pars = p1.sample((1,))
    column_names = pars.columns

    parameter_set_pd = pd.DataFrame(
        np.asarray([params_with_seeds[:-1]]), columns=column_names
    )
    out_target = simulate(
        parameter_set_pd.loc[0],
        seed=int(params_with_seeds[-1]),
        dt=0.025,
        t_max=2000,
        temperature=283,
        noise_std=0.001,
    )
    # Subsample data to keep the data at a reasonable filesize.
    subsample_factor = 100
    if WITH_DT:
        out_target["voltage"] = out_target["voltage"][:, ::subsample_factor]
        out_target["dt"] *= subsample_factor  # keeps pyloric.utils.show_traces working.
        return out_target
    else:
        out_target["voltage"] = out_target["voltage"][:, ::subsample_factor]
        return out_target["voltage"].flatten()


def slurm_simulator(thetas, simulation_batches=500):
    N = thetas.shape[0]
    if N < simulation_batches:
        simulation_batches = N
    jobs = N // simulation_batches

    # Delete intermediate results
    for j in range(jobs):
        subprocess.run(["rm", DIR_PATH + os.sep + f"thetas_{j}.pkl"])
        subprocess.run(["rm", DIR_PATH + os.sep + f"xs_{j}.pkl"])
        subprocess.run(["rm", DIR_PATH + os.sep + f"seed_{j}.pkl"])

    for fl in glob.glob(DIR_PATH + os.sep + "slurm-*"):
        os.remove(fl)

    # Run the slurm jobs...
    for j in range(jobs):
        torch.save(
            thetas[j * simulation_batches : (j + 1) * simulation_batches, :],
            DIR_PATH + os.sep + f"thetas_{j}.pkl",
        )

    # Wait to for saving thetas
    time.sleep(10)

    for j in range(jobs):
        subprocess.run(
            [
                "sbatch",
                DIR_PATH + os.sep + "run_one.sh",
                DIR_PATH + os.sep + f"thetas_{j}.pkl",
                f"{j}",
                DIR_PATH,
                f"--output={DIR_PATH}",
            ]
        )

    time.sleep(10)

    # Check for complettion
    start_time = time.time()
    jobs_status = np.zeros(jobs)
    i = 0
    while True:
        if (i % 1000) == 0:
            for j in range(jobs):
                jobs_status[j] = os.path.isfile(DIR_PATH + os.sep + f"xs_{j}.pkl")
            sys.stdout.write(f"\rCompleted {int(jobs_status.sum())}/{jobs} jobs")
            sys.stdout.flush()
            if jobs_status.sum() == jobs:
                break
            current_time = time.time()
            time_till_execution = current_time - start_time
            if time_till_execution > 300:
                start_time = time.time()
                for j in range(jobs):
                    if not jobs_status[j]:
                        subprocess.run(
                            [
                                "sbatch",
                                DIR_PATH + os.sep + "run_one.sh",
                                DIR_PATH + os.sep + f"thetas_{j}.pkl",
                                f"{j}",
                                DIR_PATH,
                                f"--output={DIR_PATH}",
                            ]
                        )

        i += 1

    # Wait to receive xs
    time.sleep(10)
    subprocess.run(["scancel", "-n", "run_one.sh"])

    # if jobs_status.sum() != jobs:
    #     return slurm_simulator(thetas)
    # Append final results
    xs = []
    for j in range(jobs):
        xs.append(torch.load(DIR_PATH + os.sep + f"xs_{j}.pkl"))

    x = torch.vstack(xs)

    # Delete intermediate results
    for j in range(jobs):
        subprocess.run(["rm", DIR_PATH + os.sep + f"thetas_{j}.pkl"])
        subprocess.run(["rm", DIR_PATH + os.sep + f"xs_{j}.pkl"])
        subprocess.run(["rm", DIR_PATH + os.sep + f"seed_{j}.pkl"])

    for fl in glob.glob(DIR_PATH + os.sep + "slurm-*"):
        os.remove(fl)
    return x.float()


class PyloricTask(InferenceTask):
    def __init__(
        self,
        sim_type="parallel",
        num_cores=16,
        seed=0,
        with_dt=False,
        use_pre_computed_dataset=True,
    ):

        _ = torch.manual_seed(seed)
        p1 = create_prior()
        self.column_names = p1.sample((1,)).columns
        lower = p1.lower
        upper = p1.upper

        self.input_dim = 2400
        self.output_dim = 31

        prior = BoxUniform(lower, upper)
        self.t = torch.arange(0, 11000, 0.025)
        self.use_pre_computed_dataset = use_pre_computed_dataset

        if sim_type == "sequential":

            def simulator(parameters):
                batch_shape = parameters.shape[:-1]
                parameters = parameters.reshape(-1, parameters.shape[-1])
                NUM_SAMPLES = parameters.shape[0]
                seed = torch.randint(0, 2 ** 32 - 1, (NUM_SAMPLES,)).float()
                xs = []
                for i in range(NUM_SAMPLES):
                    sample = pd.DataFrame(
                        parameters[i].reshape(1, -1).numpy(), columns=self.column_names
                    )
                    out_target = simulate(
                        sample.loc[0],
                        seed=int(seed[i]),
                        dt=0.025,
                        t_max=2000,
                        temperature=283,
                        noise_std=0.001,
                    )
                    subsample_factor = 100
                    x = out_target["voltage"][:, ::subsample_factor]
                    xs.append(torch.tensor(x).flatten())
                
                xs = torch.vstack(xs).float()
                return xs.reshape(batch_shape + (xs.shape[-1],))

        elif sim_type == "parallel":

            def simulator(parameters):
                batch_shape = parameters.shape[:-1]
                parameters = parameters.reshape(-1, parameters.shape[-1])
                NUM_SAMPLES = parameters.shape[0]
                seed = torch.randint(0, 2 ** 32 - 1, (NUM_SAMPLES, 1))
                params_with_seeds = np.concatenate((parameters, seed.numpy()), axis=1)
                with Pool(num_cores) as pool:
                    xs = pool.map(my_simulator, params_with_seeds)
                global WITH_DT
                WITH_DT = with_dt
                if with_dt:
                    return xs
                else:
                    xs = torch.as_tensor(xs, dtype=torch.float)
                    return xs.reshape(batch_shape + (xs.shape[-1],))

        elif sim_type == "slurm":
            simulator = slurm_simulator
        else:
            raise NotImplementedError()

        super().__init__(prior, simulator=simulator)

    def get_train_test_val_dataset(
        self,
        N_train: int,
        N_test=None,
        N_val=None,
        shuffle: bool = True,
        batch_size: int = 512,
        device: str = "cpu",
        num_workers: int = 0,
    ):
        if not self.use_pre_computed_dataset:
            return super().get_train_test_val_dataset(
                N_train, N_test, N_val, shuffle, batch_size, device, num_workers
            )
        else:
            with open(
                "/mnt/qb/work/macke/mgloeckler90/RBI_paper/pyloric/valid_simulation_outputs.pkl",
                "rb",
            ) as handle:
                x = pickle.load(handle)

            x = torch.as_tensor(x).reshape(-1, 3 * 800)

            with open(
                "/mnt/qb/work/macke/mgloeckler90/RBI_paper/pyloric/pyloric_sims/results/setup1/data/valid_circuit_parameters.pkl",
                "rb",
            ) as handle2:
                theta = pickle.load(handle2)
            theta = torch.as_tensor(theta.to_numpy())

            theta_train, x_train = theta[:N_train], x[:N_train]

            train_loader = DataLoader(
                list(zip(x_train, theta_train)),  # type: ignore
                shuffle=shuffle,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory="cuda" in device,
            )

            if N_val is not None and N_val > 0:
                theta_val, x_val = (
                    theta[N_train : N_train + N_val],
                    x[N_train : N_train + N_val],
                )
                val_loader = DataLoader(
                    list(zip(x_val, theta_val)),  # type: ignore
                    shuffle=shuffle,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory="cuda" in device,
                )
            else:
                val_loader = None

            if N_test is not None and N_test > 0:
                theta_test, x_test = (
                    theta[N_train + N_val : N_train + N_val + N_test],
                    x[N_train + N_val : N_train + N_val + N_test],
                )
                test_loader = DataLoader(
                    list(zip(x_test, theta_test)),  # type: ignore
                    shuffle=shuffle,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory="cuda" in device,
                )
            else:
                test_loader = None

            return train_loader, test_loader, val_loader