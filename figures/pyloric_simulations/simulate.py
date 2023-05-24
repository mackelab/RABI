from pyloric import simulate, create_prior, summary_stats
import numpy as np
import time
import pickle

import multiprocessing
from multiprocessing import Pool
import torch
import pandas as pd


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
    subsample_factor = 10
    out_target["voltage"] = out_target["voltage"][:, ::subsample_factor]
    out_target["dt"] *= subsample_factor  # keeps pyloric.utils.show_traces working.
    return out_target


num_repeats = 100

for _ in range(num_repeats):

    num_sims = 10000
    num_cores = 32

    global_seed = int((time.time() % 1) * 1e7)
    np.random.seed(global_seed)  # Seeding the seeds for the simulator.
    torch.manual_seed(global_seed)  # Seeding the prior.
    seeds = np.random.randint(0, 10000, (num_sims, 1))

    prior = create_prior()
    parameter_sets = prior.sample((num_sims,))
    data_np = parameter_sets.to_numpy()
    params_with_seeds = np.concatenate((data_np, seeds), axis=1)

    with Pool(num_cores) as pool:
        start_time = time.time()
        sims_out = pool.map(my_simulator, params_with_seeds)
        print("Simulation time", time.time() - start_time)

    general_path = "/mnt/qb/work/macke/mdeistler57/Documents/pyloric_sims/results/"
    path_to_data = "setup1/data/"
    path = general_path + path_to_data
    filename = f"sim_{global_seed}"

    with open(path + "simulation_outputs/" + filename + ".pkl", "wb") as handle:
        pickle.dump(sims_out, handle)
    parameter_sets.to_pickle(
        path + "circuit_parameters/" + filename + ".pkl"
    )
    np.save(path + "seeds/" + filename, seeds)

    print("============ Finished ============")