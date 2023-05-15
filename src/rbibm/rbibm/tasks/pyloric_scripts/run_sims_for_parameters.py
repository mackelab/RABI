import random
import time

time.sleep(random.uniform(0, 20))

import sys, os
import torch

from multiprocessing import Pool
import pandas as pd
import numpy as np
import sys

from pyloric import simulate, create_prior

NUM_CORES = 16

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


def main(parameter_path, id):

    thetas = torch.load(parameter_path)
    seed = torch.randint(0, 2 ** 32 - 1, (thetas.shape[0], 1)).float()
    params_with_seeds = np.concatenate((thetas, seed.numpy()), axis=1)
    with Pool(NUM_CORES) as pool:
        xs = pool.map(my_simulator, params_with_seeds)
    xs = torch.as_tensor(xs,dtype=torch.float)
    torch.save(thetas, f"thetas_{id}.pkl")
    torch.save(xs, f"xs_{id}.pkl")
    torch.save(seed, f"seed_{id}.pkl")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(*args)