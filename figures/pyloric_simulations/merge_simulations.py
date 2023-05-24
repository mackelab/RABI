import pandas as pd
import numpy as np
import os
import pickle


def merge_dataframes(file_dir: str) -> None:
    """
    Save all files that were simulated on the cluster in a single file.
    Overall, six files are created: for circuit parameters, simulations, seeds. For
    each of them, two files: one with `valid` simulations and one with `bad`
    simulations.

    Args:
        file_dir: Directory in which the files lie.
    """
    # checking for hidden files. If the file starts with '.', we discard it. Also
    # discard readme.txt

    files = os.listdir(file_dir + "simulation_outputs/")
    filenames_sims = []
    for file in files:
        if file[0] != "." and file != "readme.txt":
            filenames_sims.append(file)

    valid_params = []
    valid_sims = []
    valid_seeds = []

    for jj, fname_sims in enumerate(filenames_sims):
        if jj < 1000:
            params = pd.read_pickle(file_dir + "circuit_parameters/" + fname_sims)
            with open(file_dir + "simulation_outputs/" + fname_sims, "rb") as handle:
                sim_outs = pickle.load(handle)
            seeds = np.load(file_dir + "seeds/" + fname_sims[:-3] + "npy")

            for i, s in enumerate(sim_outs):
                if np.all(np.sum(s["voltage"] > 20, axis=1) > 5):
                    valid_params.append(pd.DataFrame([params.loc[i]]))
                    valid_sims.append(s["voltage"].astype("float32")[:, ::10])
                    valid_seeds.append(seeds[i])

            print("Finished file", jj)

    valid_params = pd.concat(valid_params, ignore_index=True)

    # Save data.
    general_path = "/mnt/qb/work/macke/mdeistler57/Documents/pyloric_sims/results/"
    path_to_data = "setup1/data/"

    valid_params.to_pickle(general_path + path_to_data + "valid_circuit_parameters.pkl")
    with open(
        general_path + path_to_data + "valid_simulation_outputs.pkl", "wb"
    ) as handle:
        pickle.dump(np.asarray(valid_sims), handle)
    np.save(general_path + path_to_data + "valid_seeds", valid_seeds)


if __name__ == "__main__":
    merge_dataframes(
        "/mnt/qb/work/macke/mdeistler57/Documents/pyloric_sims/results/setup1/data/"
    )