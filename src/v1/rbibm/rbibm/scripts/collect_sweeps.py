from datetime import date, datetime
import datetime

import os
import yaml
from time import sleep

from rbibm.utils.utils_data import add_new_sweep, ROOT_PATH, SEP


def collect_sweep_results(datum, time, *args) -> None:
    """ This collects sweeping results and saves it nicely in a csv... """

    sweeper_argument = ["sweeper=" in a for a in args]
    is_sweeped = any(sweeper_argument)

    if is_sweeped:
        sleep(2)

        folders = os.listdir(ROOT_PATH + SEP + "outputs")
        i = folders.index(datum)
        sub_folders = os.listdir(ROOT_PATH + SEP + f"outputs/{folders[i]}")

        # Find closest...
        diffs = [
            abs(int(s.replace("-", "")) - int(time.replace("-", "")))
            for s in sub_folders
        ]
        j = diffs.index(min(diffs))

        file_path = (
            ROOT_PATH
            + SEP
            + f"outputs/{folders[i]}/{sub_folders[j]}/optimization_results.yaml"
        )

        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                res = yaml.safe_load(f)

        config_path = (
            ROOT_PATH
            + SEP
            + f"outputs/{folders[i]}/{sub_folders[j]}/0/.hydra/config.yaml"
        )

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

        search_space = []
        overrides = []

        args = list(args)
        del args[0]

        for a in args:
            if "sweeper" not in a and (
                "interval" in a or "choice" in a or "," in a or "range" in a
            ):
                search_space.append(a)
            elif "sweeper" not in a:
                overrides.append(a)

        name = config["name"]
        sweeper = config["sweeper"]["name"]
        objective = config["sweeper"]["objective"]
        direction = config["sweeper"]["direction"]

        if "best_params" in res and "best_value" in res:
            # Single objective
            best_params = res["best_params"]
            best_value = res["best_value"]
        elif "solutions" in res:
            # Multiobjective
            best_params = [list(r.values())[0] for r in res["solutions"]]
            best_value = [list(r.values())[1] for r in res["solutions"]]
        else:
            best_params = None
            best_value = None

        add_new_sweep(
            name=name,
            search_space=search_space,
            sweeper=sweeper,
            objective=objective,
            direction=direction,
            overrides=overrides,
            best_params=best_params,
            best_value=best_value,
        )
