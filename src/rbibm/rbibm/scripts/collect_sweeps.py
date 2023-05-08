

import os
import yaml
from time import sleep

from rbibm.utils.utils_data import add_new_sweep, ROOT_PATH, SEP


def is_sweep_and_get_overides(*args):
    for a in args:
        if "sweeper" in a and "none" not in a:
            return True, args
        elif "experiment" in a:
            val = a.split("=")[-1]
            current_file_path = os.path.abspath(__file__)
            dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
            path = str(dir) + os.sep + "config" + os.sep + "experiment" + os.sep + str(val) + ".yaml"
            print(path)
            if os.path.exists(path):
                with open(path, "r") as f:
                    config = yaml.safe_load(f)
                    print(config)
                    sweeper = None
                    for entry in config["defaults"]:
                        for key, val in entry.items():
                            if "sweeper" in key:
                                sweeper = val
                    if sweeper is not None:
                        args = ["sweeper=" + sweeper] + [key+ "=" + str(val) for key,val in config["hydra"]["sweeper"]["params"].items()]
                        return True, args


    return False, None

def collect_sweep_results(datum, time, *args) -> None:
    """ This collects sweeping results and saves it nicely in a csv... """

    is_sweeped, args = is_sweep_and_get_overides(*args)

    if is_sweeped:
        sleep(1)

        folders = os.listdir(ROOT_PATH + SEP + "outputs")
        i = folders.index(datum)
        sub_folders = os.listdir(ROOT_PATH + SEP + f"outputs/{folders[i]}")


        search_space = []
        overrides = []
        sweeper = None
        args = list(args)
        #print(args)
        for a in args:
            if "sweeper" not in a and (
                "interval" in a or "choice" in a or "," in a or "range" in a
            ):
                search_space.append(a)
            elif "sweeper" in a:
                sweeper = a.split("=")[1]
            elif "=" in a:
                overrides.append(a)
            else:
                pass

        diffs = [
                abs(int(s.replace("-", "")) - int(time.replace("-", "")))
                for s in sub_folders
            ]
        finished = False
        #print(diffs)
        for l in range(len(diffs)):
            # Find closest... we try the 10 closest ones otherwise we deem it failed
            
            j = diffs.index(min(diffs))
            #print(j)

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

            overrides_path = (
                ROOT_PATH
                + SEP
                + f"outputs/{folders[i]}/{sub_folders[j]}/0/.hydra/overrides.yaml"
            )

            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)

            if os.path.exists(overrides_path):
                with open(overrides_path, "r") as f:
                    overrides_check = yaml.safe_load(f)
    
            overrides_keys = [o.split("=")[0] for o in overrides_check if "=" in o and "experiment" not in o]
            overrides_val = [o.split("=")[1] for o in overrides_check if "=" in o and "experiment" not in o]
            same_overrides = True

            #print(overrides_keys)
            #print(overrides_val)

            #print(overrides)
            #print(search_space)

            for override in overrides:
                override_name = override.split("=")[0]
                override_val = override.split("=")[1]
                same_overrides =  same_overrides and (override_name in overrides_keys)
                if override_name in overrides_keys:
                    val = overrides_val[overrides_keys.index(override_name)]
                    same_overrides = same_overrides and (override_val == val)


            for searched in search_space:
                override_name = searched.split("=")[0]
                same_overrides =  same_overrides and (override_name in overrides_keys)
                
            #print(same_overrides)
            if not same_overrides:
                diffs[j] = float("inf")
                continue


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

            finished = True
            break
        
        if finished:
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
