from typing import Type, Union
from rbibm.tasks.base import CDETask, InferenceTask, Task
from rbibm.tasks.gaussian_linear import GaussianLinearTask, HardGaussianLinearTask
from rbibm.tasks.glr_tasks import RBFRegressionTask
from rbibm.tasks.hh_task import HHTask
from rbibm.tasks.lv_task import LotkaVolterraTask
from rbibm.tasks.rps_task import RPSTask
from rbibm.tasks.square_task import SquareTask
from rbibm.tasks.vae_task import VAETask
from rbibm.tasks.spatial_sir import SpatialSIRTask
from rbibm.tasks.sir_task import SIRTask


def get_task(name: str) -> Union[Type[Task], Type[InferenceTask], Type[CDETask]]:
    """This is a helper function that return tasks by name.

    Args:
        name (str): Name of the tesk

    Raises:
        NotImplementedError: If the named task is not implemented.

    Returns:
        Task: Task with assciated name.
    """
    if name == "lotka_volterra":
        return LotkaVolterraTask
    elif name == "gaussian_linear":
        return GaussianLinearTask
    elif name == "hard_gaussian_linear":
        return HardGaussianLinearTask
    elif name == "square":
        return SquareTask
    elif name == "rbf_regression":
        return RBFRegressionTask
    elif name == "hudgkin_huxley":
        return HHTask
    elif name == "vae_task":
        return VAETask
    elif name == "rps_task":
        return RPSTask
    elif name == "spatial_sir":
        return SpatialSIRTask
    elif name == "sir":
        return SIRTask
    elif name == "pyloric":
        from rbibm.tasks.pyloric_stg import PyloricTask
        return PyloricTask
    else:
        raise NotImplementedError("Not interfaced yet...")
