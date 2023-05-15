from copy import deepcopy
import torch
from torch import Tensor
from torch.distributions import Distribution
from torch.utils.data import DataLoader
from abc import abstractmethod

from typing import Callable, Optional, Tuple, Union
import pickle,inspect

from rbibm.utils.batched_processing import (
    eval_function_batched_sequential,
)


class Task:
    @abstractmethod
    def get_simulator(self, device: str = "cpu") -> Callable:
        """This function returns a simulator
        Args:
            device (str, optional): Device on which to compute. Defaults to "cpu".

        Returns:
            Callable: A function that gets parameters and produces data.
        """
        pass

    @abstractmethod
    def get_generator(self, device: str = "cpu") -> Callable:
        """This function returns a data generaotr

        Args:
            device (str, optional): Device on which to compute. Defaults to "cpu".

        Returns:
            Callable: Function that if called returns data.
        """
        pass

    def get_train_test_val_dataset(
        self,
        N_train: int,
        N_test: Optional[int] = None,
        N_val: Optional[int] = None,
        shuffle: bool = True,
        batch_size: int = 512,
        sim_batch_size: Optional[int] = None,
        device: str = "cpu",
        num_workers: int = 0,
    ) -> Tuple:
        """Creates a train test and validation datasets in the form of dataloader.

        Args:
            N_train (int): Number of training datapoints.
            N_test (Optional[int], optional): Number of testing datapoints. Defaults to None.
            N_val (Optional[int], optional): Number of validation datapoints. Defaults to None.
            shuffle (bool, optional): If dataloader should shuffle data. Defaults to True.
            batch_size (int, optional): Batch size of dataloader. Defaults to 512.
            device (str, optional): Device to compute on, data is always stored on cpu. Defaults to "cpu".
            num_workers (int, optional): Workers for dataloader. Defaults to 0.

        Returns:
            Tuple: Dataloaders
        """
        generator = self.get_generator(device=device)

        if N_val is None:
            N_val = 0
        if N_test is None:
            N_test = 0

        N = N_train + N_test + N_val
        # Simulation


        # TODO FIX THIS TO BE BATCHED... THEN MOVED TO CPU

        theta, x = generator(N)

        # Move to cpu for storage
        theta = theta.cpu()
        x = x.cpu()

        # Make dataloader
        theta_train, x_train = theta[:N_train], x[:N_train]

        train_loader = DataLoader(
            list(zip(theta_train, x_train)),  # type: ignore
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory="cuda" in device,
        )

        if N_val > 0:
            theta_val, x_val = (
                theta[N_train : N_train + N_val],
                x[N_train : N_train + N_val],
            )
            val_loader = DataLoader(
                list(zip(theta_val, x_val)),  # type: ignore
                shuffle=shuffle,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory="cuda" in device,
            )
        else:
            val_loader = None

        if N_test > 0:
            theta_test, x_test = (
                theta[N_train + N_val : N_train + N_val + N_test],
                x[N_train + N_val : N_train + N_val + N_test],
            )
            test_loader = DataLoader(
                list(zip(theta_test, x_test)),  # type: ignore
                shuffle=shuffle,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory="cuda" in device,
            )
        else:
            test_loader = None

        return train_loader, test_loader, val_loader


class Simulator:
    """Base class for all simulaotrs"""

    def __init__(
        self, simulator: Callable, device: str = "cpu", batch_size: Optional[int] = None
    ) -> None:
        """An standard simulator

        Args:
            simulator (Callable): Simulator function
            device (str, optional): Device to simulate on. Defaults to "cpu".
            batch_size (Optional[int], optional): Batching, required if not enough memory. Defaults to None.
        """
        self.simulator = simulator
        self.device = device
        self.batch_size = batch_size

    def __call__(self, thetas: Tensor) -> Tensor:
        """Performs simulation."""
        thetas = thetas.to(self.device)
        if self.batch_size is None:
            return self.simulator(thetas)
        else:
            if self.device == "cpu":
                return eval_function_batched_sequential(
                    self.simulator, thetas, batch_size=self.batch_size, dim=0, device=self.device,
                )
            else:
                return eval_function_batched_sequential(
                    self.simulator, thetas, batch_size=self.batch_size, dim=0, device=self.device,
                )


class InferenceTask(Task):

    """Classical inverse problems. We are interested in an unknown parameter theta given some observed data x_o"""

    task_type = "inference"
    single_observation = True

    def __init__(
        self,
        prior: Distribution,
        loglikelihood_fn: Optional[Callable] = None,
        simulator: Optional[Callable] = None,
    ):
        """Inferece task base class.

        Args:
            prior (Distribution): Prior distribution over the paramters.
            likelihood_fn (Optional[Callable], optional): Likelihood function. Defaults to None.
            simulator (Optional[Callable], optional): Simulator function. Defaults to None.
        """
        self.simulator = simulator
        self.loglikelihood_fn = loglikelihood_fn
        self.prior = prior

    def get_true_posterior(self, device: str = "cpu"):
        """Returns the true posterior distribution if available.

        Args:
            device (str, optional): Sets the device of the object. Defaults to "cpu".

        Raises:
            NotImplementedError: If not implemented or intractable.
        """
        raise NotImplementedError("Not implemented/tractable :(")

    def get_loglikelihood_fn(self, device: str = "cpu") -> Callable:
        """Return the loglikelihood function.

        Raises:
            NotImplementedError: If intractable

        Returns:
            Callable: Function returning a distribution.
        """
        if self.loglikelihood_fn is not None:
            return self.loglikelihood_fn
        else:
            raise NotImplementedError("Pass it during initialization...")

    def get_potential_fn(self, device: str = "cpu") -> Callable:
        """Return a potential function i.e. the unormalized posterior distirbution.

        Args:
            device (str, optional): Device. Defaults to "cpu".

        Returns:
            Callable: Function that gets parameter and data and computes the log posterior potential.
        """
        likelihood = self.get_loglikelihood_fn(device=device)
        prior = self.get_prior(device=device)

        def potential_fn(x, theta):
            x = x.to(device)
            theta = theta.to(device)
            likelihood_fn = likelihood(theta)
            l = likelihood_fn.log_prob(x) + prior.log_prob(theta)

            return l.squeeze()

        return potential_fn

    def get_simulator(
        self, batch_size: Optional[int] = None, device: str = "cpu"
    ) -> Simulator:
        """Return the simulator function

        Args:
            batch_size (Optional[int], optional): Batch size. Defaults to None.
            device (str, optional): Device. Defaults to "cpu".

        Raises:
            NotImplementedError: If not implemented

        Returns:
            Simulator: An simulator which produces data given parameters.
        """

        if self.simulator is not None:
            simulator = Simulator(self.simulator, device=device, batch_size=batch_size)
            return simulator
        elif self.loglikelihood_fn is not None:
            ll = self.get_loglikelihood_fn(device=device)

            def likelihood_based_sim(theta):
                return ll(theta).sample()  # type: ignore

            simulator = Simulator(
                likelihood_based_sim, device=device, batch_size=batch_size
            )
            return simulator
        else:
            raise NotImplementedError("Either specifiy a potential_fn or a simulator")

    def get_generator(
        self, batch_size: Optional[int] = None, device: str = "cpu"
    ) -> Callable:
        """Returns a data generator

        Args:
            batch_size (Optional[int], optional): Simulation batch size if necessary. Defaults to None.
            device (str, optional): Device. Defaults to "cpu".

        Returns:
            Callable: Function that returns a specified number of datapoints.
        """
        # prior = self.get_prior(device=device) Should be this
        prior = self.get_prior()
        simulator = self.get_simulator(batch_size=batch_size, device=device)

        def generator(N):
            thetas = prior.sample((N,))  # type: ignore
            x = simulator(thetas)
            return x, thetas

        return generator

    def get_prior(self, device: str = "cpu") -> Distribution:
        """Returns the prior distribution.

        Args:
            device (str, optional): Device. Defaults to "cpu".

        Returns:
            Distributions: Prior
        """
        if device == "cpu":
            return self.prior
        else:
            prior = deepcopy(self.prior)
            if hasattr(prior, "base_dist"):
                for key, val in prior.base_dist.__dict__.items():  # type: ignore
                    if isinstance(val, torch.Tensor):
                        prior.base_dist.__dict__[key] = val.to(device)  # type: ignore
            else:
                for key, val in self.prior.__dict__.items():
                    if isinstance(val, torch.Tensor):
                        prior.__dict__[key] = val.to(device)
            return prior

    @staticmethod
    def _is_picklable(obj):
        try:
            pickle.dumps(obj)
        except:
            return False
        return True
    
    def __getstate__(self):
        args  = deepcopy(self.__dict__)
        for key,arg in args.items():
            if not InferenceTask._is_picklable(arg):
                args[key] = None

        return args

    def __setstate__(self, d):
        args = inspect.getargs(self.__init__.__code__).args[1:]
        init_args = {}
        for arg in args:
            init_args[arg] = d[arg]
        self.__init__(**init_args)


class CDETask(Task):
    @abstractmethod
    def get_generator(self, device: str = "cpu") -> Callable:
        raise NotImplementedError()
