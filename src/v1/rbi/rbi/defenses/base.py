from abc import abstractmethod
from copy import deepcopy

import torch
from torch import Tensor

from torch.nn import Module

from typing import Any, Callable, Tuple


class Defense:
    def __init__(self, model: Module, loss_fn: Callable) -> None:
        """Base class for any defense.

        Args:
            model (Module): Model
            loss_fn (Callable): Loss
        """
        self.model = model
        self.loss_fn = loss_fn

    @abstractmethod
    def activate(self, **kwargs):
        """Activates the defense this can be done by following modifications:
        - Changing the model
        - Changing the loss_fn
        - Changing the inputs to the loss_fn
        """
        raise NotImplementedError("Defense not implemented")

    @abstractmethod
    def deactivate(self, **kwargs):
        """Deactivates the defense. Reverses the activation method."""
        raise NotImplementedError("Defense not implemented")


class AdditiveRegularizer(Defense):
    """This adds an regularization term to the loss_fn."""

    def __init__(self, model: Module, loss_fn: Callable, reduce: str = "mean") -> None:
        """This adds an regularization term to the loss_fn.

        Args:
            model (Module): The model you want to regularize
            loss_fn (Callable): The loss function
            reduce (str, optional): Reduction you want to apply to the regularizer . Defaults to "mean".
        """
        super().__init__(model, loss_fn)
        self.reduce = reduce
        self._regularizer = None
        self._algorithm = None

    @property
    def regularizer(self) -> Callable:
        """Return a function that computes the regularizer

        Raises:
            NotImplementedError: If the function is not implemented

        Returns:
            Callable: Regularizer
        """
        if self._regularizer is None:
            raise NotImplementedError("Your regularizer is not implemented...")
        else:
            return self._regularizer

    @property
    def algorithm(self) -> str:
        """Returns the name of the current algorithm used to compute the regularizer.

        Raises:
            NotImplementedError: If no algorithm is set.

        Returns:
            str: Algorithm name.
        """
        if self._algorithm is None:
            raise NotImplementedError(
                "No algorithm to compute the regularizer is implemeneted"
            )
        else:
            return self._algorithm

    @algorithm.setter
    def algorithm(self, method):
        """Specific setter function, see _set_algorithm."""
        self._set_algorithm(method)

    @abstractmethod
    def _set_algorithm(self, method):
        pass

    def _reduce(self, reg: Tensor):
        """Reduce the regularizer values from multiple inputs to one number.

        Args:
            reg (str): Regularizers

        Raises:
            NotImplementedError: We only support mean, sum, max or min.

        Returns:
            _type_: _description_
        """
        if self.reduce == "mean":
            return torch.mean(reg)
        elif self.reduce == "sum":
            return torch.mean(reg)
        elif self.reduce == "max":
            return torch.max(reg)
        elif self.reduce == "min":
            return torch.min(reg)
        else:
            raise NotImplementedError()

    def activate(self, algorithm=None, **kwargs):
        """Activates regularization"""
        if algorithm is not None:
            self.algorithm = algorithm
        self.loss_fn.register_post_loss_regularizer(
            self.__class__.__name__, self.regularizer
        )

    def deactivate(self):
        """Deactivate regularization"""
        self.loss_fn.remove_post_loss_regularizer(self.__class__.__name__)


class DataAugmentationRegularizer(Defense):
    """This modifies the data *before* computing the loss."""

    def __init__(self, model: Module, loss_fn: Callable) -> None:
        """This adds an regularization term to the loss_fn.

        Args:
            model (Module): The model you want to regularize
            loss_fn (Callable): The loss function
            reduce (str, optional): Reduction you want to apply to the regularizer . Defaults to "mean".
        """
        super().__init__(model, loss_fn)

    @property
    def regularizer(self) -> Callable:
        """This return a callable which takes the x, y and outputs x', y' for which the loss is evaluated.

        Raises:
            NotImplementedError: If no regularizer is implanted.

        Returns:
            Callable: _description_
        """
        return self._transform

    @abstractmethod
    def _transform(self, input: Tensor, target: Any) -> Tuple[Tensor, Any]:
        """Transforms the input or target or both before computing the loss.

        Args:
            input (Tensor): Some input.
            target (Any): Some target

        Raises:
            NotImplementedError: Not implemented

        Returns:
            Tuple[Tensor, Any]: Transformed input and arget
        """
        raise NotImplementedError("Transfrom not implemented")

    def activate(self):
        self.loss_fn.register_pre_loss_regularizer(
            self.__class__.__name__, self.regularizer
        )

    def deactivate(self):
        self.loss_fn.remove_pre_loss_regularizer(self.__class__.__name__)

class PostHocDefense(Defense):
    def __init__(self, model: Module) -> None:
        super().__init__(model, loss_fn = None)

    @abstractmethod
    def activate(self, **kwargs):
        return None 
    
    @abstractmethod
    def deactivate(self, **kwargs):
        return None

class ArchitecturalRegularizer(Defense):
    def __init__(self, model: Module, loss_fn: Callable) -> None:
        """This constraints the models architecture in some way.

        Args:
            model (Module): The model you want to regularize
            loss_fn (Callable): The loss function
            reduce (str, optional): Reduction you want to apply to the regularizer . Defaults to "mean".
        """
        super().__init__(model, loss_fn)
        self.back_up_model = None

    @abstractmethod
    def _add_constraints_to_model(self,):
        """This function should constraint the model i.e. by enforcing Lipchitz continuity."""
        pass

    def activate(self):
        self.back_up_model = deepcopy(self.model)
        self._add_constraints_to_model()
        if hasattr(self.model, "net_constraints"):
            self.model.net_constraints.append(self.__class__.__name__)  # type: ignore
        else:
            self.model.net_constraints = [self.__class__.__name__]  # type: ignore
        self.loss_fn.model = self.model

    def deactivate(self):
        self.model.net_constraints.remove(self.__class__.__name__)  # type: ignore
        self.model = self.back_up_model
        self.back_up_model = None
        self.loss_fn.model = self.model