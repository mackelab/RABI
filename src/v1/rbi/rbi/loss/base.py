from abc import abstractmethod
from torch.nn.modules.loss import _Loss

import torch
from torch import Tensor
from torch.nn import Module

from typing import Callable, Dict, Optional, Union

from rbi.models.base import ParametricProbabilisticModel, PyroFlowModel


def reduce(
    loss: Tensor,
    reduction: Optional[str],
    weight: Optional[Union[Tensor, float]] = None,
) -> Tensor:
    """Reduces a tensor to a single value

    Args:
        loss (Tensor): The loss tensor which should be reduced
        reduction (str): The reduction. Must be one of: mean, sum, median, min, max or None
        weight (Optional[Union[Tensor, float]], optional): Multiplicative weight. Defaults to None.

    Raises:
        ValueError: If unsupported reductions is given

    Returns:
        Tensor: Reduced loss
    """
    if weight is None:
        weight = 1.0

    if reduction == "mean":
        return torch.mean(weight * loss)
    elif reduction == "sum":
        return torch.sum(weight * loss)
    elif reduction == "median":
        return torch.median(weight * loss)
    elif reduction == "min":
        return torch.min(weight * loss)
    elif reduction == "max":
        return torch.max(weight * loss)
    elif reduction is None:
        return loss
    else:
        raise ValueError(
            "We only support the reduction operations: mean, sum, median, min, max and None."
        )


class Loss(_Loss):
    """A standard PyTorch loss function"""

    def __init__(self, reduction: Optional[str] = "mean") -> None:
        super().__init__(None, None, reduction)  # type: ignore


class EvalLoss(Loss):
    """Evaluation loss to eval results"""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(reduction)

    @abstractmethod
    def _loss(self, output, target) -> Tensor:
        raise NotImplementedError("Loss not implemented")

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        return reduce(self._loss(output, target), self.reduction)


class TrainLoss(Loss):
    def __init__(
        self,
        model: Union[ParametricProbabilisticModel, PyroFlowModel],
        reduction: Optional[str] = "mean",
    ) -> None:
        """The model which is trained by this loss.

        Args:
            model (Module): A model
            reduction (str, optional): Reduction strategy for the loss. Defaults to "mean".
        """
        super().__init__(reduction)
        self.model = model
        self._pre_loss_regularizers = {}
        self._post_loss_regularizers = {}

    @property
    def pre_loss_regularizers(self, *args, **kwargs) -> Dict:
        """Return all registered pre_loss regularizer's

        Returns:
            Dict: Dictionary of pre_loss regularizer's
        """
        return self._pre_loss_regularizers

    @property
    def post_loss_regularizers(self, *args, **kwargs) -> Dict:
        """Returns all post_loss regularizers

        Returns:
            Dict: Dictionary of regularizers
        """
        return self._post_loss_regularizers

    def register_post_loss_regularizer(self, name: str, regularizer: Callable) -> None:
        """Register a post loss regularizer.

        Args:
            name (str): Name of the regularizer
            regularizer (Callable): A function that computes the regularizer.
        """
        self._post_loss_regularizers[name] = regularizer

    def remove_post_loss_regularizer(self, name: str) -> None:
        """Removes the post loss regularizer with the given name

        Args:
            name (str): Name of regularizer that should be remove
        """
        del self._post_loss_regularizers[name]

    def register_pre_loss_regularizer(self, name: str, regularizer: Callable) -> None:
        """Registers a pre loss regularizer

        Args:
            name (str): Name of the regularizer
            regularizer (Callable): Regularizer function
        """
        self._pre_loss_regularizers[name] = regularizer

    def remove_pre_loss_regularizer(self, name: str) -> None:
        """Removes a pre loss regularizer

        Args:
            name (str): Name of regularizer to remove.
        """
        del self._pre_loss_regularizers[name]

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Evaluates the loss, overwrites __call__

        Args:
            input (Tensor): Input argument
            target (Tensor): Target argument

        Returns:
            Tensor: Loss + regularizers
        """

        loss = 0
        if self.training:

            if len(self.pre_loss_regularizers) == 0:
                output = self.model(input)
                loss += self._loss(output, target, input)
            else:
                for pre_regularizer in self.pre_loss_regularizers.values():
                    input, target = pre_regularizer(input, target)
                    output = self.model(input)
                    loss += self._loss(output, target, input) / len(
                        self.pre_loss_regularizers
                    )

            post_reg = torch.zeros(1, device=input.device)
            for post_regularizer in self.post_loss_regularizers.values():
                post_reg += post_regularizer(input, output, target)

            return reduce(loss, self.reduction, None) + (post_reg - post_reg.detach())
        else:
            output = self.model(input)
            loss += self._loss(output, target, input)
            return reduce(loss, self.reduction, None)

    @abstractmethod
    def _loss(self, output, target, input) -> Tensor:
        pass
