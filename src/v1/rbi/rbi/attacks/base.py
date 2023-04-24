
import torch
from torch import Tensor

from abc import abstractmethod
from typing import Any, Callable, Callable, Optional


class Attack:
    def __init__(
        self,
        predict: Callable,
        loss_fn: Optional[Callable] = None,
        targeted: bool = False,
        clip_min: float = -1e3,
        clip_max: float = 1e3,
        *args,
        **kwargs,
    ) -> None:
        """Abstract base class for attacks

        Args:
            predict (Callable): Inputs a tensor and outputs an abstract prediction i.e. an distribution
            loss_fn (Callable): Inputs two objects and outputs a float.
            targeted (bool): If set true the attack requires a target.
            clip_min (float, optional): _description_. Defaults to -torch.inf.
            clip_max (float, optional): _description_. Defaults to torch.inf.
        """
        self.predict = predict
        self.loss_fn = loss_fn
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted

    def _get_predicted_label(self, x: Tensor) -> Any:
        """Computes the predicted label, which is used in the non-targeted regime.

        Args:
            x (Tensor): Input, which should be perturbed

        Returns:
            Tensor: Prediction output
        """
        with torch.no_grad():
            y = self.predict(x)
        return y

    def _verify_and_process_inputs(self, x: Tensor, y: Any):
        """This process the input and output/target.

        Args:
            x (Tensor): Input
            y (Any): Output/target

        Returns:
            Tupel(Tensor, Any): Processed input and output/target
        """
        if self.targeted:
            assert y is not None

        # Target must not be a tensor, but can be a distribution...
        if not self.targeted:
            if y is None:
                y = self._get_predicted_label(x)

        if isinstance(x, torch.Tensor):
            x = x.detach().clone()
        if isinstance(y, torch.Tensor):
            y = y.detach().clone()

        return x, y

    @abstractmethod
    def perturb(self, x: Tensor, target: Optional[Any] = None, **kwargs):
        """This function computes a adversarial perturbation x' = x + d

        Args:
            x (Tensor): An input.
            target (Any, optional): An optional abstract target. Defaults to None.

        Raises:
            NotImplementedError: Needs to be implemented...
        """
        raise NotImplementedError("This attack is not implemented...")
