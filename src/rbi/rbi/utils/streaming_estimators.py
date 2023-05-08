import torch
from torch import Tensor

from typing import Union


class MovingAverageEstimator:
    def __init__(self) -> None:
        """This implements a simple streaming moving average estimator."""
        self.t = 0
        self._x_old = None

    def __call__(self, x_new: Union[Tensor, list, tuple]):
        self.t += 1
        if isinstance(x_new, Tensor):
            if self._x_old is None:
                self._x_old = x_new
            else:
                self._x_old = self._x_old + (x_new - self._x_old) / self.t
        elif isinstance(x_new, list) or isinstance(x_new, tuple):
            if self._x_old is None:
                self._x_old = x_new
            else:
                self._x_old = [x + (y - x) / self.t for x, y in zip(self._x_old, x_new)]
        else:
            raise ValueError("Unknown value.")

    @property
    def value(self) -> Union[Tensor, list, tuple, None]:
        return self._x_old


class ExponentialMovingAverageEstimator:
    def __init__(self, decay: float = 0.95) -> None:
        """Implements an EMA estimator

        Args:
            decay (float, optional): Decay rete on history dependence. Defaults to 0.99.
        """
        self.t = 0
        self.decay = decay
        self._x_old = None

    def __call__(self, x_new):
        """ Adds a new sample to the EMA estimator."""
        self.t += 1

        if isinstance(x_new, Tensor):
            x_new = torch.nan_to_num(x_new)
            if self._x_old is None:
                self._x_old = self.decay * x_new
            else:
                self._x_old = self.decay * x_new + (1 - self.decay) * self._x_old  # type: ignore
        elif isinstance(x_new, list) or isinstance(x_new, tuple):

            if self._x_old is None:
                self._x_old = [self.decay * torch.nan_to_num(x_i) for x_i in x_new]
            else:
                self._x_old = [
                    self.decay * torch.nan_to_num(x_i) + (1 - self.decay) * c_i
                    for x_i, c_i in zip(x_new, self._x_old)
                ]
        else:
            raise ValueError("Unknown value")

    @property
    def value(self):
        """Return the currently estimated value."""
        if isinstance(self._x_old, Tensor):
            return self._x_old / (1 - (1 - self.decay )** self.t)
        elif isinstance(self._x_old, list) or isinstance(self._x_old, tuple):
            return [val / (1 - (1 - self.decay )** self.t) for val in self._x_old]
        else:
            raise ValueError("Unknonw value")