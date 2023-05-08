from rbi.defenses.base import AdditiveRegularizer
from rbi.loss.base import TrainLoss
from rbi.utils.autograd_tools import batch_jacobian_outer_product_hutchinson_trace, batch_jacobian_norm

import torch
from torch.nn import Module


class JacobiRegularizer(AdditiveRegularizer):
    def __init__(
        self,
        model: Module,
        loss_fn: TrainLoss,
        beta: float=0.1,
        reduce: str="mean",
        algorithm:str="jac_exact",
        mc_samples: int=10,
    ):
        super().__init__(model, loss_fn, reduce=reduce)
        self.beta = beta
        self.algorithm = algorithm
        self.trace_mc_samples = mc_samples

    def _set_algorithm(self, method, **kwargs):
        self._algorithm = method
        if method == "jac_exact":
            self._regularizer = self._regularizer_exact
        elif method == "jac_trace_mc":
            self._regularizer = self._regularizer_mc
        else:
            raise ValueError("This does not exist")

    def _regularizer_exact(self, input, output, target):

        norms = batch_jacobian_norm(self.model.forward_parameters, input, ord="fro")  # type: ignore

        return self.beta * self._reduce(norms)

    def _regularizer_mc(self, input, output, target):
        return (
            batch_jacobian_outer_product_hutchinson_trace(
                self.model.forward_parameters,
                input,
                mc_samples=self.trace_mc_samples,
            )
            ** 2
        )


