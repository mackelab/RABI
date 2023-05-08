
from rbi.defenses.base import AdditiveRegularizer

from rbi.utils.transforms import sampling_transform_jacobian
import torch







class TransformTraceRegularizer(AdditiveRegularizer):
    def __init__(self, model, loss_fn, beta=1.0, reduce="mean", **kwargs):
        super().__init__(model, loss_fn, reduce=reduce)
        self.beta = beta
        self._set_algorithm("standard", **kwargs)

    def _set_algorithm(self, method, **kwargs):
        self._algorithm = method
        if method == "standard":
            self._regularizer = lambda *args: self._regularizer_exact(*args, **kwargs)
        elif method == "asdf":
            raise NotImplementedError("asdfsaasdfasdf")
        else:
            raise ValueError("This does not exist")

    def _regularizer_exact(self, input, output, target, mc_samples=100):
        assert output.has_rsample, "We require an rsampling"
        matrix = sampling_transform_jacobian(self.model, input, mc_samples=mc_samples)
        reg = self.beta**2 * torch.diagonal(
            matrix, dim1=-2, dim2=-1
        ).sum(-1)

        return self._reduce(reg)
