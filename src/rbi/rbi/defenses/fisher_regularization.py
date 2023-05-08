
from rbi.utils.fisher_info import fisher_information_matrix, monte_carlo_fisher

import torch
from torch.distributions import Distribution
from torch import Tensor
from torch.nn import Module
from rbi.defenses.base import AdditiveRegularizer, DataAugmentationRegularizer
from rbi.loss.base import TrainLoss
from rbi.utils.streaming_estimators import ExponentialMovingAverageEstimator
from rbi.utils.spectral_methods import power_iterations
from rbi.utils.distributions import sample_lp_uniformly
    

class FIMTraceRegularizer(AdditiveRegularizer):
    def __init__(
        self,
        model: Module,
        loss_fn: TrainLoss,
        beta: float =1.0,
        algorithm: str="jac_exact",
        mc_samples: int=20,
        ema_mc_samples: int=1,
        ema_decay: float = 0.99,
        clamp_val = None,
        grad_clamp_val = None,
        reduce: str="mean",
    ):
        """FIM Trace as regularizer. Can be either estimated exactly/via MC estimators. Or using an more efficient ema.

        Args:
            model (Module): Model.
            loss_fn (TrainLoss): Loss fn.
            beta (float, optional): Regularization strength. Defaults to 1.0.
            algorithm (str, optional): Algorithm. Defaults to "jac_exact".
            mc_samples (int, optional): MC samples to use in MC estimations. Defaults to 20.
            ema_decay (float, optional): EMA. Defaults to 0.99.
            reduce (str, optional): _description_. Defaults to "mean".
        """
        super().__init__(model, loss_fn, reduce=reduce)
        self.beta = beta
        self.algorithm = algorithm
        # Only required if Fisher information cannot be computed in closed form.
        self.mc_samples = mc_samples
        self.ema = ExponentialMovingAverageEstimator(ema_decay)
        self.ema_mc_samples = ema_mc_samples
        self.clamp_val = clamp_val
        self.grad_clamp_val=grad_clamp_val

    def _set_algorithm(self, method, **kwargs) -> None:
        self._algorithm = method
        if method == "jac_exact":
            self._regularizer = self._regularizer_exact
        elif method == "ema":
            self._regularizer = self._regularizer_ema

    def _regularizer_exact(self, input: Tensor, output: Distribution, target: Tensor) -> Tensor:
        """ This implements the regularizer by explicitly computing the FIM"""
        reg = fisher_information_matrix(self.model, input, output, mc_samples = self.mc_samples, typ="trace")
        mask = torch.isfinite(reg)
        reg = reg[mask]
        if self.clamp_val is not None:
            reg = reg.clamp(min=0,max=self.clamp_val)

        return self.beta * self._reduce(reg)

    def _regularizer_ema(self, input: Tensor, output: Distribution, target: Tensor) -> Tensor:
        """ This implements the regularizer by performing an EMA estimate."""

        reg = monte_carlo_fisher(
                 self.model, input, output, mc_samples=self.ema_mc_samples, create_graph=True, retain_graph=True, typ="trace",
             )
        mask = torch.isfinite(reg)
        reg = reg[mask]
        reg.clamp(min=0, max=self.clamp_val)
        if self.clamp_val is not None:
            reg = reg.clamp(min=0, max=self.clamp_val)
        reg = self.beta * self._reduce(reg)

        grads = torch.autograd.grad(
            reg,
            self.model.parameters(),  # type: ignore
            retain_graph=True,
        )

        self.ema(grads)

        new_val = self.ema.value
        for params, grad in zip(self.model.parameters(), new_val):
            params.grad = torch.nan_to_num(grad.detach())

        if self.grad_clamp_val is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clamp_val, norm_type=2.0)

        return torch.zeros(1, device=input.device)

class NoisyFIMTraceRegularization(FIMTraceRegularizer):
    def __init__(self, model: Module, loss_fn: TrainLoss, eps=0.5, noise_order=2., beta: float = 1, algorithm: str = "jac_exact", mc_samples: int = 20, ema_mc_samples: int = 1, ema_decay: float = 0.99,  reduce: str = "mean"):
        super().__init__(model, loss_fn, beta, algorithm, mc_samples, ema_mc_samples, ema_decay, reduce)
        self.eps = eps
        self.noise_order = noise_order

    def _regularizer_exact(self, input: Tensor, output: Distribution, target: Tensor) -> Tensor:
        input = input + sample_lp_uniformly(input.shape[:-1].numel(), input.shape[-1], p=self.noise_order, eps=self.eps).reshape(input.shape)
        return super()._regularizer_exact(input, output, target)
    
    def _regularizer_ema(self, input: Tensor, output: Distribution, target: Tensor) -> Tensor:
        input = input + sample_lp_uniformly(input.shape[:-1].numel(), input.shape[-1], p=self.noise_order, eps=self.eps).reshape(input.shape)
        return super()._regularizer_ema(input, output, target)
    
class EmpiricalFIMTraceRegularizer(AdditiveRegularizer, DataAugmentationRegularizer):
    def __init__(
        self,
        model: Module,
        loss_fn: TrainLoss,
        beta: float =1.0,
        ema_decay: float = 0.99,
        clamp_val = None,
        grad_clamp_val = None,
        reduce: str="mean",
    ):
        """FIM Trace as regularizer. Can be either estimated exactly/via MC estimators. Or using an more efficient ema.

        Args:
            model (Module): Model.
            loss_fn (TrainLoss): Loss fn.
            beta (float, optional): Regularization strength. Defaults to 1.0.
            algorithm (str, optional): Algorithm. Defaults to "jac_exact".
            mc_samples (int, optional): MC samples to use in MC estimations. Defaults to 20.
            ema_decay (float, optional): EMA. Defaults to 0.99.
            reduce (str, optional): _description_. Defaults to "mean".
        """
        super().__init__(model, loss_fn, reduce=reduce)
        self.beta = beta
        # Only required if Fisher information cannot be computed in closed form.
        self.ema = ExponentialMovingAverageEstimator(ema_decay)
        self.clamp_val = clamp_val
        self.grad_clamp_val=grad_clamp_val

        self._regularizer = self._regularizer_ema

    def activate(self, algorithm=None, **kwargs):
        """Activates regularization"""
        if algorithm is not None:
            self.algorithm = algorithm
        self.loss_fn.register_post_loss_regularizer(
            self.__class__.__name__, self.regularizer
        )
        def input_grad_enabled(input, target):
            input.requires_grad_(True)
            return input, target
        self.loss_fn.register_pre_loss_regularizer(self.__class__.__name__,input_grad_enabled)


    def deactivate(self):
        """Deactivate regularization"""
        self.loss_fn.remove_post_loss_regularizer(self.__class__.__name__)
        self.loss_fn.remove_pre_loss_regularizer(self.__class__.__name__)

    def _regularizer_ema(self, input: Tensor, output: Distribution, target: Tensor) -> Tensor:
        """ This implements the regularizer by performing an EMA estimate."""

 
        emprirical_fisher = input.grad**2
        reg = emprirical_fisher.sum(-1)
        mask = torch.isfinite(reg)
        reg = reg[mask]
        reg.clamp(min=0, max=self.clamp_val)
        if self.clamp_val is not None:
            reg = reg.clamp(min=0, max=self.clamp_val)
        reg = self.beta * self._reduce(reg)

        grads = torch.autograd.grad(
            reg,
            self.model.parameters(),  # type: ignore
            retain_graph=True,
        )

        self.ema(grads)

        new_val = self.ema.value
        for params, grad in zip(self.model.parameters(), new_val):
            params.grad = torch.nan_to_num(grad.detach())

        if self.grad_clamp_val is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clamp_val, norm_type=2.0)

        return torch.zeros(1, device=input.device)



class FIMLargestEigenvalueRegularizer(AdditiveRegularizer):
    def __init__(
        self,
        model: Module,
        loss_fn: TrainLoss,
        beta: float=1.0,
        algorithm: str="jac_exact",
        reduce:str="mean",
        mc_samples: int=20,
        max_iters: int = 50,
        **kwargs,
    ):
        """Adds the largest eigenvalue of the FIM as regularizer.

        Args:
            model (Module): Model
            loss_fn (TrainLoss): Loss function
            beta (float, optional): Regularization constant. Defaults to 1.0.
            algorithm (str, optional): Algorithm to estimate the eigenvalue. Defaults to "jac_exact".
            reduce (str, optional): Reduction. Defaults to "mean".
            mc_samples (int, optional): MC sampels to use. Defaults to 20.
            max_iters (int, optional): If power iterations is used, the nmax number of itersations
        """
        super().__init__(model, loss_fn, reduce=reduce)
        self.beta = beta
        self.mc_samples = mc_samples
        self.max_iters = max_iters
        self.algorithm = algorithm

    def _set_algorithm(self, method: str, **kwargs):
        self._algorithm = method
        if method == "jac_exact":
            self._regularizer = self._regularizer_exact
        elif method == "power":
            self._regularizer = self._regularizer_power_iterations
        else:
            raise ValueError("This does not exist")

    def _regularizer_exact(self, input: Tensor, output: Distribution, target: Tensor) -> Tensor:
        """ Eigenvalues computed normally """
        matrix = fisher_information_matrix(self.model,input, output, mc_samples=self.mc_samples)
        max_eigenvalues = torch.linalg.eigvalsh(matrix).max(axis=-1).values

        mask = torch.isfinite(max_eigenvalues)
        max_eigenvalues = max_eigenvalues[mask]

        return self.beta * self._reduce(max_eigenvalues)

    def _regularizer_power_iterations(self, input: Tensor, output: Distribution, target: Tensor) -> Tensor:
        """ Eigenvalues computed """
        matrix = fisher_information_matrix(self.model,input, output, mc_samples=self.mc_samples)
        _, max_eigenvalues = power_iterations(matrix, max_iters=self.max_iters)
        # Momoize last eigenvector as init

        mask = torch.isfinite(max_eigenvalues)
        max_eigenvalues = max_eigenvalues[mask]

        return self.beta * self._reduce(max_eigenvalues)


    # def _regularizer_ema_oja(self, input, output, target):
    #     matrix = monte_carlo_fisher(self.model,input, output, mc_samples=1)
    #     self.oja(matrix)

    #     max_eigenvalues = self.oja.eigenval

    #     mask = torch.isfinite(max_eigenvalues)
    #     max_eigenvalues = max_eigenvalues[mask]

    #     return self.beta * self._reduce(max_eigenvalues)



